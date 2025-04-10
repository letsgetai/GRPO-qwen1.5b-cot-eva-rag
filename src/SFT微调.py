"""
大模型微调训练脚本

功能：
1. 加载预训练模型（支持Qwen2.5系列）
2. 配置LoRA参数进行模型适配
3. 处理训练数据并格式化为对话模板
4. 使用SFTTrainer进行监督微调
5. 集成SwanLab进行训练监控

依赖：
- unsloth: 加速训练的优化库
- transformers: HuggingFace模型库
- trl: 监督微调工具包
- swanlab: 训练过程可视化
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from swanlab.integration.transformers import SwanLabCallback
import swanlab
import json

# ==================== 模型配置 ====================
max_seq_length = 2048  # 支持RoPE内部缩放
dtype = None           # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = False   # 禁用4bit量化（如需节省内存可设为True）
output_dir = "./result/deep-eval-sft-1"

# 系统提示词模板（定义评估标准）
SYS = '''你是一个严谨的AI助手，需先推理再回答，你将会收到一组问题、正确答案、待评估答案，你的任务是根据问题和正确答案推理待评估答案的主要部分和补充部分的正确性，并生成一个json，你应该一步一步推理，你的推理应该放在<think>和</think>中间。
<评分标准>
1. 主要答案正确性（0/1）：
   - 1分：关键数据/结论正确（允许表述差异）
   - 0分：关键事实错误/缺失

2. 补充答案正确性（0/1）：
   - 1分：补充信息无错误且相关
   - 0分：存在错误/无关信息
<评分标准>
<输出格式>
{{
  "Main_Answer_Accuracy": 0/1,
  "Sub_Answer_Accuracy": 0/1
}}
<输出格式>'''

# 输入输出模板
input_prompt = '''
<问题>
{query}
<问题>

<正确答案>
{answer}
<正确答案>

<待评估答案>
{answer_hat}
<待评估答案>
'''

answer_prompt ='''
<think>
{think}
<think>

<answer>
{answer}
<answer>
'''

# ==================== 数据预处理 ====================
def formatting_prompts_func(examples):
    """将原始数据格式化为对话模板"""
    return {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": input_prompt.format(
                query=examples["question"],
                answer=examples["golden_answers"],
                answer_hat=examples["doubao_answer"]
            )}, 
            {"role": "assistant", "content": answer_prompt.format(
                think=examples["dsr1_reasoning"],
                answer=examples["dsr1_evaluation_results"]
            )}
        ]
    }

def dataset_process(name):
    """加载并预处理JSONL格式的训练数据"""
    processed_data = []
    with open(name, "r", encoding="utf-8") as fin:
        for line in fin:    
            data = json.loads(line.strip())
            processed_data.append(formatting_prompts_func(data))
    return Dataset.from_dict({
        "messages": [item["messages"] for item in processed_data[:-100]]  # 保留最后100条作为验证集
    })

# ==================== 模型初始化 ====================
def load_model():
    """加载预训练模型并配置LoRA"""
    # 初始化基础模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="model-no-4bit",  # 可替换为Qwen2.5系列模型
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        gpu_memory_utilization=0.4
    )
    
    # 添加LoRA适配器
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA秩
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 节省30%显存
        random_state=3407,
    )
    return model, tokenizer

# ==================== 训练配置 ====================
def get_trainer(model, tokenizer, train_dataset):
    """配置训练参数"""
    swanlab.login(api_key="Gt8rLkd2zG39K4JPCxNhU")  # 替换为实际API Key
    
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # 短序列训练可设为True加速5x
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,  # 训练60步
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
        callbacks=[SwanLabCallback(
            project="deep-eval-sft",
            experiment_name="deep-eval-sft-1"
        )]
    )

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 1. 加载模型
    model, tokenizer = load_model()
    
    # 2. 准备数据
    input_file = '111right_cot_dsr1.jsonl'
    train_dataset = dataset_process(input_file)
    
    # 3. 开始训练
    trainer = get_trainer(model, tokenizer, train_dataset)
    trainer.train()