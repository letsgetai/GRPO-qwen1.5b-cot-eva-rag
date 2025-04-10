"""
大模型GRPO微调训练脚本

功能：
1. 加载预训练模型（支持4bit量化）
2. 配置LoRA参数进行模型适配
3. 处理训练数据并格式化为对话模板
4. 使用GRPOTrainer进行强化学习微调
5. 集成自定义奖励函数（正确性+格式合规性）
6. 集成SwanLab进行训练监控

主要流程：
1. 模型初始化 -> 2. 数据预处理 -> 3. 奖励函数定义 -> 4. 训练配置 -> 5. 执行训练

依赖：
- unsloth: 加速训练的优化库
- transformers: HuggingFace模型库
- trl: 强化学习工具包
- swanlab: 训练过程可视化
"""

import torch
import re
import json
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from swanlab.integration.transformers import SwanLabCallback
import swanlab

# ==================== 全局配置 ====================
# 模型参数
max_seq_length = 2048  # 支持RoPE内部缩放
dtype = None           # 自动检测数据类型
load_in_4bit = True    # 启用4bit量化节省显存
output_dir = "deep-eval-cot-rag-r1"

# LoRA参数
lora_rank = 64 
lora_alpha = lora_rank * 2

# SwanLab配置
SWANLAB_API_KEY = "Gt8rLkd2zG39K4JPCxNhU"  # 实际使用时应从环境变量读取

# ==================== 模型初始化 ====================
def load_model():
    """加载预训练模型并配置LoRA适配器"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="deep-eval-cot-rag-sft1",
        max_seq_length=max_seq_length,
        fast_inference=True,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        max_lora_rank=lora_rank,
    )

    return FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    ), tokenizer

# ==================== 模板定义 ====================
SYS_PROMPT = '''你是一个严谨的AI助手，需先推理再回答。你将会收到：
1. 问题
2. 正确答案
3. 待评估答案

任务要求：
1. 推理待评估答案的正确性（分主要答案和补充信息）
2. 生成包含评分的JSON结果
3. 思维链放在<think>标签内

<评分标准>
1. 主要答案正确性（0/1）：
   - 1分：关键数据/结论正确（允许表述差异）
   - 0分：关键事实错误/缺失

2. 补充答案正确性（0/1）：
   - 1分：补充信息无错误且相关
   - 0分：存在错误/无关信息
</评分标准>

<输出格式>
{
  "Main_Answer_Accuracy": 0/1,
  "Sub_Answer_Accuracy": 0/1
}
</输出格式>'''

INPUT_TEMPLATE = '''
<问题>
{query}
</问题>

<正确答案>
{answer}
</正确答案>

<待评估答案>
{answer_hat}
</待评估答案>
'''

ANSWER_TEMPLATE = '''
<think>
{think}
</think>

<answer>
{answer}
</answer>
'''

# ==================== 数据处理 ====================
def format_prompt(examples):
    """格式化单条数据为模型输入"""
    return {
        'prompt': [
            {'role': 'system', 'content': SYS_PROMPT.strip()},
            {'role': 'user', 'content': INPUT_TEMPLATE.format(
                query=examples["question"],
                answer=examples["golden_answers"],
                answer_hat=examples["doubao_answer"]
            )}
        ],
        'answer': examples["dsr1_evaluation_results"]
    }

def load_dataset(file_path):
    """加载并预处理JSONL格式数据"""
    processed_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                result = format_prompt(data)
                
                # 数据验证
                if len(result['prompt'][1]['content']) > 1200:
                    print('[跳过] prompt长度超过1200')
                    continue
                    
                json.loads(result['answer'].replace('\n', '').replace('```json', '').replace('```', ''))
                processed_data.append(result)
            except Exception as e:
                print(f'数据处理错误: {str(e)}')
                
    print(f'成功加载 {len(processed_data)} 条数据')
    return processed_data[:-100]  # 保留最后100条作为验证集

# ==================== 奖励函数 ====================
def extract_json(text):
    """从文本中提取JSON内容"""
    try:
        match = re.search(r"<answer>([\s\S]*?)<answer>", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip().replace('\n', ''))
    except Exception:
        pass
    return None

def correctness_reward(prompts, completions, answers):
    """正确性评分（主要答案1.5分 + 补充答案0.5分）"""
    rewards = []
    for completion, answer in zip(completions, answers):
        try:
            resp = extract_json(completion[0]['content'])
            ans = json.loads(answer.replace('\n', '').replace('```json', '').replace('```', ''))
            
            score = 0
            if resp and ans:
                if resp.get('Main_Answer_Accuracy') == ans.get('Main_Answer_Accuracy'):
                    score += 1.5
                if resp.get('Sub_Answer_Accuracy') == ans.get('Sub_Answer_Accuracy'):
                    score += 0.5
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards

def format_reward(completions):
    """格式合规性评分（0.2分制）"""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        match = re.search(r"<think>([\s\S]*?)<think>\s*<answer>([\s\S]*?)<answer>", text, re.DOTALL)
        
        score = 0
        if match:
            # 思维链长度奖励
            if len(match.group(1).strip()) >= 300:
                score += 0.1
            # JSON格式奖励
            try:
                json.loads(match.group(2).strip().replace('\n', ''))
                score += 0.1
            except:
                pass
        rewards.append(score)
    return rewards

# ==================== 训练配置 ====================
def get_train_config():
    """返回GRPO训练配置"""
    return GRPOConfig(
        use_vllm=True,
        vllm_gpu_memory_utilization=0.6,
        learning_rate=3e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.1,
        warmup_ratio=0.1,
        beta=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_generations=32,
        max_prompt_length=max_seq_length,
        max_completion_length=max_seq_length//2,
        num_train_epochs=3,
        save_steps=30,
        max_grad_norm=0.5,
        output_dir=output_dir,
        torch_compile=True
    )

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 1. 初始化
    swanlab.login(api_key=SWANLAB_API_KEY)
    model, tokenizer = load_model()
    
    # 2. 加载数据
    dataset = load_dataset("111right_cot_dsr1.jsonl")
    
    # 3. 配置训练器
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=get_train_config(),
        train_dataset=dataset,
        callbacks=[SwanLabCallback(
            project="deep-eval-r1",
            experiment_name="deep-eval-r1"
        )]
    )
    
    # 4. 开始训练
    trainer.train(resume_from_checkpoint=False)