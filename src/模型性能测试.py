"""
文件功能：大模型生成答案的自动化评估系统

核心流程：
1. 读取包含问题、标准答案和待评估答案的JSONL文件
2. 格式化数据为模型输入提示（包含评分标准）
3. 使用微调后的大模型生成评估结果
4. 验证结果格式并计算评分准确率

输入文件格式要求（JSONL）：
{
    "question": "问题文本",
    "golden_answers": ["标准答案1", "标准答案2"],
    "doubao_answer": "待评估答案",
    "dsr1_evaluation_results": "{\"Main_Answer_Accuracy\":1, \"Sub_Answer_Accuracy\":0}"
}

输出指标：
- 主要答案准确率（Main_Answer_Accuracy）
- 补充答案准确率（Sub_Answer_Accuracy）
- 格式合规率（JSON格式验证）
"""

# 输入输出模板定义
input_format = '''
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

# 系统提示词（包含评分标准）
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


def formatting_prompts_func(examples):
    """格式化单条数据为模型输入"""
    query = examples["question"]
    answer = examples["golden_answers"]
    answer_hat = examples["doubao_answer"]
    outputs = examples["dsr1_evaluation_results"]
    
    input1 = input_format.format(query=query, answer=answer, answer_hat=answer_hat)
    
    return {
        'prompt': [
            {'role': 'system', 'content': SYS.strip()},
            {'role': 'user', 'content': input1}
        ],
        'answer': outputs
    }


def dataset_process(name):
    """处理原始数据集"""
    processed_data = []
    with open(name, "r", encoding="utf-8") as fin:
        for line in fin:    
            data = json.loads(line.strip())
            result = formatting_prompts_func(data)
            
            # 跳过过长的prompt
            if len(result['prompt'][1]['content']) > 1200:
                print('prompt大于1200，跳过')
                continue
                
            try:
                answer = json.loads(result['answer'].replace('\n', '').replace('```json', '').replace('```', ''))
            except:
                print('deepseekr1-结果解析不了,跳过这个数据')
                continue
                
            processed_data.append(result)
            
    print(f'共{len(processed_data)}条数据')
    return processed_data[-100:]  # 返回最后100条测试数据


def extract_answer(response):
    """
    从回复中提取 <answer> 部分并解析为 JSON
    返回: 解析后的字典或None（解析失败时）
    """
    try:
        pattern = r"<answer>([\s\S]*?)<answer>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            answer_part = match.group(1).strip()
            return json.loads(answer_part.replace('\n', '').replace('\r', ''))
    except (json.JSONDecodeError, AttributeError):
        print('response解析错误')
    return None


def judge(response, answer):
    """比较模型输出与标准答案的评分"""
    default_result = {
        "Main_Answer_Accuracy": 0,
        "Sub_Answer_Accuracy": 0
    }
    
    try:
        response = extract_answer(response)
        answer = json.loads(answer.replace('\n', '').replace('```json', '').replace('```', ''))
        
        if not all(isinstance(x, dict) for x in [response, answer]):
            return default_result
            
        # 检查必需字段
        required_fields = ['Main_Answer_Accuracy', 'Sub_Answer_Accuracy']
        if not all(field in response and field in answer for field in required_fields):
            return default_result
            
        return {
            "Main_Answer_Accuracy": int(response['Main_Answer_Accuracy'] == answer['Main_Answer_Accuracy']),
            "Sub_Answer_Accuracy": int(response['Sub_Answer_Accuracy'] == answer['Sub_Answer_Accuracy'])
        }
        
    except Exception as e:
        print(f"评分错误: {str(e)}")
        return default_result


def strict_format_reward_func(completion):
    """验证回复格式是否符合要求"""
    pattern = r"<think>([\s\S]*?)<think>\s*<answer>([\s\S]*?)<answer>"
    match = re.search(pattern, completion, re.DOTALL)
    
    if match:
        try:
            json.loads(match.group(2).strip().replace('\n', ''))
            return 1
        except json.JSONDecodeError:
            pass
    return 0


# 模型加载
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

def load_model():
    """加载微调后的模型"""
    mode_path = 'result/deep-eval-sft-1/checkpoint-60'
    model = AutoPeftModelForCausalLM.from_pretrained(mode_path)
    tokenizer = AutoTokenizer.from_pretrained(mode_path)
    return model.to("cuda").eval(), tokenizer


def process_one(prompt, answer, model, tokenizer):
    """处理单条数据"""
    tokenized_chat = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=1500)
    completion = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    
    correct = judge(completion, answer)
    return (
        correct['Main_Answer_Accuracy'],
        correct['Sub_Answer_Accuracy'],
        strict_format_reward_func(completion)
    )


def main():
    """主执行流程"""
    # 初始化
    model, tokenizer = load_model()
    input_file = '111right_cot_dsr1.jsonl'
    dataset = dataset_process(input_file)
    
    # 指标统计
    metrics = {'main': 0, 'sub': 0, 'format': 0}
    
    # 批量处理
    for item in dataset:
        m, s, f = process_one(item['prompt'], item['answer'], model, tokenizer)
        metrics['main'] += m
        metrics['sub'] += s
        metrics['format'] += f
    
    # 结果输出
    print(f"\n评估结果（共{len(dataset)}条数据）：")
    print(f"主要答案准确率: {metrics['main']/len(dataset):.2%}")
    print(f"补充答案准确率: {metrics['sub']/len(dataset):.2%}")
    print(f"格式合规率: {metrics['format']/len(dataset):.2%}")


if __name__ == "__main__":
    main()