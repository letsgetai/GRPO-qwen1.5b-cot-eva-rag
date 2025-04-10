import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# 配置常量
ARK_API_KEY = '****'
MAX_WORKERS = 25
MAX_TASKS = 1500
MODEL_NAME = "deepseek-r1-250120"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# 初始化OpenAI客户端
client = OpenAI(
    base_url=BASE_URL,
    api_key=ARK_API_KEY,
)

# 提示模板
PROMPT_TEMPLATE = '''<evaluation_rules>
[问题]
{query}
[正确答案]
{answer}
[待评估答案]
{answer_hat}

[评分标准]
1. 主要答案正确性（0/1）：
   - 1分：关键数据/结论正确（允许表述差异）
   - 0分：关键事实错误/缺失

2. 补充答案正确性（0/1）：
   - 1分：补充信息无错误且相关
   - 0分：存在错误/无关信息

[输出格式]
{{
  "Main_Answer_Accuracy": 0/1,
  "Sub_Answer_Accuracy": 0/1
}}

[示例]
问题：python列表是用什么实现的
正确答案：数组和链表
待评估答案：二叉树

二叉树是一种经典的实现列表的方式

[输出]
{{
  "Main_Answer_Accuracy": 0,
  "Sub_Answer_Accuracy": 0
}}
</evaluation_rules>'''


def construct_messages(data: dict) -> list:
    """构造对话消息"""
    return [
        {"role": "system", "content": "你是一个严谨的评分专家"},
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            query=data.get("question", ""),
            answer=data.get("golden_answers", ""),
            answer_hat=data.get("doubao_answer", ""))}
    ]


def process_evaluation(data: dict) -> dict:
    """处理单个评估请求"""
    try:
        messages = construct_messages(data)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5
        )
        
        response = completion.choices[0].message.content
        reasoning = completion.choices[0].message.model_extra['reasoning_content']
        
        print(response)
        result = {**data, "dsr1_reasoning": reasoning, "dsr1_evaluation_results": response}
        return result
    except Exception as e:
        print(f"评估异常：{str(e)}")
        result = {**data, "error": str(e)}
        return result


def process_line(data: dict) -> str:
    """封装为线程安全处理"""
    try:
        result = process_evaluation(data)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        print(f"数据处理异常：{str(e)}")
        return json.dumps({**data, "error": str(e)})


def should_skip_data(data: dict) -> bool:
    """判断是否应该跳过当前数据"""
    if not data.get("golden_answers"):
        return True
    if re.match(r'^how\b', data.get("query", ""), re.I):
        return True
    return False


def main(input_file: str, output_file: str) -> None:
    """主处理流程"""
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            count = 0
            futures = []
            
            for line in fin:
                if count >= MAX_TASKS:
                    break
                    
                data = json.loads(line.strip())
                
                if should_skip_data(data):
                    continue
                
                futures.append(executor.submit(process_line, data))
                count += 1
                
                if count % 100 == 0:
                    print(f"已提交 {count} 条任务")
            
            # 按完成顺序处理结果
            for future in futures:
                try:
                    fout.write(future.result() + "\n")
                except Exception as e:
                    print(f"结果写入异常：{str(e)}")


if __name__ == "__main__":
    main("111right_all_output.jsonl", "111right_cot_dsr1.jsonl")