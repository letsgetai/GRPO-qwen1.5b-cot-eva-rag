"""
大模型问答生成
通过 translate_text() 函数调用 ​豆包大模型（doubao-1-5-lite）​ API
使用结构化提示词模板 (prompt_template) 要求模型：
​首行输出核心答案​（以「结论：」开头 + <关键数据> 格式）
​补充延伸信息​（背景/数据来源等，不超过200字）
示例：输入问题 "金渐层是什么猫？"，输出结构化答案
"""
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

ARK_API_KEY = ''

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
)

prompt_template = '''<task role="应答器">
[问题输入]
{sentence}

[输出要求]
    ​**核心答案**​（首行加粗）
- 必须直接回答问题关键点
- 使用「结论：」开头+<关键数据>格式

    ​**延伸信息**
- 补充相关背景/数据来源/注意事项/用户想知道的其他信息/
- 延伸信息与核心答案之间需要换行
- 不超过200字

[示例]
问题：金渐层是什么猫？
结论：金渐层是一种猫<英国短毛猫的稀有色品种>

金渐层是由英短蓝猫和金吉拉猫交配改良产生的英国短毛猫的稀有色品种。它的底层绒毛为浅蜜黄色到亮杏黄色，背部、两肋、头部和尾巴上的被毛的毛尖被充分地染成黑色，外观呈现出闪烁的金色的特征。其性格温柔、对人友善，在家庭中能与人和其他宠物和谐相处。
</task>'''


def translate_text(text):
    """调用模型进行翻译"""
    if not text:
        return text
    
    messages = [
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": prompt_template.format(sentence=text)}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="doubao-1-5-lite-32k-250115",
            messages=messages,
            temperature=0.3  # 保持翻译稳定性
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # 失败时返回原文


def process_line(data):
    """处理单行JSON数据"""
    try:
        original_question = data.get("question", "")
        
        # 并行翻译两个字段
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_q = executor.submit(translate_text, original_question)
            data["doubao_answer"] = future_q.result()
            
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None


def main(input_file, output_file):
    """主处理流程"""
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=100) as executor:
            count = 0
            max_count = 1500
            futures = []
            
            for line in fin:
                line = line.strip()
                if line:
                    if count >= max_count:
                        break
                    
                    data = json.loads(line)
                    if not data.get("golden_answers"):
                        continue
                        
                    question = data.get("question", "")
                    future = executor.submit(process_line, data)
                    futures.append(future)

            # 保持输出顺序与输入一致
            for future in futures:
                result = future.result()
                if result:
                    fout.write(result + "\n")
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"已处理 {count} 条数据")
                    
                    if count >= max_count:
                        print("达到最大处理数量，终止任务")
                        executor.shutdown(wait=False)
                        break


if __name__ == "__main__":
    main("all-output.jsonl", "right_all_output.jsonl")