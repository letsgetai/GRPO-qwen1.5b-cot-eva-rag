system_prompt = '''你是一个严谨的AI助手，需先推理再回答，你将会收到一组问题、正确答案、待评估答案，你的任务是根据问题和正确答案推理待评估答案的主要部分和补充部分的正确性，并生成一个json，你应该一步一步推理，你的推理应该放在<think>和</think>中间。
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
input_='''
<问题>
方大同还在嘛
<问题>

<正确答案>
方大同于2025年2月21日去世。
<正确答案>

<待评估答案>
方大同已经于2025年2月21日去世。他是一位美国夏威夷出生的中国香港男歌手、作曲家和制作人，以其独特的音乐风格和多才多艺的才华在华语乐坛享有盛誉。方大同曾推出多张专辑，并获得多项音乐奖项。他的突然离世在音乐界和粉丝中引起了广泛的哀悼和回忆。
<待评估答案>
'''

prompt=[{'role': 'system', 'content': system_prompt.strip()},
        {'role': 'user', 'content': input_}]

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
base_model_path = os.getenv('BASE_MODEL_PATH', 'deep-eval-cot-rag-sft1')
model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = model.to("cuda")
tokenized_chat = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,  # 为生成阶段添加提示
    return_tensors="pt"
).to(model.device)
outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=1500,temperature=0.3)#记得加上温度系数一类的东西
completion = tokenizer.batch_decode(outputs[:,tokenized_chat.shape[1]:].detach().cpu().numpy(), skip_special_tokens=True)[0]
print(completion)


# <think>
# 好的，我现在需要评估用户提供的关于方大同是否还活着的信息。首先，我得确认问题中的“方大同”指的是哪位艺人。通常，“方大同”可能是指台湾歌手方文山，但这里的问题可能是指其他艺人。

# 正确答案指出方大同于2025年2月21日去世。而待评估的答案提到他已于2025年2月21日去世，这与正确答案一致。因此，主要答案正确性应该是1分，因为关键数据没有错误。

# 接下来检查补充答案。待评估答案提到了方大同的身份、国籍、生平成就以及去世时间等信息，这些都准确无误。此外，补充信息如“华语乐坛享有盛誉”、“多张专辑”等都是正确的。所以补充答案正确性也应该给1分。

# 总结下来，主答案正确性为1分，补充答案正确性也为1分。

# <think>

# <answer>


# {
#   "Main_Answer_Accuracy": 1,
#   "Sub_Answer_Accuracy": 1
# }
# <answer>