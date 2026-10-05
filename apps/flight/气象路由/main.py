import json
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "models/qwen3-1"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置填充令牌以避免警告
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

TASK_INSTRUCTION = """
你是一个航班预测的助手，你的任务是根据用户的问题，预测用户的航班信息。
<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
你的任务是根据<conversation></conversation> XML标签中的的航班相关信息，预测航班的"实际离港时间"，"实际起飞时间"，"实际落地时间"，"实际到港时间"这四个量。请按照以下指示操作：
1. 你必须重复分析<conversation></conversation> XML标签中的航班相关信息，预测出该航班的"实际离港时间"，"实际起飞时间"，"实际落地时间"，"实际到港时间"这四个量。
2. 你必须综合考虑所给的机尾号、航班号、机型、计划离港时间、计划到港时间、METAR气象报文、流量、航距等信息。
3. 你必须根据所给的信息，预测出该航班的"实际离港时间"，"实际起飞时间"，"实际落地时间"，"实际到港时间"这四个量。

根据你的分析，如果你已经预测出了时间，请按照以下JSON格式提供你的响应：
{"实际离港时间": "2023-08-01T10:00:00", "实际起飞时间": "2023-08-01T10:15:00", "实际落地时间": "2023-08-01T11:00:00", "实际到港时间": "2023-08-01T11:15:00"}
"""


# Helper function to create the system prompt for our model
def format_prompt(
     conversation: List[Dict[str, Any]]
):
    return (
        TASK_INSTRUCTION.format(
            conversation=json.dumps(conversation, ensure_ascii=False, indent=2)
        )
        + FORMAT_PROMPT
    )

# Define conversations

conversation = [
    {
        "role": "user",
        "content": '{"机尾号": "B20E8", "航班号": 8065, "机型": "78Z", "性质": "J", "计划起飞站四字码": "ZGGG", "实际起飞站四字码": "ZGGG", "计划到达站四字码": "LTFM", "实际到达站四字码": "LTFM", "计划离港时间": "2025-05-02T00:10:00", "计划到港时间": "2025-05-02T11:15:00", "计划地面航程_Mile": 4802, "实际航程_Mile": 4635.0834, "计划航段时间\n（24年同航季平均值）": 665.0, "实际航段时间\n（24年同航季平均值）": 660.0, "计划起飞数": 4, "计划降落数": 16, "计划总流量": 20, "实际起飞数": 5, "实际降落数": 19, "实际总流量": 24, "起飞站METAR": "METAR ZGGG 020010Z 04001KT 020V060 6000 CLR 26/24 A0298 RMK SLP216=", "到达站METAR": "METAR LTFM 021115Z 34505KT 305V185 9000 CLR 15/10 A0300 RMK SLP216="}'
    }
]
route_prompt = format_prompt(conversation)

messages = [
    {"role": "user", "content": route_prompt},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

# 创建注意力掩码
attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

# 2. Generate
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,  # 减少到合理的长度
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)

# 3. Strip the prompt from each sequence
prompt_lengths = input_ids.shape[1]  # same length for every row here
generated_only = [
    output_ids[prompt_lengths:]  # slice off the prompt tokens
    for output_ids in generated_ids
]

# 4. Decode if you want text
response = tokenizer.batch_decode(generated_only, skip_special_tokens=True)[0]

print("=" * 50)
print("航班时间预测结果:")
print("=" * 50)
print(response)
print("=" * 50)
