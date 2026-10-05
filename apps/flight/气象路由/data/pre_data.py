# convert.py
import json
import itertools

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

def build_conversation(plan: dict) -> str:
    """把一条计划信息转成自然语言描述，供<conversation>使用"""
    lines = [
        f"机尾号：{plan['机尾号']}",
        f"航班号：{plan['航班号']}",
        f"机型：{plan['机型']}",
        f"性质：{plan['性质']}",
        f"计划起飞站：{plan['计划起飞站四字码']}（实际{plan['实际起飞站四字码']}）",
        f"计划到达站：{plan['计划到达站四字码']}（实际{plan['实际到达站四字码']}）",
        f"计划离港时间：{plan['计划离港时间']}",
        f"计划到港时间：{plan['计划到港时间']}",
        f"计划地面航程：{plan['计划地面航程_Mile']} Mile",
        f"实际航程：{plan['实际航程_Mile']} Mile",
        f"计划航段时间：{plan['计划航段时间\n（24年同航季平均值）']} 分钟",
        f"实际航段时间：{plan['实际航段时间\n（24年同航季平均值）']} 分钟",
        f"计划起飞数：{plan['计划起飞数']}，计划降落数：{plan['计划降落数']}，计划总流量：{plan['计划总流量']}",
        f"实际起飞数：{plan['实际起飞数']}，实际降落数：{plan['实际降落数']}，实际总流量：{plan['实际总流量']}",
        f"起飞站METAR：{plan['起飞站METAR']}",
        f"到达站METAR：{plan['到达站METAR']}",
    ]
    return "\n".join(lines)

def main():
    samples = []

    with open("pre_data_cleaned.jsonl", "r", encoding="utf-8") as f1, \
         open("pre_time_data.jsonl", "r", encoding="utf-8") as f2:

        for plan_line, actual_line in itertools.zip_longest(f1, f2):
            if not plan_line or not actual_line:
                break  # 防止行数不一致时越界

            plan = json.loads(plan_line.strip())
            actual = json.loads(actual_line.strip())

            conversation = build_conversation(plan)

            instruction = TASK_INSTRUCTION.format(conversation=conversation) + FORMAT_PROMPT

            sample = {
                "instruction": instruction.strip(),
                "input": "",                    # 题目要求 input 留空
                "output": json.dumps(actual, ensure_ascii=False, separators=(",", ":")),
                "system": "你是一个专业的航班预测助手，能够根据用户提供的航班数据准确预测出航班的实际离港时间，实际起飞时间，实际落地时间，实际到港时间。"
            }
            samples.append(sample)

    # 写入训练文件
    with open("train.json", "w", encoding="utf-8") as fout:
        json.dump(samples, fout, ensure_ascii=False, indent=2)

    print(f"转换完成，共生成 {len(samples)} 条样本，已保存至 train.json")

if __name__ == "__main__":
    main()
