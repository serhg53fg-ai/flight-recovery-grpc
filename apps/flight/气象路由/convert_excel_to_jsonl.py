import pandas as pd
import json

def excel_to_jsonl(excel_file, jsonl_file):
    """
    将Excel文件转换为JSONL格式
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file, engine='openpyxl')

        # 创建JSONL文件
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            # 遍历每一行数据
            for index, row in df.iterrows():
                # 将每行数据转换为字典
                row_dict = row.to_dict()

                # 处理NaN值和特殊数据类型
                for key, value in row_dict.items():
                    if pd.isna(value):
                        row_dict[key] = None
                    elif isinstance(value, pd.Timestamp):
                        row_dict[key] = value.isoformat()
                    elif hasattr(value, 'item'):  # numpy数据类型
                        row_dict[key] = value.item()

                # 将字典转换为JSON字符串并写入文件
                json_line = json.dumps(row_dict, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"转换完成！共处理了 {len(df)} 行数据")
        print(f"JSONL文件已保存为: {jsonl_file}")

    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")

if __name__ == "__main__":
    excel_file = r"/path/to/example-data"
    jsonl_file = r"/path/to/example-data"

    excel_to_jsonl(excel_file, jsonl_file)
