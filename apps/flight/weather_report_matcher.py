import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

def find_closest_weather_report(airport_code, target_time, weather_df):
    """
    根据机场四字码和目标时间找到最接近的气象报文
    """
    # 筛选出相同机场的数据
    airport_data = weather_df[weather_df['OBCC'] == airport_code].copy()

    if airport_data.empty:
        return None

    # 计算时间差
    airport_data['time_diff'] = abs((airport_data['CSRA_INSERT_DT'] - target_time).dt.total_seconds())

    # 找到时间差最小的记录
    closest_idx = airport_data['time_diff'].idxmin()

    return airport_data.loc[closest_idx, 'REPORT_TEXT']

def load_weather_data():
    """
    加载指定的气象报文数据
    """
    weather_file = r'/path/to/example-data'

    try:
        weather_df = pd.read_csv(weather_file, encoding='utf-8')
        print(f"成功加载: {os.path.basename(weather_file)}")
        print(f"气象数据共 {len(weather_df)} 条记录")
    except Exception as e:
        raise ValueError(f"加载气象数据文件时出错: {e}")

    # 转换时间列为datetime格式
    weather_df['CSRA_INSERT_DT'] = pd.to_datetime(weather_df['CSRA_INSERT_DT'])

    return weather_df

def process_flight_data():
    """
    处理航班数据，添加气象报文
    """
    # 读取航班数据
    try:
        flight_df = pd.read_excel(r'/path/to/example-data')
        print(f"成功加载航班数据，共 {len(flight_df)} 条记录")
    except Exception as e:
        print(f"加载航班数据时出错: {e}")
        return

    # 加载气象数据
    try:
        weather_df = load_weather_data()
    except Exception as e:
        print(f"加载气象数据时出错: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ['计划起飞站四字码', '计划离港时间', '计划到达站四字码', '计划到港时间']
    missing_columns = [col for col in required_columns if col not in flight_df.columns]

    if missing_columns:
        print(f"航班数据中缺少以下列: {missing_columns}")
        print(f"可用列: {list(flight_df.columns)}")
        return

    # 转换时间列为datetime格式
    try:
        flight_df['计划离港时间'] = pd.to_datetime(flight_df['计划离港时间'])
        flight_df['计划到港时间'] = pd.to_datetime(flight_df['计划到港时间'])
    except Exception as e:
        print(f"时间格式转换出错: {e}")
        return

    # 添加起飞站气象报文列
    print("正在匹配起飞站气象报文...")
    flight_df['起飞站气象报文'] = flight_df.apply(
        lambda row: find_closest_weather_report(
            row['计划起飞站四字码'],
            row['计划离港时间'],
            weather_df
        ), axis=1
    )

    # 添加降落站气象报文列
    print("正在匹配降落站气象报文...")
    flight_df['降落站气象报文'] = flight_df.apply(
        lambda row: find_closest_weather_report(
            row['计划到达站四字码'],
            row['计划到港时间'],
            weather_df
        ), axis=1
    )

    # 统计匹配结果
    departure_matched = flight_df['起飞站气象报文'].notna().sum()
    arrival_matched = flight_df['降落站气象报文'].notna().sum()

    print(f"\n匹配结果统计:")
    print(f"起飞站气象报文匹配成功: {departure_matched}/{len(flight_df)} ({departure_matched/len(flight_df)*100:.1f}%)")
    print(f"降落站气象报文匹配成功: {arrival_matched}/{len(flight_df)} ({arrival_matched/len(flight_df)*100:.1f}%)")

    # 保存结果
    output_file = r'/path/to/example-data'
    try:
        flight_df.to_excel(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    return flight_df

if __name__ == "__main__":
    print("开始处理航班数据和气象报文匹配...")
    result_df = process_flight_data()
    print("处理完成！")
