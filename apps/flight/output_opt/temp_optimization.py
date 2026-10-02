import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_and_preprocess_data(file_path):
    """
    加载航班数据并进行全面的预处理。

    Args:
        file_path (str): 航班数据文件的路径 (Excel或CSV)。

    Returns:
        pandas.DataFrame: 经过清洗、格式转换和特征工程后的数据。
    """
    # 根据文件类型读取数据
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        df = pd.read_csv(file_path)

    # 1. 数据清洗和格式转换
    time_cols = ['计划离港时间', '计划到港时间', '实际离港时间', '实际到港时间']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df.dropna(subset=['机尾号'] + time_cols, inplace=True)
    # 筛选掉机尾号为“-”的值
    df = df[df['机尾号'] != '-']
    df['id'] = df.index  # 为每个航班创建唯一ID

    # 2. 计算最短过站时间 (MTT)
    # 计算每个航班的计划过站时间
    df = df.sort_values(by=['机尾号', '计划离港时间'])
    df['前序计划到港时间'] = df.groupby('机尾号')['计划到港时间'].shift(1)
    df['计划过站时间'] = (df['计划离港时间'] - df['前序计划到港时间']).dt.total_seconds() / 60

    # 打印包含机尾号和计划过站时间的部分数据
    print(df[['机尾号', '计划过站时间']].head())

    # 计算每种机型的MTT (15%分位点)
    mtt_map = df.groupby('机型')['计划过站时间'].quantile(0.15).to_dict()
    # 对于数据不足的机型，设置一个默认值，例如45分钟
    default_mtt = 45
    df['mtt'] = df['机型'].map(mtt_map).fillna(default_mtt)
    # 确保MTT不小于一个实际的最小值
    df['mtt'] = df['mtt'].apply(lambda x: max(x, default_mtt))

    # 3. 构建航线网络约束 (串班)
    df = df.sort_values(by=['机尾号', '计划离港时间']).reset_index(drop=True)
    df['id'] = df.index  # 重新生成ID以保证排序后的一致性

    # 使用shift操作高效地找到前后续航班
    df['previous_flight_id'] = df.groupby('机尾号')['id'].shift(1)
    df['next_flight_id'] = df.groupby('机尾号')['id'].shift(-1)

    # 4. 计算优先级所需特征
    # a. 计算航线链长度 (Li)
    # 创建一个从航班ID到其后续航班ID的映射
    next_flight_map = df.set_index('id')['next_flight_id'].to_dict()
    chain_lengths = {}
    for flight_id in df['id']:
        count = 0
        current_id = flight_id
        while pd.notna(next_flight_map.get(current_id)):
            current_id = next_flight_map.get(current_id)
            count += 1
        chain_lengths[flight_id] = count
    df['chain_length'] = df['id'].map(chain_lengths)

    # b. 计算衔接紧迫性 (Ui)
    df['slack_time'] = df['计划过站时间'] - df['mtt']
    # 将负或零的slack_time设为一个很小的值，避免后续计算问题
    df['slack_time'] = df['slack_time'].apply(lambda x: max(x, 1))
    # 对于没有前序航班的，slack_time为NaN，紧迫性为0
    df['urgency'] = 1 / df['slack_time']
    df['urgency'] = df['urgency'].fillna(0)

    print("数据预处理完成。")
    return df


def calculate_priority(flight, is_departure=True):
    """
    计算单个航班的优先级分数。

    Args:
        flight (pandas.Series): 代表单个航班的行。
        is_departure (bool): 当前是否在为离港航班排序。

    Returns:
        float: 综合优先级分数。
    """
    # 权重系数 (可根据策略调整)
    w_type = 1.0  # 航班类型权重
    w_chain = 5.0  # 航线链长度权重
    w_urgency = 10.0  # 衔接紧迫性权重
    w_original_time = 0.001  # 原始计划时间权重 (作为次要排序标准)

    # Ti: 航班类型因子
    type_factor = 1 if is_departure else 0.5  # 示例：离港优先

    # Li: 航线链长度因子
    chain_factor = flight['chain_length']

    # Ui: 衔接紧迫性因子
    urgency_factor = flight['urgency']

    # 原始计划时间因子 (越早越优先)
    # 为了避免数值过大，可以用当天开始的分钟数表示
    time_factor = (flight['计划离港时间'] - flight['计划离港时间'].normalize()).total_seconds() / 60

    # 综合得分，时间因子为负，因为越小（越早）越好
    score = (w_type * type_factor) + \
            (w_chain * chain_factor) + \
            (w_urgency * urgency_factor) - \
            (w_original_time * time_factor)

    return score


def run_simulation(flights_df, start_time_str, end_time_str, capacity_profile, time_slot_minutes=15):
    """
    运行航班调度模拟。

    Args:
        flights_df (pandas.DataFrame): 预处理后的航班数据。
        start_time_str (str): 模拟开始时间 'YYYY-MM-DD HH:MM:SS'。
        end_time_str (str): 模拟结束时间 'YYYY-MM-DD HH:MM:SS'。
        capacity_profile (dict): 时间片容量配置。 e.g., {'departure': 10, 'arrival': 8}
        time_slot_minutes (int): 时间片长度（分钟）。

    Returns:
        pandas.DataFrame: 包含调度结果的航班数据。
    """
    # 初始化
    df = flights_df.copy()
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    # 将ID转换为字典以便快速访问
    df_dict = df.set_index('id').to_dict('index')

    # 动态列初始化
    for flight_id in df_dict:
        df_dict[flight_id]['earliest_departure_time'] = df_dict[flight_id]['计划离港时间']
        df_dict[flight_id]['status'] = 'unassigned'  # unassigned, assigned
        df_dict[flight_id]['new_departure_time'] = None
        df_dict[flight_id]['new_arrival_time'] = None

    # 按时间片循环
    current_time = start_time
    time_slot = timedelta(minutes=time_slot_minutes)

    unassigned_flights = set(df_dict.keys())

    while current_time <= end_time and len(unassigned_flights) > 0:
        print(f"正在处理时间片: {current_time}")

        slot_dep_capacity = capacity_profile['departure']
        slot_arr_capacity = capacity_profile['arrival']  # 当前模型主要处理离港，到港容量暂作保留

        # 筛选当前时间片合格的离港航班
        eligible_departures =[]
        for flight_id in unassigned_flights:
            flight = df_dict[flight_id]
            # 检查是否为离港航班 (本例假设所有航班都从受控机场离港)
            # 并且其最早可起飞时间已到
            if flight['earliest_departure_time'] <= current_time + time_slot:
                eligible_departures.append(flight_id)

        if not eligible_departures:
            current_time += time_slot
            continue

        # 计算优先级并排序
        eligible_departures.sort(
            key=lambda fid: calculate_priority(df_dict[fid], is_departure=True),
            reverse=True
        )

        # 分配容量
        for flight_id in eligible_departures:
            if slot_dep_capacity > 0:
                flight = df_dict[flight_id]

                # 业务规则：不允许提前起飞
                actual_departure_time = max(current_time, flight['计划离港时间'])

                flight['new_departure_time'] = actual_departure_time
                flight['status'] = 'assigned'

                # 计算新的到达时间
                flight_duration = flight['计划到港时间'] - flight['计划离港时间']
                flight['new_arrival_time'] = flight['new_departure_time'] + flight_duration

                unassigned_flights.remove(flight_id)
                slot_dep_capacity -= 1

                # 延误传播
                next_flight_id = flight.get('next_flight_id')
                if pd.notna(next_flight_id) and next_flight_id in df_dict:
                    next_flight = df_dict[next_flight_id]
                    mtt_delta = timedelta(minutes=next_flight['mtt'])

                    # 后续航班的最早起飞时间 = max(原计划起飞时间, 当前航班新到港时间 + MTT)
                    new_earliest_dep_time = flight['new_arrival_time'] + mtt_delta
                    if new_earliest_dep_time > next_flight['earliest_departure_time']:
                        next_flight['earliest_departure_time'] = new_earliest_dep_time

        current_time += time_slot

    # 将字典转回DataFrame并返回
    result_df = pd.DataFrame.from_dict(df_dict, orient='index')
    return result_df


def generate_output_report(results_df):
    """
    计算延误并生成最终的输出报告。
    """
    df = results_df.copy()

    # 计算延误 (分钟)
    # 业务规则：延误 > 15分钟才算延误
    df['delay_minutes'] = (df['new_departure_time'] - df['计划离港时间']).dt.total_seconds() / 60
    df['final_delay'] = df['delay_minutes'].apply(lambda x: x if x > 15 else 0)

    # 计算优先级分数以供分析
    df['priority_score'] = df.apply(lambda row: calculate_priority(row), axis=1)

    # 选择并重命名列以便报告
    output_cols = [
        '航班号', '机尾号', '机型',
        '计划离港时间', 'new_departure_time',
        '计划到港时间', 'new_arrival_time',
        'delay_minutes', 'final_delay',
        'priority_score', 'chain_length', 'urgency'
    ]
    report_df = df[output_cols].rename(columns={
        'new_departure_time': '优化后离港时间',
        'new_arrival_time': '优化后到港时间',
        'delay_minutes': '原始延误(分钟)',
        'final_delay': '有效延误(分钟)'
    })

    # 保存到CSV
    report_df.to_csv("航班优化调度结果.csv", index=False, encoding='utf-8-sig')
    print("优化结果已保存到 '航班优化调度结果.csv'")

    # 打印总结信息
    total_delay = report_df['有效延误(分钟)'].sum()
    delayed_flights_count = (report_df['有效延误(分钟)'] > 0).sum()
    total_flights = len(report_df)

    print("\n--- 优化结果摘要 ---")
    print(f"总航班数: {total_flights}")
    print(f"受影响航班数 (延误>15分钟): {delayed_flights_count}")
    print(f"总延误时间 (分钟): {total_delay:.2f}")
    if delayed_flights_count > 0:
        avg_delay = total_delay / delayed_flights_count
        print(f"平均延误时间 (分钟/每延误航班): {avg_delay:.2f}")
    print("--------------------")


if __name__ == '__main__':
    # --- 配置参数 ---
    DATA_FILE_PATH = r'/path/to/example-data'
    SIM_START_TIME = '2025-05-01 10:03:00'
    SIM_END_TIME = '2025-05-01 15:08:00'

    # 假设一个简化的、全天不变的容量配置
    # 在实际应用中，这应该是一个随时间变化的详细剖面
    CAPACITY_PROFILE = {
        'departure': 10,  # 每15分钟最多允许10个航班离港
        'arrival': 8  # 每15分钟最多允许8个航班到港 (本模型暂未使用)
    }


    processed_flights = load_and_preprocess_data(DATA_FILE_PATH)

    # 2. 运行调度模拟
    results = run_simulation(processed_flights, SIM_START_TIME, SIM_END_TIME, CAPACITY_PROFILE)

    # 3. 生成并保存报告
    if not results.empty:
        generate_output_report(results)
    else:
        print("没有生成任何结果。")
