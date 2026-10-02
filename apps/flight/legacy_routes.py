"""Preserved page, scenario and scheduling routes; prediction lives in api/predictions.py."""
from flask import Blueprint, current_app as app, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json
import traceback
import glob
from werkzeug.utils import secure_filename
from apps.flight.new_Greedy_algorithm.new_optimization_modify import (
    load_and_preprocess_data, run_simulation, generate_output_report, calculate_priority)
legacy = Blueprint('legacy', __name__)
ALLOWED_EXTENSIONS = {'csv','xlsx','xls'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@legacy.route('/')
def index():
    return render_template('index.html')


@legacy.route('/timeline')
def timeline():
    """24小时时间轴页面"""
    return render_template('timeline.html')


def process_timeline_data(df):
    """处理时间轴数据"""
    try:
        # 确保时间列存在且格式正确
        if '预测实际离港时间' not in df.columns:
            raise Exception('预测数据中缺少预测实际离港时间列')

        # 转换时间格式
        df['预测实际离港时间'] = pd.to_datetime(df['预测实际离港时间'], errors='coerce')
        df['计划离港时间'] = pd.to_datetime(df['计划离港时间'], errors='coerce')
        df['预测实际到港时间'] = pd.to_datetime(df['预测实际到港时间'], errors='coerce')
        df['计划到港时间'] = pd.to_datetime(df['计划到港时间'], errors='coerce')

        # 识别出港和入港航班
        df['是否出港'] = df['起飞站四字码'] == 'ZGGG'
        df['是否入港'] = df['到达站四字码'] == 'ZGGG'

        # 处理出港航班
        departure_df = df[df['是否出港'] & df['预测实际离港时间'].notna() & df['计划离港时间'].notna()].copy()
        departure_df['延误分钟'] = (departure_df['预测实际离港时间'] - departure_df['计划离港时间']).dt.total_seconds() / 60
        departure_df['小时'] = departure_df['预测实际离港时间'].dt.hour

        # 处理入港航班
        arrival_df = df[df['是否入港'] & df['预测实际到港时间'].notna() & df['计划到港时间'].notna()].copy()
        arrival_df['延误分钟'] = (arrival_df['预测实际到港时间'] - arrival_df['计划到港时间']).dt.total_seconds() / 60
        arrival_df['小时'] = arrival_df['预测实际到港时间'].dt.hour

        # 生成24小时数据
        departure_flights = {}
        arrival_flights = {}
        flight_counts = {}
        delay_stats = {}

        for hour in range(24):
            hour_key = f'{hour:02d}:00'

            # 获取该小时的出港和入港航班
            hour_departure_flights = departure_df[departure_df['小时'] == hour]
            hour_arrival_flights = arrival_df[arrival_df['小时'] == hour]

            # 出港航班列表
            departure_list = []
            for _, flight in hour_departure_flights.iterrows():
                departure_list.append({
                    '航班号': flight.get('航班号', 'N/A'),
                    '机尾号': flight.get('机尾号', 'N/A'),
                    '起飞站': flight.get('起飞站四字码', 'N/A'),
                    '到达站': flight.get('到达站四字码', 'N/A'),
                    '预测离港时间': flight['预测实际离港时间'].strftime('%H:%M'),
                    '预测到港时间': flight['预测实际到港时间'].strftime('%H:%M') if pd.notna(flight.get('预测实际到港时间')) else 'N/A',
                    '延误分钟': round(flight['延误分钟'], 1),
                    '航班类型': 'departure'
                })

            # 入港航班列表
            arrival_list = []
            for _, flight in hour_arrival_flights.iterrows():
                arrival_list.append({
                    '航班号': flight.get('航班号', 'N/A'),
                    '机尾号': flight.get('机尾号', 'N/A'),
                    '起飞站': flight.get('起飞站四字码', 'N/A'),
                    '到达站': flight.get('到达站四字码', 'N/A'),
                    '预测离港时间': flight['预测实际离港时间'].strftime('%H:%M') if pd.notna(flight.get('预测实际离港时间')) else 'N/A',
                    '预测到港时间': flight['预测实际到港时间'].strftime('%H:%M'),
                    '延误分钟': round(flight['延误分钟'], 1),
                    '航班类型': 'arrival'
                })

            departure_flights[hour_key] = departure_list
            arrival_flights[hour_key] = arrival_list

            # 航班数量统计
            flight_counts[hour_key] = {
                'departure_count': len(hour_departure_flights),
                'arrival_count': len(hour_arrival_flights),
                'total_count': len(hour_departure_flights) + len(hour_arrival_flights)
            }

            # 延误统计（分别统计出港和入港航班的延误）
            delayed_departure_flights = hour_departure_flights[hour_departure_flights['延误分钟'] > 45]
            delayed_arrival_flights = hour_arrival_flights[hour_arrival_flights['延误分钟'] > 45]

            delay_stats[hour_key] = {
                'departure': {
                    '总航班数': len(hour_departure_flights),
                    '延误航班数': len(delayed_departure_flights),
                    '延误率': round(len(delayed_departure_flights) / len(hour_departure_flights) * 100, 1) if len(hour_departure_flights) > 0 else 0
                },
                'arrival': {
                    '总航班数': len(hour_arrival_flights),
                    '延误航班数': len(delayed_arrival_flights),
                    '延误率': round(len(delayed_arrival_flights) / len(hour_arrival_flights) * 100, 1) if len(hour_arrival_flights) > 0 else 0
                }
            }

        return {
            'departure_flights': departure_flights,
            'arrival_flights': arrival_flights,
            'flight_counts': flight_counts,
            'delay_stats': delay_stats
        }

    except Exception as e:
        print(f"处理时间轴数据失败: {str(e)}")
        raise e


@legacy.route('/scenario')
def scenario():
    """情景设定页面"""
    return render_template('scenario.html')


@legacy.route('/overview')
def overview():
    """航班总览页面"""
    return render_template('overview.html')


@legacy.route('/api/save-scenario', methods=['POST'])
def save_scenario():
    """保存情景设定API"""
    try:
        data = request.get_json()
        scenario_name = data.get('scenario_name', '').strip()
        stop_periods = data.get('stop_periods', [])
        hourly_capacity = data.get('hourly_capacity', 60)  # 新增容量参数

        if not scenario_name:
            return jsonify({'error': '请输入情景名称'}), 400

        # 验证容量参数
        if not isinstance(hourly_capacity, int) or hourly_capacity < 1 or hourly_capacity > 200:
            return jsonify({'error': '每小时容量必须在1-200之间'}), 400

        # 保存情景设定到文件
        scenario_data = {
            'name': scenario_name,
            'stop_periods': stop_periods,
            'hourly_capacity': hourly_capacity,  # 新增容量字段
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        scenario_file = os.path.join(app.config['RESULTS_FOLDER'], f'scenario_{scenario_name}.json')
        with open(scenario_file, 'w', encoding='utf-8') as f:
            json.dump(scenario_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'message': f'情景 "{scenario_name}" 保存成功'
        })

    except Exception as e:
        print(f"保存情景失败: {str(e)}")
        return jsonify({'error': f'保存失败: {str(e)}'}), 500


@legacy.route('/api/get-scenarios')
def get_scenarios():
    """获取所有情景设定API"""
    try:
        scenario_files = glob.glob(os.path.join(app.config['RESULTS_FOLDER'], 'scenario_*.json'))
        scenarios = []

        for file_path in scenario_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    scenario_data = json.load(f)
                    scenarios.append(scenario_data)
            except Exception as e:
                print(f"读取情景文件失败: {file_path}, {str(e)}")
                continue

        return jsonify({
            'success': True,
            'scenarios': scenarios
        })

    except Exception as e:
        print(f"获取情景列表失败: {str(e)}")
        return jsonify({'error': f'获取失败: {str(e)}'}), 500


@legacy.route('/flight_input')
def flight_input():
    """航班信息输入页面"""
    return render_template('flight_input.html')


def apply_stop_periods_with_capacity(df, stop_periods, hourly_capacity=60):
    """应用停止起降时段到航班数据，并根据容量限制重新安排航班"""
    try:
        df_copy = df.copy()

        # 确保时间列存在且格式正确
        df_copy['预测实际离港时间'] = pd.to_datetime(df_copy['预测实际离港时间'], errors='coerce')
        df_copy['计划离港时间'] = pd.to_datetime(df_copy['计划离港时间'], errors='coerce')
        df_copy['预测实际到港时间'] = pd.to_datetime(df_copy['预测实际到港时间'], errors='coerce')
        df_copy['计划到港时间'] = pd.to_datetime(df_copy['计划到港时间'], errors='coerce')

        # 识别出港和入港航班
        df_copy['是否出港'] = df_copy['起飞站四字码'] == 'ZGGG'
        df_copy['是否入港'] = df_copy['到达站四字码'] == 'ZGGG'

        # 收集受影响的航班
        affected_flights = []

        for period in stop_periods:
            start_time = period['start_time']
            end_time = period['end_time']

            # 解析时间
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))

            for idx, row in df_copy.iterrows():
                # 检查出港航班
                if row['是否出港'] and pd.notna(row['预测实际离港时间']):
                    flight_time = row['预测实际离港时间']
                    if is_in_stop_period(flight_time, start_hour, start_minute, end_hour, end_minute):
                        affected_flights.append({
                            'index': idx,
                            'type': 'departure',
                            'original_time': flight_time,
                            'stop_end_hour': end_hour,
                            'stop_end_minute': end_minute
                        })

                # 检查入港航班
                if row['是否入港'] and pd.notna(row['预测实际到港时间']):
                    flight_time = row['预测实际到港时间']
                    if is_in_stop_period(flight_time, start_hour, start_minute, end_hour, end_minute):
                        affected_flights.append({
                            'index': idx,
                            'type': 'arrival',
                            'original_time': flight_time,
                            'stop_end_hour': end_hour,
                            'stop_end_minute': end_minute
                        })

        # 按原始时间排序受影响的航班
        affected_flights.sort(key=lambda x: x['original_time'])

        # 重新安排航班
        reschedule_flights(df_copy, affected_flights, hourly_capacity)

        return df_copy

    except Exception as e:
        print(f"应用停止时段失败: {str(e)}")
        return df


def is_in_stop_period(flight_time, start_hour, start_minute, end_hour, end_minute):
    """检查航班时间是否在停止时段内"""
    flight_hour = flight_time.hour
    flight_minute = flight_time.minute

    flight_minutes_total = flight_hour * 60 + flight_minute
    start_minutes_total = start_hour * 60 + start_minute
    end_minutes_total = end_hour * 60 + end_minute

    # 处理跨天情况
    if end_minutes_total < start_minutes_total:
        # 跨天的停止时段
        return flight_minutes_total >= start_minutes_total or flight_minutes_total <= end_minutes_total
    else:
        # 同一天的停止时段
        return start_minutes_total <= flight_minutes_total <= end_minutes_total


def reschedule_flights(df, affected_flights, hourly_capacity):
    """重新安排受影响的航班"""
    # 统计每小时的正常航班数量
    hourly_counts = {}

    # 统计正常航班（未受停止起降影响的航班）
    affected_indices = {flight['index'] for flight in affected_flights}

    for idx, row in df.iterrows():
        if idx in affected_indices:
            continue

        # 统计出港航班
        if row['是否出港'] and pd.notna(row['预测实际离港时间']):
            hour = row['预测实际离港时间'].hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        # 统计入港航班
        if row['是否入港'] and pd.notna(row['预测实际到港时间']):
            hour = row['预测实际到港时间'].hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

    # 重新安排受影响的航班
    for flight in affected_flights:
        idx = flight['index']
        flight_type = flight['type']
        stop_end_hour = flight['stop_end_hour']
        stop_end_minute = flight['stop_end_minute']

        # 从停止时段结束后开始寻找可用时段
        target_hour = stop_end_hour
        target_minute = stop_end_minute

        # 寻找有容量的时段
        while True:
            current_count = hourly_counts.get(target_hour, 0)

            if current_count < hourly_capacity:
                # 找到有容量的时段
                new_time = flight['original_time'].replace(
                    hour=target_hour,
                    minute=target_minute
                )

                # 如果新时间早于原时间，说明跨天了
                if new_time < flight['original_time']:
                    new_time = new_time + pd.Timedelta(days=1)

                # 更新航班时间
                if flight_type == 'departure':
                    df.at[idx, '预测实际离港时间'] = new_time
                    # 同时更新到港时间（假设飞行时间不变）
                    if pd.notna(df.at[idx, '预测实际到港时间']) and pd.notna(df.at[idx, '预测实际离港时间']):
                        flight_duration = df.at[idx, '预测实际到港时间'] - flight['original_time']
                        df.at[idx, '预测实际到港时间'] = new_time + flight_duration
                else:  # arrival
                    df.at[idx, '预测实际到港时间'] = new_time

                # 更新该时段的计数
                hourly_counts[target_hour] = current_count + 1
                break
            else:
                # 当前时段已满，移到下一个小时
                target_hour = (target_hour + 1) % 24
                target_minute = 0  # 整点开始


@legacy.route('/api/delete-scenario', methods=['DELETE'])
def delete_scenario():
    """删除情景设定API"""
    try:
        data = request.get_json()
        scenario_name = data.get('scenario_name', '').strip()

        if not scenario_name:
            return jsonify({'error': '请提供情景名称'}), 400

        # 构建情景文件路径
        scenario_file = os.path.join(app.config['RESULTS_FOLDER'], f'scenario_{scenario_name}.json')

        # 检查文件是否存在
        if not os.path.exists(scenario_file):
            return jsonify({'error': f'情景 "{scenario_name}" 不存在'}), 404

        # 删除文件
        os.remove(scenario_file)

        return jsonify({
            'success': True,
            'message': f'情景 "{scenario_name}" 删除成功'
        })

    except Exception as e:
        print(f"删除情景失败: {str(e)}")
        return jsonify({'error': f'删除失败: {str(e)}'}), 500


@legacy.route('/api/upload-optimization-file', methods=['POST'])
def upload_optimization_file():
    """上传航班优化调度的CSV文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if file and file.filename.lower().endswith('.csv'):
            # 生成唯一文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_input_{timestamp}.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath
            })
        else:
            return jsonify({'error': '请上传CSV格式文件'}), 400

    except Exception as e:
        print(f"文件上传失败: {str(e)}")
        return jsonify({'error': f'文件上传失败: {str(e)}'}), 500


@legacy.route('/api/run-optimization', methods=['POST'])
def run_optimization():
    """运行航班优化调度 - 使用贪心算法"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        start_time_raw = data.get('start_time')
        end_time_raw = data.get('end_time')
        departure_capacity = data.get('departure_capacity', 10)
        arrival_capacity = data.get('arrival_capacity', 8)

        # 处理HTML datetime-local格式转换为标准格式
        try:
            if 'T' in start_time_raw:
                start_time = start_time_raw.replace('T', ' ') + ':00'
            else:
                start_time = start_time_raw

            if 'T' in end_time_raw:
                end_time = end_time_raw.replace('T', ' ') + ':00'
            else:
                end_time = end_time_raw
        except Exception as e:
            return jsonify({'error': f'时间格式错误: {str(e)}'}), 400

        if not all([filename, start_time, end_time]):
            return jsonify({'error': '缺少必要参数'}), 400

        # 构建文件路径
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_filepath):
            return jsonify({'error': '输入文件不存在'}), 404

        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'航班优化调度结果_{timestamp}.csv'
        output_filepath = os.path.join(app.config['OUTPUT_OPT_FOLDER'], output_filename)

        # 使用new_optimization_modify.py的贪心算法
        try:
            print(f"开始使用贪心算法进行航班优化调度...")

            # 1. 加载和预处理数据
            processed_flights = load_and_preprocess_data(input_filepath)
            print(f"数据预处理完成，共处理 {len(processed_flights)} 个航班")

            # 2. 构建容量配置
            capacity_profile = {
                'departure': departure_capacity,
                'arrival': arrival_capacity
            }

            # 3. 运行调度模拟
            results = run_simulation(
                processed_flights,
                start_time,
                end_time,
                capacity_profile
            )

            if results.empty:
                return jsonify({'error': '优化结果为空，请检查输入数据和时间范围'}), 400

            # 4. 生成输出报告并保存到指定位置
            # 直接处理结果数据，避免路径问题
            df = results.copy()

            # 计算延误 (分钟)
            df['delay_minutes'] = (df['new_departure_time'] - df['计划离港时间']).dt.total_seconds() / 60
            df['final_delay'] = df['delay_minutes'].apply(lambda x: x if x > 15 else 0)

            # 计算优先级分数
            df['priority_score'] = df.apply(lambda row: calculate_priority(row), axis=1)

            # 选择并重命名列
            output_cols = [
                '航班号', '机尾号', '机型',
                '计划离港时间', 'new_departure_time',
                '计划到港时间', 'new_arrival_time',
                'delay_minutes', 'final_delay',
                'priority_score', 'chain_length', 'urgency'
            ]

            # 确保所有列都存在
            available_cols = [col for col in output_cols if col in df.columns]
            report_df = df[available_cols].rename(columns={
                'new_departure_time': '优化后离港时间',
                'new_arrival_time': '优化后到港时间',
                'delay_minutes': '原始延误(分钟)',
                'final_delay': '有效延误(分钟)'
            })

            # 保存主要结果到指定路径
            report_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
            print(f"优化结果已保存到 '{output_filepath}'")

            # 生成延误航班列表文件
            delayed_flights_df = report_df[report_df['有效延误(分钟)'] > 0].copy() if '有效延误(分钟)' in report_df.columns else pd.DataFrame()
            delayed_filename = f'延误航班列表_{timestamp}.csv'
            delayed_filepath = os.path.join(app.config['OUTPUT_OPT_FOLDER'], delayed_filename)

            if not delayed_flights_df.empty:
                delayed_flights_df.to_csv(delayed_filepath, index=False, encoding='utf-8-sig')
                print(f"延误航班列表已保存到 '{delayed_filepath}'")
            else:
                print("没有延误航班，未生成延误航班列表文件。")

            # 计算统计信息
            total_delay = report_df['有效延误(分钟)'].sum() if '有效延误(分钟)' in report_df.columns else 0
            delayed_flights_count = (report_df['有效延误(分钟)'] > 0).sum() if '有效延误(分钟)' in report_df.columns else 0
            total_flights = len(report_df)

            print(f"优化完成: 总航班数 {total_flights}, 延误航班数 {delayed_flights_count}, 总延误时间 {total_delay:.2f}分钟")

            return jsonify({
                'success': True,
                'message': '航班优化调度完成',
                'output_filename': output_filename,
                'delayed_flights_filename': delayed_filename if not delayed_flights_df.empty else None,
                'processed_flights': total_flights,
                'delayed_flights': int(delayed_flights_count),
                'total_delay_minutes': float(total_delay),
                'optimization_period': f'{start_time} 至 {end_time}',
                'capacity_limits': {
                    'departure': departure_capacity,
                    'arrival': arrival_capacity
                }
            })

        except Exception as e:
            print(f"优化处理失败: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'优化处理失败: {str(e)}'}), 500

    except Exception as e:
        print(f"运行优化失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'运行优化失败: {str(e)}'}), 500


@legacy.route('/api/get-optimization-result/<filename>')
def get_optimization_result(filename):
    """获取优化结果数据"""
    try:
        filepath = os.path.join(app.config['OUTPUT_OPT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '结果文件不存在'}), 404

        # 读取CSV文件
        df = pd.read_csv(filepath)

        # 转换为JSON格式
        result_data = df.to_dict('records')

        return jsonify({
            'success': True,
            'data': result_data,
            'columns': list(df.columns),
            'total_rows': len(df)
        })

    except Exception as e:
        print(f"获取优化结果失败: {str(e)}")
        return jsonify({'error': f'获取优化结果失败: {str(e)}'}), 500


@legacy.route('/api/download-optimization-result/<filename>')
def download_optimization_result(filename):
    """下载优化结果文件"""
    try:
        filepath = os.path.join(app.config['OUTPUT_OPT_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500
