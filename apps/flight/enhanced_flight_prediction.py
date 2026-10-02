import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# GPU配置 - 改进版
print("可用GPU设备:")
print(tf.config.list_physical_devices('GPU'))

# 检查TensorFlow版本和GPU支持
print(f"TensorFlow版本: {tf.__version__}")
print(f"是否构建了GPU支持: {tf.test.is_built_with_cuda()}")
print(f"是否有GPU可用: {tf.test.is_gpu_available()}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 指定使用第二张显卡（索引为1）
        if len(gpus) > 1:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            print(f"使用GPU: {gpus[1]}")
        else:
            print("使用第一张GPU")
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("未检测到GPU设备，将使用CPU训练")
    print("如果您有GPU，请检查：")
    print("1. NVIDIA驱动是否正确安装")
    print("2. CUDA工具包是否安装")
    print("3. TensorFlow是否为GPU版本")
    print("4. 运行 'python check_cuda_gpu.py' 进行详细诊断")

class METARParser:
    """METAR报文解析器"""

    def parse_wind(self, metar_text):
        """解析风向风速信息"""
        wind_pattern = r'(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT( (\d{3})V(\d{3}))?'
        match = re.search(wind_pattern, metar_text)

        if match:
            wind_dir = 0 if match.group(1) == 'VRB' else int(match.group(1))
            wind_speed = int(match.group(2))
            wind_gust = int(match.group(4)) if match.group(4) else wind_speed
            wind_var_from = int(match.group(6)) if match.group(6) else wind_dir
            wind_var_to = int(match.group(7)) if match.group(7) else wind_dir
        else:
            wind_dir = wind_speed = wind_gust = wind_var_from = wind_var_to = 0

        return wind_dir, wind_speed, wind_gust, wind_var_from, wind_var_to

    def parse_visibility(self, metar_text):
        """解析能见度信息"""
        if 'CAVOK' in metar_text:
            return 10000, 1  # visibility, cavok

        vis_pattern = r'\b(\d{4})\b'
        match = re.search(vis_pattern, metar_text)
        visibility = int(match.group(1)) if match else 9999

        return visibility, 0

    def parse_weather_phenomena(self, metar_text):
        """解析天气现象"""
        has_cb = 1 if 'CB' in metar_text else 0
        return has_cb

    def parse_temperature_dewpoint(self, metar_text):
        """解析温度露点"""
        temp_pattern = r'(M?\d{2})/(M?\d{2})'
        match = re.search(temp_pattern, metar_text)

        if match:
            temp_str = match.group(1)
            dew_str = match.group(2)

            temperature = -int(temp_str[1:]) if temp_str.startswith('M') else int(temp_str)
            dewpoint = -int(dew_str[1:]) if dew_str.startswith('M') else int(dew_str)
        else:
            temperature = dewpoint = 0

        return temperature, dewpoint

    def parse_pressure(self, metar_text):
        """解析气压信息"""
        pressure_pattern = r'A(\d{4})'
        match = re.search(pressure_pattern, metar_text)

        if match:
            # 转换为hPa
            pressure_inhg = int(match.group(1)) / 100
            pressure_hpa = pressure_inhg * 33.8639
        else:
            pressure_hpa = 1013.25

        return pressure_hpa

    def parse_metar(self, metar_text):
        """解析完整的METAR报文"""
        if pd.isna(metar_text) or not isinstance(metar_text, str):
            return {
                'wind_dir': 0, 'wind_speed': 0, 'wind_gust': 0,
                'wind_var_from': 0, 'wind_var_to': 0, 'visibility': 9999,
                'cavok': 0, 'has_CB': 0, 'temperature': 15, 'dewpoint': 10,
                'pressure': 1013.25
            }

        wind_dir, wind_speed, wind_gust, wind_var_from, wind_var_to = self.parse_wind(metar_text)
        visibility, cavok = self.parse_visibility(metar_text)
        has_cb = self.parse_weather_phenomena(metar_text)
        temperature, dewpoint = self.parse_temperature_dewpoint(metar_text)
        pressure = self.parse_pressure(metar_text)

        return {
            'wind_dir': wind_dir,
            'wind_speed': wind_speed,
            'wind_gust': wind_gust,
            'wind_var_from': wind_var_from,
            'wind_var_to': wind_var_to,
            'visibility': visibility,
            'cavok': cavok,
            'has_CB': has_cb,
            'temperature': temperature,
            'dewpoint': dewpoint,
            'pressure': pressure
        }

class EnhancedFlightPredictionModel:
    def __init__(self, excel_file_path):
        self.excel_file_path = excel_file_path
        self.data = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = ['实际离港时间', '实际起飞时间', '实际落地时间', '实际到港时间']
        self.metar_parser = METARParser()

        print(f"当前使用的设备: {tf.config.list_logical_devices()}")

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        try:
            # 指定引擎来读取Excel文件
            self.data = pd.read_excel(self.excel_file_path, engine='openpyxl')
            print(f"数据加载成功，共有 {len(self.data)} 行数据")
            print(f"列名: {list(self.data.columns)}")

            # 解析METAR数据
            self.parse_metar_data()

            # 定义特征列
            self.feature_columns = [
                '机尾号', '航班号', '机型', '性质',
                '计划起飞站四字码', '实际起飞站四字码', '计划到达站四字码', '实际到达站四字码',
                '计划离港时间', '计划到港时间',
                '计划地面航程_Mile', '实际航程_Mile',
                '计划航段时间\n（24年同航季平均值）', '实际航段时间\n（24年同航季平均值）',
                '计划起飞数', '计划降落数', '计划总流量', '实际起飞数', '实际降落数', '实际总流量',
                # 起飞站天气特征
                'wind_dir_s', 'wind_speed_s', 'wind_gust_s', 'wind_var_from_s', 'wind_var_to_s',
                'visibility_s', 'cavok_s', 'has_CB_s', 'temperature_s', 'dewpoint_s', 'pressure_s',
                # 到达站天气特征
                'wind_dir_e', 'wind_speed_e', 'wind_gust_e', 'wind_var_from_e', 'wind_var_to_e',
                'visibility_e', 'cavok_e', 'has_CB_e', 'temperature_e', 'dewpoint_e', 'pressure_e'
            ]

            # 检查可用特征
            available_features = [col for col in self.feature_columns if col in self.data.columns]
            missing_features = [col for col in self.feature_columns if col not in self.data.columns]

            print(f"\n可用特征列数量: {len(available_features)}")
            if missing_features:
                print(f"缺失特征列: {missing_features}")

            self.feature_columns = available_features

            # 检查目标列
            available_targets = [col for col in self.target_columns if col in self.data.columns]
            missing_targets = [col for col in self.target_columns if col not in self.data.columns]

            print(f"\n可用目标列: {available_targets}")
            if missing_targets:
                print(f"缺失目标列: {missing_targets}")

            self.target_columns = available_targets

            return True

        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False

    def parse_metar_data(self):
        """解析METAR数据并提取气象特征"""
        print("正在解析METAR数据...")

        # 解析起飞站METAR
        if '起飞站METAR' in self.data.columns:
            print("解析起飞站METAR...")
            for idx, metar in self.data['起飞站METAR'].items():
                weather_data = self.metar_parser.parse_metar(metar)
                for key, value in weather_data.items():
                    col_name = f"{key}_s"
                    if col_name not in self.data.columns:
                        self.data[col_name] = np.nan
                    self.data.at[idx, col_name] = value

        # 解析到达站METAR
        if '到达站METAR' in self.data.columns:
            print("解析到达站METAR...")
            for idx, metar in self.data['到达站METAR'].items():
                weather_data = self.metar_parser.parse_metar(metar)
                for key, value in weather_data.items():
                    col_name = f"{key}_e"
                    if col_name not in self.data.columns:
                        self.data[col_name] = np.nan
                    self.data.at[idx, col_name] = value

        print("METAR数据解析完成")

    def preprocess_features(self):
        """预处理特征数据"""
        print("正在进行特征预处理...")

        # 创建高级特征
        enhanced_data = self.create_enhanced_features(self.data)

        # 处理分类变量
        categorical_columns = ['机尾号', '航班号', '机型', '性质',
                             '计划起飞站四字码', '实际起飞站四字码',
                             '计划到达站四字码', '实际到达站四字码']

        for col in categorical_columns:
            if col in enhanced_data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    enhanced_data[col] = self.label_encoders[col].fit_transform(
                        enhanced_data[col].astype(str).fillna('Unknown')
                    )
                else:
                    enhanced_data[col] = self.label_encoders[col].transform(
                        enhanced_data[col].astype(str).fillna('Unknown')
                    )

        # 处理时间列
        time_columns = ['计划离港时间', '计划到港时间']
        for col in time_columns:
            if col in enhanced_data.columns:
                try:
                    enhanced_data[col] = pd.to_datetime(enhanced_data[col], errors='coerce')
                    enhanced_data[col] = enhanced_data[col].dt.hour + enhanced_data[col].dt.minute/60
                except:
                    enhanced_data[col] = enhanced_data[col].fillna(0)

        # 处理数值列
        numeric_columns = [col for col in self.feature_columns
                          if col not in categorical_columns and col not in time_columns]

        for col in numeric_columns:
            if col in enhanced_data.columns:
                enhanced_data[col] = pd.to_numeric(enhanced_data[col], errors='coerce')
                enhanced_data[col] = enhanced_data[col].fillna(enhanced_data[col].median())

        # 添加新创建的特征
        new_features = ['temp_dewpoint_diff_s', 'wind_pressure_interaction_s',
                       'temp_dewpoint_diff_e', 'wind_pressure_interaction_e',
                       'route_efficiency', 'congestion_ratio', 'weather_risk_score_s', 'weather_risk_score_e']

        for feature in new_features:
            if feature in enhanced_data.columns and feature not in self.feature_columns:
                self.feature_columns.append(feature)

        # 选择特征列
        available_features = [col for col in self.feature_columns if col in enhanced_data.columns]
        X = enhanced_data[available_features]

        # 处理异常值
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower_bound, upper_bound)

        # 标准化
        if 'features' not in self.scalers:
            self.scalers['features'] = RobustScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)

        print(f"特征预处理完成，最终特征维度: {X_scaled.shape}")
        return X_scaled

    def create_enhanced_features(self, data):
        """创建增强的特征工程"""
        enhanced_data = data.copy()

        # 时间特征
        time_columns = ['计划离港时间', '计划到港时间']
        for col in time_columns:
            if col in enhanced_data.columns:
                time_data = pd.to_datetime(enhanced_data[col], errors='coerce')
                enhanced_data[f'{col}_hour'] = time_data.dt.hour
                enhanced_data[f'{col}_minute'] = time_data.dt.minute
                enhanced_data[f'{col}_weekday'] = time_data.dt.weekday
                enhanced_data[f'{col}_is_weekend'] = (time_data.dt.weekday >= 5).astype(int)
                enhanced_data[f'{col}_is_peak'] = ((time_data.dt.hour >= 7) & (time_data.dt.hour <= 9) |
                                             (time_data.dt.hour >= 17) & (time_data.dt.hour <= 19)).astype(int)

        # 航班时长特征
        if '计划离港时间' in enhanced_data.columns and '计划到港时间' in enhanced_data.columns:
            planned_dep = pd.to_datetime(enhanced_data['计划离港时间'], errors='coerce')
            planned_arr = pd.to_datetime(enhanced_data['计划到港时间'], errors='coerce')
            enhanced_data['planned_flight_duration'] = (planned_arr - planned_dep).dt.total_seconds() / 3600

        # 天气综合指数
        if all(col in enhanced_data.columns for col in ['wind_speed_s', 'visibility_s', 'temperature_s']):
            enhanced_data['weather_severity_s'] = (
                (enhanced_data['wind_speed_s'] / 50) * 0.3 +  # 标准化风速
                (1 / (enhanced_data['visibility_s'] / 1000 + 1)) * 0.4 +  # 能见度倒数
                (abs(enhanced_data['temperature_s']) / 40) * 0.3  # 极端温度
            )

        # 机场拥堵程度
        if '实际总流量' in enhanced_data.columns:
            enhanced_data['traffic_density'] = enhanced_data['实际总流量'] / enhanced_data['实际总流量'].rolling(window=10, min_periods=1).mean()

        # 航线繁忙程度
        route_cols = ['计划起飞站四字码', '计划到达站四字码']
        if all(col in enhanced_data.columns for col in route_cols):
            enhanced_data['route'] = enhanced_data['计划起飞站四字码'] + '_' + enhanced_data['计划到达站四字码']
            route_counts = enhanced_data['route'].value_counts()
            enhanced_data['route_frequency'] = enhanced_data['route'].map(route_counts)

        return enhanced_data

    def preprocess_targets_improved(self):
        """改进的目标变量预处理"""
        processed_targets = {}

        for target in self.target_columns:
            if target in self.data.columns:
                try:
                    target_data = pd.to_datetime(self.data[target], errors='coerce')

                    # 计算与计划时间的差异（分钟）而不是绝对时间戳
                    if target == '实际离港时间' and '计划离港时间' in self.data.columns:
                        planned_time = pd.to_datetime(self.data['计划离港时间'], errors='coerce')
                        target_values = (target_data - planned_time).dt.total_seconds() / 60
                    elif target == '实际到港时间' and '计划到港时间' in self.data.columns:
                        planned_time = pd.to_datetime(self.data['计划到港时间'], errors='coerce')
                        target_values = (target_data - planned_time).dt.total_seconds() / 60
                    else:
                        # 提取时间特征：小时 + 分钟/60
                        target_values = target_data.dt.hour + target_data.dt.minute/60

                    # 处理异常值
                    target_values = target_values.fillna(target_values.median())
                    Q1 = target_values.quantile(0.25)
                    Q3 = target_values.quantile(0.75)
                    IQR = Q3 - Q1
                    target_values = target_values.clip(Q1 - 3*IQR, Q3 + 3*IQR)

                    # 使用RobustScaler而不是StandardScaler
                    if target not in self.scalers:
                        self.scalers[target] = RobustScaler()
                        target_scaled = self.scalers[target].fit_transform(target_values.values.reshape(-1, 1)).flatten()
                    else:
                        target_scaled = self.scalers[target].transform(target_values.values.reshape(-1, 1)).flatten()

                    processed_targets[target] = target_scaled

                except Exception as e:
                    print(f"处理目标变量 {target} 时出错: {str(e)}")
                    continue

        return processed_targets

    def create_improved_model(self, input_dim, output_name=None):
        """创建改进的神经网络模型"""
        if output_name:
            print(f"正在为 {output_name} 创建改进模型...")

        # 使用更简单但更有效的架构
        model = keras.Sequential([
            # 输入层
            layers.Dense(256, activation='swish', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # 第二层
            layers.Dense(128, activation='swish'),
            layers.BatchNormalization(),
            layers.Dropout(0.15),

            # 第三层
            layers.Dense(64, activation='swish'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),

            # 输出层
            layers.Dense(1, activation='linear')
        ])

        # 使用更好的优化器配置
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.001,
            beta_1=0.9,
            beta_2=0.999
        )

        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )

        return model

    def plot_prediction_analysis(self, y_test, y_pred, target_name):
        """绘制预测分析图"""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{target_name} - 预测vs实际值')
            plt.savefig(f'prediction_analysis_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"预测分析图已保存: prediction_analysis_{target_name}.png")
        except Exception as e:
            print(f"绘制预测分析图时出错: {str(e)}")

    def train_models_improved(self, test_size=0.2, validation_split=0.2, epochs=150, batch_size=64):
        """改进的模型训练方法"""
        print("\n开始改进的模型训练...")

        # 预处理数据
        X = self.preprocess_features()
        y_dict = self.preprocess_targets_improved()  # 使用改进的目标处理

        # 存储所有模型的评估结果
        all_evaluation_results = {}
        all_test_data = {}  # 存储测试数据用于样本展示
        original_test_data = {}  # 存储原始测试数据
        all_valid_indices = {}  # 存储每个目标的有效索引
        test_indices_mapping = {}  # 存储测试集索引映射

        # 为每个目标变量训练模型
        for target_name, y in y_dict.items():
            print(f"\n训练 {target_name} 预测模型...")

            # 数据清理
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[valid_indices]
            y_clean = y[valid_indices]

            # 存储有效索引
            all_valid_indices[target_name] = valid_indices

            # 分割数据
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X_clean, y_clean, np.where(valid_indices)[0],
                test_size=test_size, random_state=42
            )

            # 存储测试集索引映射（只需要存储一次）
            if not test_indices_mapping:
                test_indices_mapping['original_indices'] = test_indices

            print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

            # 创建模型
            model = self.create_improved_model(X_train.shape[1], target_name)

            # 定义回调函数
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,  # 更温和的学习率衰减
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    f'best_model_{target_name}.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
            ]

            # 训练模型
            history = model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            self.models[target_name] = model

            # 评估改进
            y_pred = model.predict(X_test)

            # 反标准化预测结果和真实值
            y_test_original = self.scalers[target_name].inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.scalers[target_name].inverse_transform(y_pred).flatten()

            # 计算分钟误差（基于反标准化后的值）
            if target_name in ['实际离港时间', '实际到港时间']:
                # 离港/到港时间：逆变换结果已经是分钟数
                minute_errors = np.abs(y_test_original - y_pred_original)
            else:
                # 起飞/落地时间：逆变换结果是小时数，需要转换为分钟数
                minute_errors = np.abs(y_test_original - y_pred_original) * 60

            mean_minute_error = np.mean(minute_errors)

            # 计算更详细的评估指标
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 存储评估结果
            all_evaluation_results[target_name] = {
                'MAE': mae,
                'RMSE': np.sqrt(mse),
                'R²': r2,
                'Mean_Minute_Error': mean_minute_error,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_test_original': y_test_original,
                'y_pred_original': y_pred_original
            }

            # 绘制预测vs实际值散点图
            self.plot_prediction_analysis(y_test, y_pred, target_name)

            # 绘制训练历史
            self.plot_training_history(history, target_name)

        # 统一打印所有模型的评估结果
        print("\n" + "="*80)
        print("所有模型评估结果汇总")
        print("="*80)

        for target_name, results in all_evaluation_results.items():
            print(f"\n{target_name} 模型评估结果:")
            print(f"  MAE: {results['MAE']:.4f}")
            print(f"  RMSE: {results['RMSE']:.4f}")
            print(f"  R²: {results['R²']:.4f}")
            print(f"  平均误差(分钟): {results['Mean_Minute_Error']:.2f}")

        # 从测试集中抽取3组数据展示预测结果
        print("\n" + "="*80)
        print("测试集样本预测结果展示（随机抽取3组）")
        print("="*80)

        # 随机选择3个样本索引
        first_target = list(all_evaluation_results.keys())[0]
        test_size = len(all_evaluation_results[first_target]['y_test'])
        sample_indices = np.random.choice(test_size, size=min(3, test_size), replace=False)

        for i, idx in enumerate(sample_indices, 1):
            print(f"\n样本 {i} (测试集索引: {idx}):")
            print("-" * 80)

            # 获取原始数据索引
            original_idx = test_indices_mapping['original_indices'][idx]

            # 先打印完整的原始数据信息
            print("原始数据信息:")
            sample_data = self.data.iloc[original_idx]

            # 打印主要特征信息
            feature_columns_to_show = [
                '机尾号', '航班号', '机型', '性质',
                '计划起飞站四字码', '实际起飞站四字码', '计划到达站四字码', '实际到达站四字码',
                '计划离港时间', '计划到港时间', '实际离港时间', '实际起飞时间', '实际落地时间', '实际到港时间',
                '计划地面航程_Mile', '实际航程_Mile',
                '计划航段时间\n（24年同航季平均值）', '实际航段时间\n（24年同航季平均值）',
                '计划起飞数', '计划降落数', '计划总流量', '实际起飞数', '实际降落数', '实际总流量',
                '起飞站METAR', '到达站METAR'
            ]

            for col in feature_columns_to_show:
                if col in sample_data.index:
                    value = sample_data[col]
                    if pd.notna(value):
                        if col in ['计划离港时间', '计划到港时间', '实际离港时间', '实际起飞时间', '实际落地时间', '实际到港时间']:
                            if isinstance(value, str):
                                print(f"  {col}: {value}")
                            else:
                                print(f"  {col}: {pd.to_datetime(value).strftime('%Y/%m/%d %H:%M')}")
                        else:
                            print(f"  {col}: {value}")
                    else:
                        print(f"  {col}: 无数据")

            print("\n" + "-" * 50)

            # 显示真实时间（包含年月日）
            print("真实时间:")
            for target_name in self.target_columns:
                if target_name in sample_data:
                    real_time = sample_data[target_name]
                    if pd.notna(real_time):
                        if isinstance(real_time, str):
                            print(f"  {target_name}: {real_time}")
                        else:
                            print(f"  {target_name}: {pd.to_datetime(real_time).strftime('%Y/%m/%d %H:%M')}")
                    else:
                        print(f"  {target_name}: 无数据")

            print("\n预测时间:")
            for target_name, results in all_evaluation_results.items():
                pred_value = results['y_pred_original'][idx]

                if target_name in ['实际离港时间', '实际到港时间']:
                    # 对于离港和到港时间，pred_value是延误分钟数
                    base_time_col = '计划离港时间' if target_name == '实际离港时间' else '计划到港时间'
                    base_time = pd.to_datetime(sample_data[base_time_col])
                    pred_time = base_time + pd.Timedelta(minutes=pred_value)

                    # 显示预测时间和延误/提前信息
                    if pred_value >= 0:
                        print(f"  {target_name}: {pred_time.strftime('%Y/%m/%d %H:%M')} (延误 {pred_value:.1f} 分钟)")
                    else:
                        print(f"  {target_name}: {pred_time.strftime('%Y/%m/%d %H:%M')} (提前 {abs(pred_value):.1f} 分钟)")
                else:
                    # 对于起飞和落地时间，pred_value是小时数（小时+分钟/60）
                    if pred_value >= 0:
                        total_minutes = int(pred_value * 60)  # 转换为总分钟数
                        hours = total_minutes // 60
                        minutes = total_minutes % 60

                        # 获取基准日期（使用计划离港时间的日期）
                        base_date = pd.to_datetime(sample_data['计划离港时间']).date()

                        # 处理跨日期情况
                        if hours >= 24:
                            days_to_add = hours // 24
                            hours = hours % 24
                            pred_date = base_date + pd.Timedelta(days=days_to_add)
                        else:
                            pred_date = base_date

                        pred_time_str = f"{pred_date.strftime('%Y/%m/%d')} {hours:02d}:{minutes:02d}"
                        print(f"  {target_name}: {pred_time_str}")
                    else:
                        print(f"  {target_name}: 无效时间 ({pred_value:.1f})")

            print("\n误差:")
            for target_name, results in all_evaluation_results.items():
                real_value = results['y_test_original'][idx]
                pred_value = results['y_pred_original'][idx]

                if target_name in ['实际离港时间', '实际到港时间']:
                    # 延误时间的误差，单位：分钟
                    error_minutes = abs(real_value - pred_value)
                    print(f"  {target_name}: {error_minutes:.1f}分钟")
                else:
                    # 时间的误差，转换为分钟
                    error_hours = abs(real_value - pred_value)
                    error_minutes = error_hours * 60
                    print(f"  {target_name}: {error_minutes:.1f}分钟")

        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)

        return True

    def plot_training_history(self, history, target_name):
        """绘制训练历史"""
        try:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='训练损失')
            plt.plot(history.history['val_loss'], label='验证损失')
            plt.title(f'{target_name} - 损失函数')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='训练MAE')
            plt.plot(history.history['val_mae'], label='验证MAE')
            plt.title(f'{target_name} - 平均绝对误差')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'training_history_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"训练历史图已保存: training_history_{target_name}.png")
        except Exception as e:
            print(f"绘制训练历史时出错: {str(e)}")

    def predict(self, input_data):
        """使用训练好的模型进行预测"""
        if not self.models:
            print("没有训练好的模型，请先训练模型")
            return None

        # 预处理输入数据
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # 应用相同的预处理步骤
        processed_input = self.preprocess_single_input(input_df)

        predictions = {}
        for target_name, model in self.models.items():
            pred_scaled = model.predict(processed_input)
            pred_original = self.scalers[target_name].inverse_transform(pred_scaled)
            predictions[target_name] = pred_original.flatten()

        return predictions

    def save_models(self, model_dir='models'):
        """保存训练好的模型"""
        import os
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for target_name, model in self.models.items():
            model.save(f'{model_dir}/model_{target_name}.h5')

        # 保存预处理器
        import pickle
        with open(f'{model_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        with open(f'{model_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)

        print(f"模型已保存到 {model_dir} 目录")

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = EnhancedFlightPredictionModel(r'/path/to/example-data')

    # 加载和预处理数据
    # 在主程序部分，将第603行的调用改为：
    if model.load_and_preprocess_data():
    # 训练模型 - 使用改进版方法
        success = model.train_models_improved(  # 改为调用改进版方法
            test_size=0.2,
            validation_split=0.2,
            epochs=150,  # 减少训练轮数
            batch_size=64   # 减少批次大小
            )

        if success:
                # 保存模型
            model.save_models()
            print("\n训练完成！模型已保存。")
        else:
            print("训练失败")
