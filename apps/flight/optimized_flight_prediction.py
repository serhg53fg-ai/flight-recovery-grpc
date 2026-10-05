import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb  # 移除xgboost依赖
# import lightgbm as lgb  # 移除lightgbm依赖
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta
import warnings
import pickle
import os
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# GPU配置
print("可用GPU设备:")
print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) > 1:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            print(f"使用GPU: {gpus[1]}")
        else:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("使用CPU训练")

class EnhancedMETARParser:
    """增强的METAR报文解析器"""

    def __init__(self):
        self.weather_phenomena = {
            'RA': 1, 'SN': 2, 'FG': 3, 'BR': 4, 'HZ': 5, 'DZ': 6,
            'TS': 7, 'SH': 8, 'FZ': 9, 'GR': 10, 'GS': 11
        }

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

        # 计算风向变化范围
        wind_variability = abs(wind_var_to - wind_var_from) if wind_var_to != wind_var_from else 0

        # 计算阵风因子
        gust_factor = wind_gust / max(wind_speed, 1)

        return {
            'wind_dir': wind_dir,
            'wind_speed': wind_speed,
            'wind_gust': wind_gust,
            'wind_var_from': wind_var_from,
            'wind_var_to': wind_var_to,
            'wind_variability': wind_variability,
            'gust_factor': gust_factor
        }

    def parse_visibility(self, metar_text):
        """解析能见度信息"""
        if 'CAVOK' in metar_text:
            return {'visibility': 10000, 'cavok': 1, 'vis_category': 4}

        vis_pattern = r'\b(\d{4})\b'
        match = re.search(vis_pattern, metar_text)
        visibility = int(match.group(1)) if match else 9999

        # 能见度等级分类
        if visibility >= 10000:
            vis_category = 4  # 优秀
        elif visibility >= 5000:
            vis_category = 3  # 良好
        elif visibility >= 1500:
            vis_category = 2  # 一般
        else:
            vis_category = 1  # 较差

        return {'visibility': visibility, 'cavok': 0, 'vis_category': vis_category}

    def parse_weather_phenomena(self, metar_text):
        """解析天气现象"""
        phenomena_count = 0
        severity_score = 0

        for phenomenon, score in self.weather_phenomena.items():
            if phenomenon in metar_text:
                phenomena_count += 1
                severity_score += score

        has_cb = 1 if 'CB' in metar_text else 0
        has_ts = 1 if 'TS' in metar_text else 0

        return {
            'phenomena_count': phenomena_count,
            'severity_score': severity_score,
            'has_CB': has_cb,
            'has_TS': has_ts
        }

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

        # 计算温露点差
        temp_dewpoint_spread = temperature - dewpoint

        # 计算相对湿度（近似）
        relative_humidity = max(0, min(100, 100 - 5 * temp_dewpoint_spread))

        return {
            'temperature': temperature,
            'dewpoint': dewpoint,
            'temp_dewpoint_spread': temp_dewpoint_spread,
            'relative_humidity': relative_humidity
        }

    def parse_pressure(self, metar_text):
        """解析气压信息"""
        pressure_pattern = r'A(\d{4})'
        match = re.search(pressure_pattern, metar_text)

        if match:
            pressure_inhg = int(match.group(1)) / 100
            pressure_hpa = pressure_inhg * 33.8639
        else:
            pressure_hpa = 1013.25

        # 气压等级分类
        if pressure_hpa >= 1020:
            pressure_category = 3  # 高压
        elif pressure_hpa >= 1000:
            pressure_category = 2  # 正常
        else:
            pressure_category = 1  # 低压

        return {'pressure': pressure_hpa, 'pressure_category': pressure_category}

    def parse_metar(self, metar_text):
        """解析完整的METAR报文"""
        if pd.isna(metar_text) or not isinstance(metar_text, str):
            return {
                'wind_dir': 0, 'wind_speed': 0, 'wind_gust': 0,
                'wind_var_from': 0, 'wind_var_to': 0, 'wind_variability': 0,
                'gust_factor': 1, 'visibility': 9999, 'cavok': 0, 'vis_category': 4,
                'phenomena_count': 0, 'severity_score': 0, 'has_CB': 0, 'has_TS': 0,
                'temperature': 15, 'dewpoint': 10, 'temp_dewpoint_spread': 5,
                'relative_humidity': 75, 'pressure': 1013.25, 'pressure_category': 2
            }

        result = {}
        result.update(self.parse_wind(metar_text))
        result.update(self.parse_visibility(metar_text))
        result.update(self.parse_weather_phenomena(metar_text))
        result.update(self.parse_temperature_dewpoint(metar_text))
        result.update(self.parse_pressure(metar_text))

        # 计算综合天气风险评分
        weather_risk = (
            (result['wind_speed'] / 50) * 0.2 +
            (1 / (result['visibility'] / 1000 + 1)) * 0.3 +
            (result['severity_score'] / 10) * 0.2 +
            (abs(result['temperature']) / 40) * 0.1 +
            (result['gust_factor'] - 1) * 0.2
        )
        result['weather_risk'] = min(1.0, weather_risk)

        return result

class AttentionLayer(layers.Layer):
    """自注意力层"""

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W_q = layers.Dense(units)
        self.W_k = layers.Dense(units)
        self.W_v = layers.Dense(units)
        self.dense = layers.Dense(units)

    def call(self, inputs):
        # 计算注意力权重
        q = self.W_q(inputs)
        k = self.W_k(inputs)
        v = self.W_v(inputs)

        # 计算注意力分数
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # 应用注意力权重
        attended = tf.matmul(attention_weights, v)
        output = self.dense(attended)

        return output

class OptimizedFlightPredictionModel:
    def __init__(self, excel_file_path):
        self.excel_file_path = excel_file_path
        self.data = None
        self.models = {}
        self.ensemble_models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = ['实际离港时间', '实际起飞时间', '实际落地时间', '实际到港时间']
        self.metar_parser = EnhancedMETARParser()
        self.feature_importance = {}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        try:
            # 根据文件扩展名选择读取方式
            if self.excel_file_path.lower().endswith('.csv'):
                self.data = pd.read_csv(self.excel_file_path, encoding='utf-8')
            else:
                # 指定引擎来读取Excel文件
                self.data = pd.read_excel(self.excel_file_path, engine='openpyxl')
            print(f"数据加载成功，共有 {len(self.data)} 行数据")

            # 解析METAR数据
            self.parse_metar_data()

            # 定义基础特征列
            self.feature_columns = [
                '机尾号', '航班号', '机型', '性质',
                '计划起飞站四字码', '实际起飞站四字码', '计划到达站四字码', '实际到达站四字码',
                '计划离港时间', '计划到港时间',
                '计划地面航程_Mile', '实际航程_Mile',
                '计划航段时间\n（24年同航季平均值）', '实际航段时间\n（24年同航季平均值）',
                '计划起飞数', '计划降落数', '计划总流量', '实际起飞数', '实际降落数', '实际总流量'
            ]

            # 添加天气特征
            weather_features = [
                'wind_dir', 'wind_speed', 'wind_gust', 'wind_variability', 'gust_factor',
                'visibility', 'vis_category', 'phenomena_count', 'severity_score',
                'has_CB', 'has_TS', 'temperature', 'dewpoint', 'temp_dewpoint_spread',
                'relative_humidity', 'pressure', 'pressure_category', 'weather_risk'
            ]

            for suffix in ['_s', '_e']:
                self.feature_columns.extend([f + suffix for f in weather_features])

            # 检查可用特征
            available_features = [col for col in self.feature_columns if col in self.data.columns]
            self.feature_columns = available_features

            print(f"可用特征列数量: {len(available_features)}")
            return True

        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False

    def parse_metar_data(self):
        """解析METAR数据并提取气象特征"""
        print("正在解析METAR数据...")

        try:
            # 解析起飞站METAR
            if '起飞站METAR' in self.data.columns:
                for idx, metar in self.data['起飞站METAR'].items():
                    if pd.notna(metar):
                        weather_data = self.metar_parser.parse_metar(metar)
                        for key, value in weather_data.items():
                            col_name = f"{key}_s"
                            if col_name not in self.data.columns:
                                self.data[col_name] = np.nan
                            self.data.at[idx, col_name] = value
            else:
                print("警告：未找到起飞站METAR列，跳过气象数据解析")

            # 解析到达站METAR
            if '到达站METAR' in self.data.columns:
                for idx, metar in self.data['到达站METAR'].items():
                    if pd.notna(metar):
                        weather_data = self.metar_parser.parse_metar(metar)
                        for key, value in weather_data.items():
                            col_name = f"{key}_e"
                            if col_name not in self.data.columns:
                                self.data[col_name] = np.nan
                            self.data.at[idx, col_name] = value
            else:
                print("警告：未找到到达站METAR列，跳过气象数据解析")

            print("METAR数据解析完成")
        except Exception as e:
            print(f"METAR数据解析失败: {str(e)}")
            print("继续处理其他数据...")

    def create_advanced_features(self, data):
        """创建高级特征工程"""
        enhanced_data = data.copy()

        # 时间特征增强
        time_columns = ['计划离港时间', '计划到港时间']
        for col in time_columns:
            if col in enhanced_data.columns:
                time_data = pd.to_datetime(enhanced_data[col], errors='coerce')
                enhanced_data[f'{col}_hour'] = time_data.dt.hour
                enhanced_data[f'{col}_minute'] = time_data.dt.minute
                enhanced_data[f'{col}_weekday'] = time_data.dt.weekday
                enhanced_data[f'{col}_month'] = time_data.dt.month
                enhanced_data[f'{col}_quarter'] = time_data.dt.quarter
                enhanced_data[f'{col}_is_weekend'] = (time_data.dt.weekday >= 5).astype(int)
                enhanced_data[f'{col}_is_holiday'] = self._is_holiday(time_data)
                enhanced_data[f'{col}_is_peak'] = self._is_peak_hour(time_data.dt.hour)
                enhanced_data[f'{col}_season'] = self._get_season(time_data.dt.month)

        # 航班时长和效率特征
        if '计划离港时间' in enhanced_data.columns and '计划到港时间' in enhanced_data.columns:
            planned_dep = pd.to_datetime(enhanced_data['计划离港时间'], errors='coerce')
            planned_arr = pd.to_datetime(enhanced_data['计划到港时间'], errors='coerce')
            enhanced_data['planned_flight_duration'] = (planned_arr - planned_dep).dt.total_seconds() / 3600

            # 航班效率指标
            if '计划地面航程_Mile' in enhanced_data.columns:
                enhanced_data['flight_speed'] = enhanced_data['计划地面航程_Mile'] / enhanced_data['planned_flight_duration']

        # 天气差异特征
        weather_features = ['temperature', 'pressure', 'wind_speed', 'visibility', 'weather_risk']
        for feature in weather_features:
            if f'{feature}_s' in enhanced_data.columns and f'{feature}_e' in enhanced_data.columns:
                enhanced_data[f'{feature}_diff'] = enhanced_data[f'{feature}_e'] - enhanced_data[f'{feature}_s']
                enhanced_data[f'{feature}_avg'] = (enhanced_data[f'{feature}_s'] + enhanced_data[f'{feature}_e']) / 2

        # 机场拥堵和流量特征
        if '实际总流量' in enhanced_data.columns:
            enhanced_data['traffic_density'] = enhanced_data['实际总流量'] / enhanced_data['实际总流量'].rolling(window=10, min_periods=1).mean()
            enhanced_data['traffic_pressure'] = enhanced_data['实际总流量'] / enhanced_data['计划总流量']

        # 航线特征
        route_cols = ['计划起飞站四字码', '计划到达站四字码']
        if all(col in enhanced_data.columns for col in route_cols):
            enhanced_data['route'] = enhanced_data['计划起飞站四字码'] + '_' + enhanced_data['计划到达站四字码']
            route_counts = enhanced_data['route'].value_counts()
            enhanced_data['route_frequency'] = enhanced_data['route'].map(route_counts)
            enhanced_data['route_popularity'] = enhanced_data['route_frequency'] / len(enhanced_data)

        # 机型特征
        if '机型' in enhanced_data.columns:
            aircraft_counts = enhanced_data['机型'].value_counts()
            enhanced_data['aircraft_frequency'] = enhanced_data['机型'].map(aircraft_counts)

        return enhanced_data

    def _is_holiday(self, time_data):
        """判断是否为节假日（简化版）"""
        # 这里可以添加更复杂的节假日判断逻辑
        return ((time_data.dt.month == 1) & (time_data.dt.day == 1) |  # 元旦
                (time_data.dt.month == 5) & (time_data.dt.day == 1) |  # 劳动节
                (time_data.dt.month == 10) & (time_data.dt.day == 1)).astype(int)  # 国庆节

    def _is_peak_hour(self, hour):
        """判断是否为高峰时段"""
        return ((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 19)).astype(int)

    def _get_season(self, month):
        """获取季节"""
        if month in [12, 1, 2]:
            return 0  # 冬季
        elif month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        else:
            return 3  # 秋季

    def preprocess_features(self):
        """预处理特征数据"""
        print("正在进行特征预处理...")

        # 创建高级特征
        enhanced_data = self.create_advanced_features(self.data)

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

        # 添加新创建的特征到特征列表
        new_features = [col for col in enhanced_data.columns
                       if col not in self.feature_columns and col not in self.target_columns]
        self.feature_columns.extend(new_features)

        # 选择特征列
        available_features = [col for col in self.feature_columns if col in enhanced_data.columns]
        X = enhanced_data[available_features]

        # 处理数值列
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

            # 处理异常值
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

    def preprocess_targets_improved(self):
        """改进的目标变量预处理"""
        processed_targets = {}

        for target in self.target_columns:
            if target in self.data.columns:
                try:
                    target_data = pd.to_datetime(self.data[target], errors='coerce')

                    # 计算与计划时间的差异（分钟）
                    if target == '实际离港时间' and '计划离港时间' in self.data.columns:
                        planned_time = pd.to_datetime(self.data['计划离港时间'], errors='coerce')
                        target_values = (target_data - planned_time).dt.total_seconds() / 60
                    elif target == '实际到港时间' and '计划到港时间' in self.data.columns:
                        planned_time = pd.to_datetime(self.data['计划到港时间'], errors='coerce')
                        target_values = (target_data - planned_time).dt.total_seconds() / 60
                    else:
                        target_values = target_data.dt.hour + target_data.dt.minute/60

                    # 处理异常值
                    target_values = target_values.fillna(target_values.median())
                    Q1 = target_values.quantile(0.25)
                    Q3 = target_values.quantile(0.75)
                    IQR = Q3 - Q1
                    target_values = target_values.clip(Q1 - 3*IQR, Q3 + 3*IQR)

                    # 标准化
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

    def create_advanced_neural_network(self, input_dim, output_name=None):
        """创建高级神经网络模型"""
        inputs = layers.Input(shape=(input_dim,))

        # 第一个残差块
        x1 = layers.Dense(512, activation='swish')(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)

        x2 = layers.Dense(512, activation='swish')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.2)(x2)

        # 残差连接
        x2 = layers.Add()([x1, x2])

        # 注意力层
        x3 = AttentionLayer(256)(tf.expand_dims(x2, axis=1))
        x3 = tf.squeeze(x3, axis=1)

        # 第二个残差块
        x4 = layers.Dense(256, activation='swish')(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Dropout(0.2)(x4)

        x5 = layers.Dense(256, activation='swish')(x4)
        x5 = layers.BatchNormalization()(x5)
        x5 = layers.Dropout(0.1)(x5)

        # 残差连接
        x5 = layers.Add()([x4, x5])

        # 输出层
        x6 = layers.Dense(128, activation='swish')(x5)
        x6 = layers.BatchNormalization()(x6)
        x6 = layers.Dropout(0.1)(x6)

        outputs = layers.Dense(1, activation='linear')(x6)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # 使用学习率调度
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
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

    def create_ensemble_model(self, X_train, y_train, target_name):
        """创建集成模型"""
        models = {}

        # XGBoost模型已移除（缺少依赖）

        # LightGBM模型已移除（缺少依赖）

        # 随机森林模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        models['random_forest'] = rf_model.fit(X_train, y_train)

        return models

    def weighted_loss(self, y_true, y_pred):
        """加权损失函数"""
        # 对延误较大的样本给予更高权重
        weights = tf.where(tf.abs(y_true) > 1.0, 2.0, 1.0)
        loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
        return loss

    def train_models_optimized(self, test_size=0.2, validation_split=0.2, epochs=200, batch_size=32):
        """优化的模型训练方法"""
        print("\n开始优化模型训练...")

        # 预处理数据
        X = self.preprocess_features()
        y_dict = self.preprocess_targets_improved()

        all_evaluation_results = {}

        for target_name, y in y_dict.items():
            print(f"\n训练 {target_name} 预测模型...")

            # 数据清理
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[valid_indices]
            y_clean = y[valid_indices]

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size, random_state=42
            )

            print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

            # 创建神经网络模型
            nn_model = self.create_advanced_neural_network(X_train.shape[1], target_name)

            # 定义回调函数
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    f'best_optimized_model_{target_name}.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
            ]

            # 训练神经网络
            history = nn_model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # 创建集成模型
            ensemble_models = self.create_ensemble_model(X_train, y_train, target_name)

            # 预测
            nn_pred = nn_model.predict(X_test)
            ensemble_preds = {}
            for model_name, model in ensemble_models.items():
                ensemble_preds[model_name] = model.predict(X_test)

            # 集成预测（加权平均）- 移除xgboost和lightgbm后重新分配权重
            weights = {'neural_network': 0.7, 'random_forest': 0.3}
            final_pred = (weights['neural_network'] * nn_pred.flatten() +
                         weights['random_forest'] * ensemble_preds['random_forest'])

            # 保存模型
            self.models[target_name] = nn_model
            self.ensemble_models[target_name] = ensemble_models

            # 评估模型
            y_test_original = self.scalers[target_name].inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.scalers[target_name].inverse_transform(final_pred.reshape(-1, 1)).flatten()

            # 计算评估指标
            mae = mean_absolute_error(y_test, final_pred)
            mse = mean_squared_error(y_test, final_pred)
            r2 = r2_score(y_test, final_pred)

            # 计算分钟误差
            if target_name in ['实际离港时间', '实际到港时间']:
                minute_errors = np.abs(y_test_original - y_pred_original)
            else:
                minute_errors = np.abs(y_test_original - y_pred_original) * 60

            mean_minute_error = np.mean(minute_errors)

            all_evaluation_results[target_name] = {
                'MAE': mae,
                'RMSE': np.sqrt(mse),
                'R²': r2,
                'Mean_Minute_Error': mean_minute_error,
                'y_test': y_test,
                'y_pred': final_pred,
                'y_test_original': y_test_original,
                'y_pred_original': y_pred_original
            }

            # 绘制结果
            self.plot_prediction_analysis(y_test, final_pred, target_name)
            self.plot_training_history(history, target_name)

            # 特征重要性分析（使用随机森林）
            if 'random_forest' in ensemble_models:
                self.feature_importance[target_name] = ensemble_models['random_forest'].feature_importances_

        # 打印评估结果
        self.print_evaluation_results(all_evaluation_results)

        return True

    def plot_prediction_analysis(self, y_test, y_pred, target_name):
        """绘制预测分析图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 预测vs实际值散点图
            axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('实际值')
            axes[0, 0].set_ylabel('预测值')
            axes[0, 0].set_title(f'{target_name} - 预测vs实际值')

            # 残差图
            residuals = y_test - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('预测值')
            axes[0, 1].set_ylabel('残差')
            axes[0, 1].set_title('残差分布')

            # 误差分布直方图
            axes[1, 0].hist(residuals, bins=50, alpha=0.7)
            axes[1, 0].set_xlabel('残差')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('残差分布直方图')

            # Q-Q图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q图')

            plt.tight_layout()
            plt.savefig(f'optimized_prediction_analysis_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"绘制预测分析图时出错: {str(e)}")

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
            plt.savefig(f'optimized_training_history_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"绘制训练历史时出错: {str(e)}")

    def plot_feature_importance(self, target_name, top_n=20):
        """绘制特征重要性"""
        if target_name not in self.feature_importance:
            return

        try:
            importance = self.feature_importance[target_name]
            feature_names = [f'feature_{i}' for i in range(len(importance))]

            # 获取top_n重要特征
            indices = np.argsort(importance)[::-1][:top_n]

            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), importance[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('重要性')
            plt.title(f'{target_name} - 特征重要性 (Top {top_n})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'feature_importance_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"绘制特征重要性时出错: {str(e)}")

    def print_evaluation_results(self, results):
        """打印评估结果"""
        print("\n" + "="*80)
        print("优化模型评估结果汇总")
        print("="*80)

        for target_name, metrics in results.items():
            print(f"\n{target_name} 模型评估结果:")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  R²: {metrics['R²']:.4f}")
            print(f"  平均误差(分钟): {metrics['Mean_Minute_Error']:.2f}")

            # 绘制特征重要性
            self.plot_feature_importance(target_name)

    def predict_ensemble(self, input_data):
        """使用集成模型进行预测"""
        if not self.models or not self.ensemble_models:
            print("没有训练好的模型")
            return None

        # 预处理输入数据
        processed_input = self.preprocess_single_input(input_data)

        predictions = {}
        for target_name in self.target_columns:
            if target_name in self.models and target_name in self.ensemble_models:
                # 神经网络预测
                nn_pred = self.models[target_name].predict(processed_input)

                # 集成模型预测
                ensemble_preds = {}
                for model_name, model in self.ensemble_models[target_name].items():
                    ensemble_preds[model_name] = model.predict(processed_input)

                # 加权集成（移除xgboost和lightgbm后重新分配权重）
                weights = {'neural_network': 0.7, 'random_forest': 0.3}
                final_pred = (weights['neural_network'] * nn_pred.flatten() +
                             weights['random_forest'] * ensemble_preds['random_forest'])

                # 反标准化
                pred_original = self.scalers[target_name].inverse_transform(final_pred.reshape(-1, 1))
                predictions[target_name] = pred_original.flatten()

        return predictions

    def save_models(self, model_dir='optimized_models'):
        """保存优化模型"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 保存神经网络模型
        for target_name, model in self.models.items():
            model.save(f'{model_dir}/nn_model_{target_name}.h5')

        # 保存集成模型
        with open(f'{model_dir}/ensemble_models.pkl', 'wb') as f:
            pickle.dump(self.ensemble_models, f)

        # 保存预处理器
        with open(f'{model_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        with open(f'{model_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)

        # 保存特征重要性
        with open(f'{model_dir}/feature_importance.pkl', 'wb') as f:
            pickle.dump(self.feature_importance, f)

        print(f"优化模型已保存到 {model_dir} 目录")

# 使用示例
if __name__ == "__main__":
    # 创建优化模型实例
    model = OptimizedFlightPredictionModel(r'/path/to/example-data')

    # 加载和预处理数据
    if model.load_and_preprocess_data():
        # 训练优化模型
        success = model.train_models_optimized(
            test_size=0.2,
            validation_split=0.2,
            epochs=200,
            batch_size=32
        )

        if success:
            # 保存模型
            model.save_models()
            print("\n优化模型训练完成！")
        else:
            print("训练失败")
