import pandas as pd
import numpy as np
from datetime import datetime
import re

class METARConverter:
    def __init__(self, excel_file_path):
        self.excel_file_path = excel_file_path
        self.data = None

    def load_data(self):
        """加载Excel数据"""
        try:
            # 指定引擎来读取Excel文件
            self.data = pd.read_excel(self.excel_file_path, engine='openpyxl')
            print(f"数据加载成功，共有 {len(self.data)} 行数据")
            print(f"列名: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False

    def format_time_to_metar(self, time_str):
        """将时间格式转换为METAR格式 (DDHHMMZ)"""
        try:
            if pd.isna(time_str):
                return "010000Z"

            # 如果是字符串格式的时间
            if isinstance(time_str, str):
                # 尝试解析不同的时间格式
                try:
                    dt = pd.to_datetime(time_str)
                except:
                    return "010000Z"
            else:
                dt = time_str

            # 格式化为METAR时间格式 DDHHMMZ
            day = dt.strftime('%d')
            hour = dt.strftime('%H')
            minute = dt.strftime('%M')
            return f"{day}{hour}{minute}Z"
        except:
            return "010000Z"

    def format_wind(self, wind_dir, wind_speed, wind_gust=None, wind_var_from=None, wind_var_to=None):
        """格式化风向风速信息"""
        try:
            # 处理风向
            if pd.isna(wind_dir) or wind_dir == 0:
                wind_dir_str = "VRB"
            else:
                wind_dir_str = f"{int(wind_dir):03d}"

            # 处理风速
            if pd.isna(wind_speed):
                wind_speed = 0
            wind_speed_str = f"{int(wind_speed):02d}"

            # 基本风信息
            wind_info = f"{wind_dir_str}{wind_speed_str}KT"

            # 处理阵风
            if not pd.isna(wind_gust) and wind_gust > wind_speed:
                wind_info += f"G{int(wind_gust):02d}"

            # 处理风向变化
            if not pd.isna(wind_var_from) and not pd.isna(wind_var_to):
                wind_info += f" {int(wind_var_from):03d}V{int(wind_var_to):03d}"

            return wind_info
        except:
            return "00000KT"

    def format_visibility(self, visibility, cavok=None):
        """格式化能见度信息"""
        try:
            # 检查CAVOK条件
            if not pd.isna(cavok) and cavok == 1:
                return "CAVOK"

            if pd.isna(visibility):
                return "9999"

            # 转换能见度为米
            vis_m = int(visibility)

            if vis_m >= 10000:
                return "9999"
            elif vis_m >= 5000:
                return f"{vis_m//1000}000"
            else:
                return f"{vis_m:04d}"
        except:
            return "9999"

    def format_weather_phenomena(self, has_cb=None):
        """格式化天气现象"""
        phenomena = []

        try:
            # 检查积雨云
            if not pd.isna(has_cb) and has_cb == 1:
                phenomena.append("CB")

            if not phenomena:
                return "CLR"  # 晴朗

            return " ".join(phenomena)
        except:
            return "CLR"

    def format_temperature_dewpoint(self, temperature, dewpoint):
        """格式化温度露点信息"""
        try:
            if pd.isna(temperature):
                temp_str = "00"
            else:
                temp = int(temperature)
                if temp < 0:
                    temp_str = f"M{abs(temp):02d}"
                else:
                    temp_str = f"{temp:02d}"

            if pd.isna(dewpoint):
                dew_str = "00"
            else:
                dew = int(dewpoint)
                if dew < 0:
                    dew_str = f"M{abs(dew):02d}"
                else:
                    dew_str = f"{dew:02d}"

            return f"{temp_str}/{dew_str}"
        except:
            return "00/00"

    def format_pressure(self, pressure):
        """格式化气压信息"""
        try:
            if pd.isna(pressure):
                return "A3000"

            # 转换为英寸汞柱 * 100
            pressure_inhg = int(pressure * 2.953 / 10)  # 假设输入是hPa
            return f"A{pressure_inhg:04d}"
        except:
            return "A3000"

    def generate_metar(self, airport_code, time_str, wind_dir, wind_speed, wind_gust,
                      wind_var_from, wind_var_to, visibility, cavok, has_cb,
                      temperature, dewpoint, pressure):
        """生成完整的METAR报文"""

        # METAR头部
        metar_parts = ["METAR"]

        # 机场代码
        metar_parts.append(str(airport_code) if not pd.isna(airport_code) else "XXXX")

        # 时间
        metar_parts.append(self.format_time_to_metar(time_str))

        # 移除AUTO标识
        # metar_parts.append("AUTO")

        # 风信息
        wind_info = self.format_wind(wind_dir, wind_speed, wind_gust, wind_var_from, wind_var_to)
        metar_parts.append(wind_info)

        # 能见度
        vis_info = self.format_visibility(visibility, cavok)
        if vis_info != "CAVOK":
            metar_parts.append(vis_info)
            # 天气现象
            weather_info = self.format_weather_phenomena(has_cb)
            metar_parts.append(weather_info)
        else:
            metar_parts.append(vis_info)

        # 温度露点
        temp_dew = self.format_temperature_dewpoint(temperature, dewpoint)
        metar_parts.append(temp_dew)

        # 气压
        pressure_info = self.format_pressure(pressure)
        metar_parts.append(pressure_info)

        # 备注
        metar_parts.append("RMK")
        metar_parts.append("SLP216=")

        return " ".join(metar_parts)

    def convert_to_metar(self):
        """将数据转换为METAR报文"""
        if self.data is None:
            print("请先加载数据")
            return None

        result_data = self.data.copy()

        # 生成起飞站METAR报文
        print("正在生成起飞站METAR报文...")
        result_data['起飞站METAR'] = result_data.apply(
            lambda row: self.generate_metar(
                row.get('计划起飞站四字码'),
                row.get('计划离港时间'),
                row.get('wind_dir_s'),
                row.get('wind_speed_s'),
                row.get('wind_gust_s'),
                row.get('wind_var_from_s'),
                row.get('wind_var_to_s'),
                row.get('visibility_s'),
                row.get('cavok_s'),
                row.get('has_CB_s'),
                row.get('temperature_s'),
                row.get('dewpoint_s'),
                row.get('pressure_s')
            ), axis=1
        )

        # 生成到达站METAR报文
        print("正在生成到达站METAR报文...")
        result_data['到达站METAR'] = result_data.apply(
            lambda row: self.generate_metar(
                row.get('计划到达站四字码'),
                row.get('计划到港时间'),
                row.get('wind_dir_e'),
                row.get('wind_speed_e'),
                row.get('wind_gust_e'),
                row.get('wind_var_from_e'),
                row.get('wind_var_to_e'),
                row.get('visibility_e'),
                row.get('cavok_e'),
                row.get('has_CB_e'),
                row.get('temperature_e'),
                row.get('dewpoint_e'),
                row.get('pressure_e')
            ), axis=1
        )

        return result_data

    def save_result(self, result_data, output_file=None):
        """保存结果到Excel文件"""
        if output_file is None:
            output_file = self.excel_file_path.replace('.xlsx', '_with_METAR.xlsx')

        try:
            result_data.to_excel(output_file, index=False)
            print(f"结果已保存到: {output_file}")
            return True
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建转换器实例
    converter = METARConverter(r'/path/to/example-data')

    # 加载数据
    if converter.load_data():
        # 转换为METAR报文
        result = converter.convert_to_metar()

        if result is not None:
            # 显示前几行结果
            print("\n前5行METAR报文示例:")
            for i in range(min(5, len(result))):
                print(f"\n第{i+1}行:")
                print(f"起飞站METAR: {result.iloc[i]['起飞站METAR']}")
                print(f"到达站METAR: {result.iloc[i]['到达站METAR']}")

            # 保存结果
            converter.save_result(result)

            print(f"\n转换完成！共处理 {len(result)} 条记录")
            print("新增列：'起飞站METAR' 和 '到达站METAR'")
        else:
            print("转换失败")
    else:
        print("数据加载失败")
