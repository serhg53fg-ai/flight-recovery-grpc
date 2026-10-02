"""Versioned source schema and leakage boundary."""

SCHEMA_VERSION = "flight-duration-v1"
ZGGG_AIRPORT = "ZGGG"
WEATHER_FEATURE_VERSION = "aviation-weather-v1"

FEATURE_COLUMNS = {
    "机尾号": "tail_number",
    "航班号": "flight_number",
    "机型": "aircraft_type",
    "性质": "flight_nature",
    "计划起飞站四字码": "departure_airport",
    "计划到达站四字码": "arrival_airport",
    "计划离港时间": "planned_off_block",
    "计划到港时间": "planned_on_block",
    "计划地面航程_Mile": "planned_distance_miles",
    "计划航段时间\n（24年同航季平均值）": "planned_flight_minutes",
    "计划起飞数": "planned_takeoff_count",
    "计划降落数": "planned_landing_count",
    "计划总流量": "planned_total_flow",
}

TARGET_COLUMNS = {
    "实际离港时间": "actual_off_block",
    "实际起飞时间": "actual_takeoff",
    "实际落地时间": "actual_landing",
    "实际到港时间": "actual_on_block",
}

PROHIBITED_FEATURE_COLUMNS = frozenset({
    "起飞站METAR", "departure_metar", "到达站METAR", "arrival_metar",
    "实际离港时间", "实际起飞时间", "实际落地时间", "实际到港时间",
    "实际起飞站四字码", "实际到达站四字码", "实际航程_Mile",
    "实际航段时间\n（24年同航季平均值）", "实际起飞数", "实际降落数", "实际总流量",
    "actual_off_block", "actual_takeoff", "actual_landing", "actual_on_block",
    "actual_departure_airport", "actual_arrival_airport", "actual_distance",
    "actual_flight_minutes", "actual_takeoff_count", "actual_landing_count", "actual_total_flow",
})

LABEL_FIELDS = (
    "off_block_delay_min", "taxi_out_min", "airborne_min", "taxi_in_min",
)
