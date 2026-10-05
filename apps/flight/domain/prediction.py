"""Prediction boundary types; no model or Web dependencies."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flight.v1 import prediction_pb2 as pb
from google.protobuf.json_format import MessageToDict, ParseDict


TIME_FIELDS = ('off_block', 'takeoff', 'landing', 'on_block')
TIME_LABELS = ('实际离港时间', '实际起飞时间', '实际落地时间', '实际到港时间')
ALIASES = {
    'flight_number': ('航班号', 'flight_number', 'flightNo'),
    'tail_number': ('机尾号', 'tail_number', 'tailNo'),
    'aircraft_type': ('机型', 'aircraft_type', 'plane_type'),
    'flight_nature': ('性质', 'nature'),
    'departure_airport': ('计划起飞站四字码', '起飞站四字码', 'departure_airport'),
    'arrival_airport': ('计划到达站四字码', '到达站四字码', 'arrival_airport'),
    'planned_off_block': ('计划离港时间', 'planned_departure_time', 'scheduled_departure_time'),
    'planned_on_block': ('计划到港时间', 'planned_arrival_time', 'scheduled_arrival_time'),
    'departure_metar': ('起飞站METAR', 'departure_metar', 'metar_data'),
    'arrival_metar': ('到达站METAR', 'arrival_metar'),
    'planned_distance_miles': ('计划地面航程_Mile', 'planned_distance', 'planned_distance_miles'),
    'planned_flight_minutes': ('计划航段时间\n（24年同航季平均值）', 'planned_flight_time', 'planned_flight_minutes'),
    'planned_takeoff_count': ('计划起飞数', 'planned_takeoff_count'),
    'planned_landing_count': ('计划降落数', 'planned_landing_count'),
    'planned_total_flow': ('计划总流量', 'planned_total_flow'),
}


def timezone_for(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
        raise ValueError('无效 input_timezone') from exc


def parse_time(value, input_timezone='Asia/Shanghai'):
    tz = timezone_for(input_timezone)
    if not isinstance(value, (str, datetime)):
        raise ValueError('时间必须包含完整日期和时刻')
    try:
        if isinstance(value, datetime):
            dt = value
        else:
            if not re.match(r'^\d{4}-\d\d-\d\d[ T]\d\d:\d\d', value.strip()):
                raise ValueError('时间必须包含完整日期和时刻')
            dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        if dt.tzinfo is None:
            # Ambiguous/nonexistent local times must be supplied with an offset.
            if dt.replace(tzinfo=tz, fold=0).utcoffset() != dt.replace(tzinfo=tz, fold=1).utcoffset():
                raise ValueError('夏令时切换时刻必须提供 UTC 偏移量')
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'无效时间: {value}') from exc


def _value(data, key):
    for alias in (key, *ALIASES.get(key, ())):
        if alias in data and data[alias] is not None and data[alias] != '':
            return data[alias]
    return None


def normalize_flight(data, input_timezone='Asia/Shanghai'):
    if not isinstance(data, dict):
        raise ValueError('每条航班必须为对象')
    timezone_for(input_timezone)
    f = pb.FlightFeatures(flight_id=str(uuid4()))
    nature = _value(data, 'flight_nature')
    if nature is not None:
        if not isinstance(nature, str) or not re.fullmatch(r'[A-Z0-9_-]{1,32}', nature.strip().upper()):
            raise ValueError('无效航班性质')
        f.flight_nature = nature.strip().upper()
    for field in ('flight_number', 'tail_number', 'aircraft_type', 'departure_airport', 'arrival_airport'):
        value = _value(data, field)
        if value is None or isinstance(value, (bool, dict, list)):
            raise ValueError(f'缺少或无效字段: {field}')
        setattr(f, field, str(value).strip().upper())
    identifier_patterns = {
        'flight_number': r'[A-Z0-9]{2,16}',
        'tail_number': r'[A-Z0-9-]{1,32}',
        'aircraft_type': r'[A-Z0-9-]{1,32}',
    }
    for field, pattern in identifier_patterns.items():
        if not re.fullmatch(pattern, getattr(f, field)):
            raise ValueError(f'无效标识字段: {field}')
    for field in ('planned_off_block', 'planned_on_block'):
        getattr(f, field).FromDatetime(parse_time(_value(data, field), input_timezone))
    for field in ('departure_metar', 'arrival_metar'):
        value = _value(data, field)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f'{field} 必须为字符串')
            setattr(f, field, value.strip())
    for field in ('planned_distance_miles', 'planned_flight_minutes', 'planned_takeoff_count',
                  'planned_landing_count', 'planned_total_flow'):
        value = _value(data, field)
        if value is None:
            continue
        try:
            number = float(value)
            if isinstance(value, bool) or not math.isfinite(number) or number < 0:
                raise ValueError()
            if field.endswith(('count', 'flow')):
                if not number.is_integer() or number > 2147483647:
                    raise ValueError()
                number = int(number)
            setattr(f, field, number)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'无效非负数: {field}') from exc
    validate_request(pb.PredictRequest(trace_id='normalization', flight=f))
    return f


def normalize_zggg_context(data, flight, input_timezone='Asia/Shanghai'):
    if not isinstance(data, dict):
        return None
    raw = data.get('zggg_context')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('zggg_context 必须为对象')
    if (flight.departure_airport == 'ZGGG') == (flight.arrival_airport == 'ZGGG'):
        raise ValueError('ZGGG 模式必须是广州白云机场进港或离港航班')
    cutoff = raw.get('prediction_cutoff', raw.get('prediction_cutoff_time'))
    cleaned = {'prediction_cutoff': parse_time(cutoff, input_timezone).isoformat(),
               'data_version': raw.get('data_version', '')}
    weather_keys = {'kind', 'airport', 'issue_time', 'valid_from', 'valid_to',
                    'missing', 'stale', 'report_age_minutes', 'features', 'source'}
    feature_keys = {'wind_direction_deg', 'wind_speed_kt', 'wind_gust_kt', 'visibility_m',
                    'ceiling_ft', 'temperature_c', 'dewpoint_c', 'pressure_hpa',
                    'precipitation', 'thunderstorm', 'cumulonimbus', 'cavok'}
    for side in ('departure_weather', 'arrival_weather'):
        cleaned[side] = {}
        supplied = raw.get(side, {})
        if not isinstance(supplied, dict):
            raise ValueError(f'{side} 必须为对象')
        for kind in ('metar', 'taf'):
            report = supplied.get(kind)
            if report is None:
                continue
            if not isinstance(report, dict):
                raise ValueError('天气快照必须为对象')
            value = {key: report[key] for key in weather_keys if key in report}
            features = value.get('features', {})
            if not isinstance(features, dict):
                raise ValueError('天气特征必须为对象')
            value['features'] = {key: features[key] for key in feature_keys if key in features}
            cleaned[side][kind] = value
    windows = raw.get('flow_features', [])
    if not isinstance(windows, list):
        raise ValueError('flow_features 必须为数组')
    cleaned['flow_features'] = windows
    try:
        context = ParseDict(cleaned, pb.ZgggPredictionContext())
    except (ValueError, TypeError) as exc:
        raise ValueError('无效 ZGGG 上下文') from exc
    validate_request(pb.PredictRequest(trace_id='normalization', flight=flight,
                                       zggg_context=context))
    return context


def flow_prediction_to_dict(response, cutoff=None):
    if not response.HasField('airport_flow'):
        return None
    windows = [{"horizon_minutes": value.horizon_minutes,
                "takeoff": value.takeoff, "landing": value.landing,
                "total": value.total}
               for value in response.airport_flow.windows]
    if cutoff is not None:
        start = datetime.fromisoformat(cutoff)
        for window in windows:
            window['window_start'] = start.isoformat()
            window['window_end'] = (start + timedelta(minutes=window['horizon_minutes'])).isoformat()
    ordered = sorted(windows, key=lambda window: window['horizon_minutes'])
    warning = any(later[key] < earlier[key]
                  for earlier, later in zip(ordered, ordered[1:])
                  for key in ('takeoff', 'landing', 'total'))
    return {"windows": windows, "window_semantics": "cumulative",
            "flow_consistency_warning": warning,
            "model_version": response.airport_flow.model_version,
            "degraded": response.flow_degraded}


def _timestamp(message, field):
    if not message.HasField(field):
        raise ValueError(f'缺少时间: {field}')
    ts = getattr(message, field)
    if not (-62135596800 <= ts.seconds <= 253402300799) or not (0 <= ts.nanos < 1000000000):
        raise ValueError(f'无效 Timestamp: {field}')
    return ts.seconds, ts.nanos


def validate_request(request):
    if not request.trace_id.strip() or len(request.trace_id) > 128 or not request.HasField('flight'):
        raise ValueError('缺少或无效 trace_id/flight')
    f = request.flight
    if f.HasField('flight_nature') and not re.fullmatch(r'[A-Z0-9_-]{1,32}', f.flight_nature):
        raise ValueError('无效航班性质')
    for field in ('flight_id', 'flight_number', 'tail_number', 'aircraft_type'):
        if not getattr(f, field).strip() or len(getattr(f, field)) > 128:
            raise ValueError(f'缺少或无效字段: {field}')
    for field, pattern in {
        'flight_number': r'[A-Z0-9]{2,16}',
        'tail_number': r'[A-Z0-9-]{1,32}',
        'aircraft_type': r'[A-Z0-9-]{1,32}',
    }.items():
        if not re.fullmatch(pattern, getattr(f, field)):
            raise ValueError(f'无效标识字段: {field}')
    for field in ('departure_airport', 'arrival_airport'):
        if not re.fullmatch('[A-Z0-9]{4}', getattr(f, field)):
            raise ValueError(f'{field} 必须为四字机场码')
    if _timestamp(f, 'planned_on_block') <= _timestamp(f, 'planned_off_block'):
        raise ValueError('计划到港必须晚于计划离港')
    for field in ('planned_distance_miles', 'planned_flight_minutes', 'planned_takeoff_count',
                  'planned_landing_count', 'planned_total_flow'):
        if f.HasField(field) and (not math.isfinite(getattr(f, field)) or getattr(f, field) < 0):
            raise ValueError(f'无效非负数: {field}')
    if len(f.departure_metar) > 8192 or len(f.arrival_metar) > 8192:
        raise ValueError('METAR 过长')
    if request.HasField('zggg_context'):
        context = request.zggg_context
        cutoff = _timestamp(context, 'prediction_cutoff')
        if cutoff != _timestamp(f, 'planned_off_block'):
            raise ValueError('ZGGG prediction cutoff 必须等于计划离港时间')
        if not context.data_version.strip() or len(context.data_version) > 128:
            raise ValueError('ZGGG data version 无效')
        horizons = set()
        for window in context.flow_features:
            if window.horizon_minutes not in (15, 30, 60) or window.horizon_minutes in horizons:
                raise ValueError('无效流量时间窗')
            horizons.add(window.horizon_minutes)
            if min(window.planned_takeoff, window.planned_landing,
                   window.completed_takeoff, window.completed_landing) < 0:
                raise ValueError('流量特征必须非负')
        for weather_name in ('departure_weather', 'arrival_weather'):
            weather = getattr(context, weather_name)
            for kind in ('metar', 'taf'):
                if not weather.HasField(kind):
                    continue
                report = getattr(weather, kind)
                if report.missing:
                    continue
                if report.kind != kind.upper() or not re.fullmatch(r'[A-Z0-9]{4}', report.airport):
                    raise ValueError('天气快照类型或机场无效')
                issue = _timestamp(report, 'issue_time')
                if issue > cutoff:
                    raise ValueError('天气发布时间晚于 prediction cutoff')
                if report.HasField('report_age_minutes') and (
                        not math.isfinite(report.report_age_minutes) or report.report_age_minutes < 0):
                    raise ValueError('天气数据年龄无效')


def validate_response(response, trace_id=None):
    if not response.trace_id.strip() or (trace_id is not None and response.trace_id != trace_id):
        raise ValueError('推理响应 trace_id 不匹配')
    if response.source not in (pb.LLM, pb.TEST, pb.BASELINE):
        raise ValueError('推理来源无效')
    if not response.model_version.strip() or not response.worker_id.strip():
        raise ValueError('推理响应缺少模型/节点标识')
    if not response.HasField('prediction'):
        raise ValueError('推理响应缺少预测结果')
    times = [_timestamp(response.prediction, field) for field in TIME_FIELDS]
    if times != sorted(times):
        raise ValueError('预测时序必须满足离港 ≤ 起飞 ≤ 落地 ≤ 到港')
    if response.inference_ms < 0 or response.gateway_ms < 0:
        raise ValueError('推理耗时无效')
    if response.HasField('airport_flow'):
        horizons = set()
        for window in response.airport_flow.windows:
            if window.horizon_minutes not in (15, 30, 60) or window.horizon_minutes in horizons:
                raise ValueError('无效机场流量时间窗')
            horizons.add(window.horizon_minutes)
            if min(window.takeoff, window.landing, window.total) < 0:
                raise ValueError('机场流量预测必须非负')
            if window.total != window.takeoff + window.landing:
                raise ValueError('机场流量总数不一致')
        if response.airport_flow.model_version and len(response.airport_flow.model_version) > 128:
            raise ValueError('流量模型版本无效')
    if len(response.data_version) > 128 or len(response.prompt_version) > 128:
        raise ValueError('推理数据或提示词版本无效')


def prediction_to_dict(prediction, timezone='Asia/Shanghai'):
    tz = timezone_for(timezone)
    return {label: getattr(prediction, field).ToDatetime(tzinfo=tz).isoformat()
            for field, label in zip(TIME_FIELDS, TIME_LABELS)}
