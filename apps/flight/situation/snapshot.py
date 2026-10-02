"""Build an immutable ZGGG operating-situation snapshot."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math


def _instant(value):
    if not isinstance(value, str):
        raise ValueError('态势时刻必须为字符串')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('态势时刻必须包含时区')
    return parsed


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'无效{name}')
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f'无效{name}')
    return int(number)


def _flow_windows(rows, scenario):
    selected = {}
    for row in rows:
        flow = row.get('airport_flow')
        if not isinstance(flow, dict):
            continue
        if flow.get('window_semantics') not in (None, 'cumulative'):
            raise ValueError('仅支持累计流量窗口')
        for raw in flow.get('windows', []):
            if not isinstance(raw, dict):
                raise ValueError('无效流量窗口')
            # Older stored jobs exposed only total flow. They remain valid
            # recovery inputs, but cannot support directional pressure claims.
            if not all(name in raw for name in ('takeoff', 'landing', 'total')):
                continue
            horizon = _nonnegative_int(raw.get('horizon_minutes'), '流量窗口')
            if horizon <= 0:
                raise ValueError('无效流量窗口')
            values = {name: _nonnegative_int(raw.get(name), name)
                      for name in ('takeoff', 'landing', 'total')}
            if values['total'] != values['takeoff'] + values['landing']:
                raise ValueError('流量总量不一致')
            candidate = {'horizon_minutes': horizon, **values}
            previous = selected.get(horizon)
            if previous is not None and previous != candidate:
                raise ValueError('同一任务包含冲突的流量预测')
            selected[horizon] = candidate

    slot = _nonnegative_int(scenario.get('slot_minutes'), '时隙长度')
    if slot <= 0:
        raise ValueError('无效时隙长度')
    departure = _nonnegative_int(scenario.get('departure_capacity'), '离港容量')
    arrival = _nonnegative_int(scenario.get('arrival_capacity'), '到港容量')
    windows, conflicts = [], []
    for horizon, value in sorted(selected.items()):
        slots = math.ceil(horizon / slot)
        capacity = {'takeoff': departure * slots, 'landing': arrival * slots}
        pressure = {
            kind: round(value[kind] / capacity[kind], 4) if capacity[kind] else (
                0.0 if value[kind] == 0 else None)
            for kind in ('takeoff', 'landing')
        }
        output = {**value, 'capacity': capacity, 'pressure': pressure}
        windows.append(output)
        for kind, conflict_type in (('takeoff', 'DEPARTURE_CAPACITY'),
                                    ('landing', 'ARRIVAL_CAPACITY')):
            if value[kind] > capacity[kind]:
                conflicts.append({
                    'type': conflict_type,
                    'severity': 'HIGH',
                    'horizon_minutes': horizon,
                    'forecast': value[kind],
                    'capacity': capacity[kind],
                    'excess': value[kind] - capacity[kind],
                })
    return windows, conflicts



def _slot_capacity_conflicts(rows, scenario):
    """Count block events in the same horizon-anchored slots as recovery."""
    start, end = _instant(scenario['horizon_start']), _instant(scenario['horizon_end'])
    slot = timedelta(minutes=_nonnegative_int(scenario['slot_minutes'], '时隙长度'))
    if end <= start or slot.total_seconds() <= 0:
        raise ValueError('无效态势时窗或时间槽')
    events = {'departure': defaultdict(list), 'arrival': defaultdict(list)}
    for row in rows:
        info, prediction = row['flight_info'], row['prediction_data']
        for kind, station, field in (
            ('departure', '计划起飞站四字码', '实际离港时间'),
            ('arrival', '计划到达站四字码', '实际到港时间'),
        ):
            if info[station] != scenario['airport']:
                continue
            planned = '计划离港时间' if kind == 'departure' else '计划到港时间'
            time = max(_instant(prediction[field]), _instant(info[planned]))
            if start <= time < end:
                events[kind][(time - start) // slot].append(row['flight_id'])
    conflicts = []
    for kind, groups in events.items():
        capacity = _nonnegative_int(scenario[kind + '_capacity'], '容量')
        for index, flight_ids in sorted(groups.items()):
            if len(flight_ids) > capacity:
                slot_start = start + index * slot
                conflicts.append({
                    'type': kind.upper() + '_SLOT_CAPACITY', 'severity': 'HIGH',
                    'event_basis': 'block_events',
                    'slot_start': slot_start.isoformat(),
                    'slot_end': min(slot_start + slot, end).isoformat(),
                    'count': len(flight_ids), 'capacity': capacity,
                    'excess': len(flight_ids) - capacity,
                    'flight_ids': sorted(flight_ids),
                })
    return conflicts


def _rotation_conflicts(rows, mtt_minutes):
    by_tail = defaultdict(list)
    for row in rows:
        info, prediction = row.get('flight_info'), row.get('prediction_data')
        if not isinstance(info, dict) or not isinstance(prediction, dict):
            continue
        try:
            by_tail[info['机尾号']].append({
                'flight_id': row['flight_id'],
                'origin': info['计划起飞站四字码'],
                'destination': info['计划到达站四字码'],
                'departure': _instant(prediction['实际离港时间']),
                'arrival': _instant(prediction['实际到港时间']),
            })
        except KeyError as exc:
            raise ValueError('成功预测缺少态势字段') from exc
    conflicts = []
    for tail, flights in by_tail.items():
        flights.sort(key=lambda item: item['departure'])
        for predecessor, successor in zip(flights, flights[1:]):
            if predecessor['destination'] != successor['origin']:
                continue
            connection = (successor['departure'] - predecessor['arrival']).total_seconds() / 60
            shortfall = float(mtt_minutes) - connection
            if shortfall > 0:
                conflicts.append({
                    'type': 'ROTATION_CONNECTION', 'severity': 'HIGH',
                    'tail_number': tail,
                    'predecessor_flight_id': predecessor['flight_id'],
                    'successor_flight_id': successor['flight_id'],
                    'shortfall_minutes': round(shortfall, 3),
                })
    return conflicts


def _closure_conflicts(rows, scenario):
    airport = scenario.get('airport')
    closures = scenario.get('closures', [])
    if not isinstance(closures, list):
        raise ValueError('关闭时段必须为数组')
    parsed = []
    for closure in closures:
        if not isinstance(closure, dict) or closure.get('kind') not in {'departure', 'arrival'}:
            raise ValueError('无效关闭时段')
        start, end = _instant(closure.get('start')), _instant(closure.get('end'))
        if start >= end:
            raise ValueError('无效关闭时段')
        parsed.append((closure['kind'], start, end))
    conflicts = []
    for row in rows:
        info, prediction = row.get('flight_info', {}), row.get('prediction_data', {})
        events = (
            ('departure', info.get('计划起飞站四字码') == airport, prediction.get('实际离港时间')),
            ('arrival', info.get('计划到达站四字码') == airport, prediction.get('实际到港时间')),
        )
        for kind, applies, raw_time in events:
            if not applies:
                continue
            event = _instant(raw_time)
            if any(closure_kind == kind and start <= event < end
                   for closure_kind, start, end in parsed):
                conflicts.append({
                    'type': 'AIRPORT_CLOSURE', 'severity': 'HIGH', 'kind': kind,
                    'flight_id': row.get('flight_id'), 'event_time': raw_time,
                })
    return conflicts


def _has_adverse_weather(row):
    context = row.get('normalized_context')
    if not isinstance(context, dict):
        return False
    for side in ('departure_weather', 'arrival_weather'):
        weather = context.get(side, {})
        if not isinstance(weather, dict):
            continue
        for kind in ('metar', 'taf'):
            report = weather.get(kind, {})
            features = report.get('features', {}) if isinstance(report, dict) else {}
            if not isinstance(features, dict):
                continue
            if any(features.get(flag) is True for flag in (
                    'thunderstorm', 'cumulonimbus', 'precipitation')):
                return True
            thresholds = (('visibility_m', 3000, 'below'), ('ceiling_ft', 1000, 'below'),
                          ('wind_speed_kt', 20, 'above'), ('wind_gust_kt', 25, 'above'))
            for name, threshold, direction in thresholds:
                value = features.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if ((direction == 'below' and value < threshold) or
                            (direction == 'above' and value >= threshold)):
                        return True
    return False


def build_situation_snapshot(job, scenario):
    """Project a terminal prediction job into one auditable situation view."""
    if not isinstance(job, dict) or not isinstance(scenario, dict):
        raise ValueError('态势输入必须为对象')
    rows = job.get('results')
    if job.get('status') not in {'SUCCEEDED', 'PARTIAL', 'FAILED', 'EXPIRED', 'CANCELLED'}:
        raise ValueError('态势快照只接受终态预测任务')
    if not isinstance(rows, list) or not rows:
        raise ValueError('态势快照缺少预测结果')
    successful = [row for row in rows if isinstance(row, dict) and row.get('success') is True]
    failed = [row for row in rows if not isinstance(row, dict) or row.get('success') is not True]
    sources = Counter(str(row.get('source', 'UNKNOWN')) for row in successful)
    windows, conflicts = _flow_windows(successful, scenario)
    conflicts.extend(_slot_capacity_conflicts(successful, scenario))
    mtt = _nonnegative_int(scenario.get('mtt_minutes'), '最小过站时间')
    conflicts.extend(_rotation_conflicts(successful, mtt))
    conflicts.extend(_closure_conflicts(successful, scenario))
    adverse = [row for row in successful if _has_adverse_weather(row)]
    if adverse:
        conflicts.append({'type': 'ADVERSE_WEATHER', 'severity': 'HIGH',
                          'affected_flights': len(adverse)})
    snapshot = {
        'source_job_id': job.get('job_id'),
        'airport': scenario.get('airport'),
        'horizon_start': scenario.get('horizon_start'),
        'horizon_end': scenario.get('horizon_end'),
        'prediction': {
            'requested': len(rows), 'successful': len(successful), 'failed': len(failed),
            'sources': dict(sorted(sources.items())),
            'flight_fallbacks': sum(bool(row.get('flight_degraded')) for row in successful),
            'flow_fallbacks': sum(bool(row.get('flow_degraded')) for row in successful),
            'stale_weather': sum(bool(row.get('weather_stale')) for row in successful),
            'adverse_weather': len(adverse),
        },
        'flow': {'window_semantics': 'cumulative', 'windows': windows},
        'conflicts': conflicts,
        'risk_level': 'HIGH' if conflicts or failed else 'NORMAL',
    }
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False)
    snapshot['snapshot_digest'] = hashlib.sha256(canonical.encode()).hexdigest()
    return snapshot
