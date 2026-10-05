"""Validated, immutable scheduling inputs; airport capacities concern block events."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import re


def instant(value):
    if not isinstance(value, str):
        raise ValueError('时间必须为带时区ISO字符串')
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError('无效ISO时间') from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('调度时间必须带时区')
    return result


def integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError('无效' + name)
    return value


def identity(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError('无效' + name)
    return value


def airport(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Z]{4}', value):
        raise ValueError('机场必须为四字码')
    return value


@dataclass(frozen=True)
class Flight:
    flight_id: str
    tail: str
    origin: str
    destination: str
    planned_departure: datetime
    planned_arrival: datetime
    departure: datetime
    arrival: datetime

    @property
    def duration(self):
        return self.arrival - self.departure


@dataclass(frozen=True)
class Scenario:
    airport: str
    start: datetime
    end: datetime
    slot: timedelta
    departure_capacity: int
    arrival_capacity: int
    mtt: timedelta
    closures: tuple

    def slot_index(self, time):
        return (time - self.start) // self.slot

    def slot_end(self, time):
        return self.start + (self.slot_index(time) + 1) * self.slot


def inputs(records, config):
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        raise ValueError('航班数必须为1～1000')
    required = {'airport', 'horizon_start', 'horizon_end', 'slot_minutes',
                'departure_capacity', 'arrival_capacity', 'mtt_minutes'}
    if not isinstance(config, dict) or not required <= config.keys() or set(config) - required - {'closures'}:
        raise ValueError('无效调度场景字段')
    start, end = instant(config['horizon_start']), instant(config['horizon_end'])
    if not timedelta(0) < end - start <= timedelta(hours=48):
        raise ValueError('调度时窗必须为正且不超过48小时')
    slot = integer(config['slot_minutes'], 1, 60, '时间槽')
    if 60 % slot:
        raise ValueError('时间槽须整除60分钟')
    closures = config.get('closures', [])
    if not isinstance(closures, list) or len(closures) > 100:
        raise ValueError('关闭时段最多100个')
    parsed = []
    for entry in closures:
        if not isinstance(entry, dict) or set(entry) != {'kind', 'start', 'end'} or entry['kind'] not in {'departure', 'arrival'}:
            raise ValueError('无效关闭时段')
        a, b = instant(entry['start']), instant(entry['end'])
        if not start <= a < b <= end:
            raise ValueError('关闭时段必须在时窗内')
        parsed.append((entry['kind'], a, b))
    scenario = Scenario(airport(config['airport']), start, end, timedelta(minutes=slot),
                        integer(config['departure_capacity'], 0, 200, '离港容量'),
                        integer(config['arrival_capacity'], 0, 200, '到港容量'),
                        timedelta(minutes=integer(config['mtt_minutes'], 0, 240, 'MTT')), tuple(parsed))
    flights, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('无效航班对象')
        try:
            fid = identity(record['flight_id'], 'flight_id')
            if fid in seen:
                raise ValueError('重复flight_id')
            seen.add(fid)
            pd, pa = instant(record['planned_departure']), instant(record['planned_arrival'])
            dep = max(pd, instant(record['predicted_departure']))
            arr = max(pa, instant(record['predicted_arrival']))
            if pa <= pd or arr <= dep:
                raise ValueError('航程时间必须为正')
            flights.append(Flight(fid, identity(record['tail'], '机尾'), airport(record['origin']),
                                  airport(record['destination']), pd, pa, dep, arr))
        except KeyError:
            raise ValueError('航班缺少必要字段') from None
    return flights, scenario


def rotations(flights):
    groups = {}
    for flight in flights:
        groups.setdefault(flight.tail, []).append(flight)
    for group in groups.values():
        group.sort(key=lambda f: (f.planned_departure, f.planned_arrival, f.flight_id))
    return groups
