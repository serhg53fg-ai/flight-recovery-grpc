"""Deterministic greedy recovery with monotonically increasing candidate times."""
from collections import Counter
import heapq

from .domain import inputs, rotations

ALGORITHM_VERSION = 'recovery-greedy-v1'
ROTATION_ALGORITHM_VERSION = 'recovery-greedy-rotation-v1'


def schedule(records, config, *, priority_policy='earliest'):
    if priority_policy not in {'earliest', 'rotation_urgency'}:
        raise ValueError('invalid recovery priority policy')
    flights, scenario = inputs(records, config)
    groups = rotations(flights)
    def pending_key(time, tail, index):
        if priority_policy == 'earliest':
            return time, tail, index
        group = groups[tail]
        flight = group[index]
        chain_length = len(group) - index - 1
        urgency = 0.0
        if index:
            predecessor = group[index - 1]
            slack = (flight.planned_departure - predecessor.planned_arrival - scenario.mtt).total_seconds() / 60
            urgency = 1 / max(slack, 1)
        type_factor = 1.0 if flight.origin == scenario.airport else 0.5
        minute_of_day = flight.planned_departure.hour * 60 + flight.planned_departure.minute
        score = type_factor + 5 * chain_length + 10 * urgency - .001 * minute_of_day
        return scenario.slot_index(max(time, scenario.start)), -score, time, tail, index

    pending = [pending_key(group[0].departure, tail, 0) for tail, group in groups.items()]
    heapq.heapify(pending)
    usage = {'departure': Counter(), 'arrival': Counter()}
    assigned, lost = {}, {}
    while pending:
        *_, tail, index = heapq.heappop(pending)
        flight = groups[tail][index]
        predecessor = groups[tail][index - 1] if index else None
        earliest = max(flight.departure, scenario.start)
        reason = None
        if predecessor:
            if predecessor.flight_id in lost:
                reason = 'PREDECESSOR_UNSCHEDULED'
            elif predecessor.destination != flight.origin:
                reason = 'ROTATION_DISCONTINUITY'
            else:
                earliest = max(earliest, assigned[predecessor.flight_id][1] + scenario.mtt)
        if not reason and (earliest >= scenario.end or earliest + flight.duration >= scenario.end):
            reason = 'OUTSIDE_HORIZON'
        candidate = earliest
        if not reason and ((flight.origin == scenario.airport and scenario.departure_capacity == 0)
                           or (flight.destination == scenario.airport and scenario.arrival_capacity == 0)):
            reason = 'NO_FEASIBLE_SLOT'
        if not reason:
            while candidate < scenario.end and candidate + flight.duration < scenario.end:
                next_time = candidate
                for kind, applies, offset in [('departure', flight.origin == scenario.airport, flight.duration * 0),
                                             ('arrival', flight.destination == scenario.airport, flight.duration)]:
                    if not applies:
                        continue
                    time = candidate + offset
                    capacity = getattr(scenario, kind + '_capacity')
                    if usage[kind][scenario.slot_index(time)] >= capacity:
                        next_time = max(next_time, scenario.slot_end(time) - offset)
                    for closure_kind, start, end in scenario.closures:
                        if kind == closure_kind and start <= time < end:
                            next_time = max(next_time, end - offset)
                if next_time == candidate:
                    break
                candidate = next_time
            if candidate >= scenario.end or candidate + flight.duration >= scenario.end:
                reason = 'NO_FEASIBLE_SLOT'
        if reason:
            lost[flight.flight_id] = reason
        else:
            arrival = candidate + flight.duration
            assigned[flight.flight_id] = (candidate, arrival)
            if flight.origin == scenario.airport:
                usage['departure'][scenario.slot_index(candidate)] += 1
            if flight.destination == scenario.airport:
                usage['arrival'][scenario.slot_index(arrival)] += 1
        if index + 1 < len(groups[tail]):
            following = groups[tail][index + 1]
            next_earliest = max(following.departure, assigned[flight.flight_id][1] + scenario.mtt) if not reason else following.departure
            heapq.heappush(pending, pending_key(next_earliest, tail, index + 1))
    # Output follows immutable input order; scheduling order is independent.
    return dict(algorithm_version=(ALGORITHM_VERSION if priority_policy == 'earliest'
                                   else ROTATION_ALGORITHM_VERSION),
                status='SUCCEEDED' if not lost else 'PARTIAL' if assigned else 'INFEASIBLE',
                assignments=[dict(flight_id=f.flight_id,
                                  departure=assigned[f.flight_id][0].astimezone(scenario.start.tzinfo).isoformat(),
                                  arrival=assigned[f.flight_id][1].astimezone(scenario.start.tzinfo).isoformat(),
                                  delay_minutes=(assigned[f.flight_id][0] - f.departure).total_seconds()/60)
                             for f in flights if f.flight_id in assigned],
                unassigned=[dict(flight_id=f.flight_id, reason=lost[f.flight_id]) for f in flights if f.flight_id in lost])
