"""Independent feasibility audit; rebuild all event counts from the result."""
from collections import Counter
from datetime import timedelta
import math

from .domain import inputs, rotations, instant


def validate_plan(records, config, plan):
    flights, scenario = inputs(records, config)
    errors, assigned, lost = [], {}, {}
    expected = {f.flight_id: f for f in flights}
    try:
        for item in plan['assignments']:
            fid = item['flight_id']
            if fid not in expected or fid in assigned:
                errors.append('ASSIGNMENT_IDENTITY')
                continue
            assigned[fid] = (instant(item['departure']), instant(item['arrival']))
            delay = item.get('delay_minutes')
            actual = (assigned[fid][0] - expected[fid].departure).total_seconds()/60
            if type(delay) not in (int, float) or not math.isfinite(delay) or not math.isclose(delay, actual, rel_tol=0, abs_tol=1e-9):
                errors.append('DELAY:' + fid)
        for item in plan['unassigned']:
            fid = item['flight_id']
            if fid not in expected or fid in assigned or fid in lost:
                errors.append('UNASSIGNED_IDENTITY')
                continue
            lost[fid] = item['reason']
    except (KeyError, TypeError, ValueError):
        return ['MALFORMED_PLAN']
    if set(assigned) | set(lost) != set(expected):
        errors.append('INCOMPLETE_COVERAGE')
    status = 'SUCCEEDED' if not lost else 'PARTIAL' if assigned else 'INFEASIBLE'
    if plan.get('status') != status or plan.get('algorithm_version') not in {
            'recovery-greedy-v1', 'recovery-greedy-rotation-v1'}:
        errors.append('STATUS_OR_VERSION')
    counts = {'departure': Counter(), 'arrival': Counter()}
    for fid, (departure, arrival) in assigned.items():
        f = expected[fid]
        if departure < f.departure or arrival < f.arrival:
            errors.append('EARLY_EVENT:' + fid)
        if arrival - departure != f.duration:
            errors.append('DURATION:' + fid)
        if not scenario.start <= departure < arrival < scenario.end:
            errors.append('HORIZON:' + fid)
        for kind, time, applies in [('departure', departure, f.origin == scenario.airport),
                                    ('arrival', arrival, f.destination == scenario.airport)]:
            if applies:
                counts[kind][scenario.slot_index(time)] += 1
                if any(k == kind and a <= time < b for k, a, b in scenario.closures):
                    errors.append('CLOSURE:' + fid)
    for kind, events in counts.items():
        if any(n > getattr(scenario, kind + '_capacity') for n in events.values()):
            errors.append('CAPACITY:' + kind)
    for group in rotations(flights).values():
        for index, f in enumerate(group):
            previous = group[index - 1] if index else None
            blocked = previous is not None and previous.flight_id not in assigned
            broken = previous is not None and previous.destination != f.origin
            earliest = max(f.departure, scenario.start)
            if previous and not blocked:
                earliest = max(earliest, assigned[previous.flight_id][1] + scenario.mtt)
            if f.flight_id in assigned:
                if blocked or broken or assigned[f.flight_id][0] < earliest:
                    errors.append('ROTATION:' + f.flight_id)
                continue
            if f.flight_id not in lost:
                continue
            reason = lost[f.flight_id]
            if blocked:
                valid = reason == 'PREDECESSOR_UNSCHEDULED'
            elif broken:
                valid = reason == 'ROTATION_DISCONTINUITY'
            elif earliest + f.duration >= scenario.end:
                valid = reason == 'OUTSIDE_HORIZON'
            else:
                valid = reason == 'NO_FEASIBLE_SLOT' and not _has_available_time(f, earliest, scenario, counts)
            if not valid:
                errors.append('UNASSIGNED_REASON:' + f.flight_id)
    return errors


def _has_available_time(flight, earliest, scenario, counts):
    # Convert every full event slot/closure to a forbidden departure interval;
    # merge intervals instead of using the engine's incremental search state.
    intervals = []
    for kind, applies, offset in [('departure', flight.origin == scenario.airport, timedelta(0)),
                                  ('arrival', flight.destination == scenario.airport, flight.duration)]:
        if not applies:
            continue
        capacity = getattr(scenario, kind + '_capacity')
        if capacity == 0:
            return False
        for index, count in counts[kind].items():
            if count >= capacity:
                a = scenario.start + index * scenario.slot - offset
                intervals.append((a, a + scenario.slot))
        intervals.extend((a - offset, b - offset) for k, a, b in scenario.closures if k == kind)
    candidate = earliest
    for start, end in sorted(intervals):
        if end <= candidate:
            continue
        if start > candidate:
            break
        candidate = end
    return candidate + flight.duration < scenario.end
