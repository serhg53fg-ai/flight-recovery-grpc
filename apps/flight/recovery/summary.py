"""Explain recovery coverage and compare against the same predicted flights."""

from .validation import validate_plan
from .domain import inputs as normalized_inputs


def summarize_recovery(inputs: list[dict], plan: dict) -> dict:
    assigned = plan['assignments']
    unassigned = plan['unassigned']
    flights, scenario = normalized_inputs(inputs, plan['scenario'])
    baseline = {
        'algorithm_version': plan['algorithm_version'],
        'status': 'SUCCEEDED',
        'assignments': [dict(flight_id=flight.flight_id,
                             departure=flight.departure.astimezone(scenario.start.tzinfo).isoformat(),
                             arrival=flight.arrival.astimezone(scenario.start.tzinfo).isoformat(),
                             delay_minutes=0) for flight in flights],
        'unassigned': [],
    }
    baseline_errors = validate_plan(inputs, plan['scenario'], baseline)
    delays = [float(row['delay_minutes']) for row in assigned]
    total = sum(delays)
    maximum = max(delays, default=0.0)
    affected = sum(delay > 0 for delay in delays)
    scheduled = len(assigned)
    requested = scheduled + len(unassigned)
    violations = len(plan.get('validation_errors', []))
    objective = {
        'validation_violations': violations,
        'unscheduled_flights': len(unassigned),
        'total_delay_minutes': total,
        'max_delay_minutes': maximum,
        'affected_flights': affected,
    }
    return {
        'scheduled_count': len(assigned),
        'unscheduled_count': len(unassigned),
        'excluded_prediction_count': len(plan.get('excluded', [])),
        'added_delay_minutes': total,
        'total_delay_minutes': total,
        'max_delay_minutes': maximum,
        'average_delay_minutes': total / scheduled if scheduled else 0.0,
        'affected_flight_count': affected,
        'completion_rate': scheduled / requested if requested else 0.0,
        'validation_violation_count': violations,
        'objective_vector': objective,
        'predicted_baseline_feasible': not baseline_errors,
        'predicted_baseline_validation_errors': baseline_errors,
        'improvement_claimed': False,
    }
