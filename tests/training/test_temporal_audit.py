"""A01 availability checks use the validation boundary, not row date alone."""
from datetime import datetime, timedelta, timezone


def row(name, planned, *, delay=0, weather_issue=None, availability=None):
    features = {'planned_off_block': planned.isoformat(),
                'planned_on_block': (planned + timedelta(hours=2)).isoformat(),
                'departure_airport': 'ZGGG', 'arrival_airport': 'ZSPD'}
    if weather_issue:
        features['departure_weather'] = {'metar': {'missing': False,
                                                   'issue_time': weather_issue.isoformat()}}
    if availability:
        features['feature_available_at'] = availability.isoformat()
    return {'record_id': name, 'features': features,
            'labels': {'off_block_delay_min': delay, 'taxi_out_min': 10,
                       'airborne_min': 90, 'taxi_in_min': 10}}


def test_unavailable_training_labels_excluded():
    from training.temporal_audit import audit_temporal_rows, eligible_training_rows

    boundary = datetime(2025, 5, 10, tzinfo=timezone.utc)
    early = row('early', boundary - timedelta(hours=5))
    late = row('late', boundary - timedelta(minutes=5))
    audit = audit_temporal_rows([early, late], boundary.isoformat())
    assert audit['reasons']['label_not_available_at_boundary'] == 1
    assert [item['record_id'] for item in eligible_training_rows([early, late], boundary.isoformat())] == ['early']


def test_future_feature_detected():
    from training.temporal_audit import audit_temporal_rows

    planned = datetime(2025, 5, 5, tzinfo=timezone.utc)
    future = row('future', planned, weather_issue=planned + timedelta(minutes=15))
    audit = audit_temporal_rows([future], '2025-05-10T00:00:00Z')
    assert audit['reasons']['future_weather_issue'] == 1
    assert audit['eligible_rows'] == 0


def test_unknown_availability_retained_as_limitation():
    from training.temporal_audit import audit_temporal_rows

    audit = audit_temporal_rows([row('unknown', datetime(2025, 5, 5, tzinfo=timezone.utc))],
                                '2025-05-10T00:00:00Z')
    assert audit['eligible_rows'] == 1
    assert audit['limitations']['feature_available_at_unknown'] == 1
    assert audit['limitations']['label_received_at_unknown'] == 1


def test_audit_does_not_silently_remove_extreme_delays():
    from training.temporal_audit import audit_temporal_rows

    extreme = row('extreme', datetime(2025, 5, 5, tzinfo=timezone.utc), delay=700)
    audit = audit_temporal_rows([extreme], '2025-05-10T00:00:00Z')
    assert audit['extreme_delay_rows'] == 1
    assert audit['eligible_rows'] == 1
    assert audit['reasons'] == {}
