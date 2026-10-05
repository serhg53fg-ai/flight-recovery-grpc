from datetime import datetime, timedelta, timezone

import pytest


def rows(days=10):
    return [
        {'record_id': str(day), 'features': {
            'planned_off_block': (datetime(2025, 5, 1, tzinfo=timezone.utc)
                                  + timedelta(days=day)).isoformat(),
        }, 'labels': {'off_block_delay_min': 0, 'taxi_out_min': 10,
                      'airborne_min': 90, 'taxi_in_min': 10}} for day in range(days)
    ]


def test_expanding_weather_folds_train_only_on_earlier_local_dates():
    from training.rolling_weather import expanding_folds

    folds = expanding_folds(rows(), initial_train_dates=4, validation_dates=2,
                            step_dates=2, input_timezone='Asia/Shanghai')
    assert len(folds) == 3
    assert [[row['record_id'] for row in fold.validation_rows] for fold in folds] == [
        ['4', '5'], ['6', '7'], ['8', '9']]
    for fold in folds:
        assert max(fold.train_dates) < min(fold.validation_dates)
        assert not ({row['record_id'] for row in fold.train_rows} &
                    {row['record_id'] for row in fold.validation_rows})


def test_expanding_folds_reject_insufficient_dates_or_naive_times():
    from training.rolling_weather import expanding_folds

    with pytest.raises(ValueError, match='insufficient dates'):
        expanding_folds(rows(5), initial_train_dates=4, validation_dates=2,
                        step_dates=2)
    values = rows()
    values[0]['features']['planned_off_block'] = '2025-05-01T00:00:00'
    with pytest.raises(ValueError, match='timezone'):
        expanding_folds(values, initial_train_dates=4, validation_dates=2,
                        step_dates=2)


def test_expanding_folds_exclude_labels_unavailable_at_validation_start():
    from training.rolling_weather import expanding_folds

    values = rows()
    values[3]['labels']['off_block_delay_min'] = 3000
    folds = expanding_folds(values, initial_train_dates=4, validation_dates=2,
                            step_dates=2, input_timezone='UTC')
    assert '3' not in {row['record_id'] for row in folds[0].train_rows}
    assert folds[0].train_audit['reasons']['label_not_available_at_boundary'] == 1
