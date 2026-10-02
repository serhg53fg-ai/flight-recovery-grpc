from datetime import datetime, timedelta, timezone

from training.schema import LABEL_FIELDS


def row(day, delay, *, adverse=False):
    stamp = (datetime(2025, 5, 1, tzinfo=timezone.utc) + timedelta(days=day)).isoformat()
    return {'record_id': str(day), 'features': {
        'planned_off_block': stamp, 'departure_airport': 'ZGGG',
        'arrival_airport': 'ZBAA', 'prediction_cutoff_time': stamp,
        'departure_weather': {'metar': {'missing': False, 'stale': False,
            'issue_time': stamp, 'features': {'precipitation': adverse}}},
    }, 'labels': {'off_block_delay_min': delay, 'taxi_out_min': 10,
                  'airborne_min': 100, 'taxi_in_min': 5}}


def test_weather_residual_uses_only_prior_date_errors_and_caps_adjustment():
    from training.weather_residual import WeatherResidualBaseline

    rows = [row(i, 0 if i < 4 else 100, adverse=i >= 4) for i in range(8)]
    model = WeatherResidualBaseline(min_bucket_samples=2, shrink_samples=0,
                                    max_adjustment_min=8, initial_dates=3).fit(rows)
    dry, wet = model.predict([row(8, 0), row(9, 0, adverse=True)])
    assert dry['off_block_delay_min'] == 50
    assert 0 < wet['off_block_delay_min'] - dry['off_block_delay_min'] <= 8
    assert all(wet[field] == dry[field] for field in LABEL_FIELDS if field != 'off_block_delay_min')


def test_weather_residual_falls_back_when_bucket_has_no_support():
    from training.weather_residual import WeatherResidualBaseline

    rows = [row(i, 0) for i in range(6)]
    model = WeatherResidualBaseline(min_bucket_samples=2, initial_dates=3).fit(rows)
    prediction = model.predict([row(7, 0, adverse=True)])[0]
    assert prediction['off_block_delay_min'] == 0


def test_adverse_only_mode_keeps_normal_weather_at_historical_baseline():
    from training.baselines import HistoricalMedianBaseline
    from training.weather_residual import WeatherResidualBaseline

    rows = [row(i, 0 if i < 4 else 100, adverse=i >= 4) for i in range(8)]
    query = [row(8, 0), row(9, 0, adverse=True)]
    model = WeatherResidualBaseline(min_bucket_samples=2, shrink_samples=0,
                                    max_adjustment_min=8, initial_dates=3,
                                    adverse_only=True).fit(rows)
    base = HistoricalMedianBaseline().fit(rows).predict(query)
    result = model.predict(query)
    assert result[0] == base[0]
    assert result[1]['off_block_delay_min'] > base[1]['off_block_delay_min']
