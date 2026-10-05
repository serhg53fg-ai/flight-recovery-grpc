from training.baselines import GradientBoostingBaseline, HistoricalMedianBaseline, ScheduleZeroBaseline
from tests.training.test_features import records


LABELS = [
    {"off_block_delay_min": 0.0, "taxi_out_min": 10.0, "airborne_min": 100.0, "taxi_in_min": 5.0},
    {"off_block_delay_min": 20.0, "taxi_out_min": 20.0, "airborne_min": 120.0, "taxi_in_min": 15.0},
]


def labelled():
    values = records()
    for value, labels in zip(values, LABELS): value["labels"] = labels
    return values


def test_schedule_zero_uses_zero_delay_and_train_duration_medians():
    model = ScheduleZeroBaseline().fit(labelled())
    prediction = model.predict(records()[:1])[0]
    assert prediction == {"record_id": "a", "off_block_delay_min": 0.0,
                          "taxi_out_min": 15.0, "airborne_min": 95.0, "taxi_in_min": 10.0}


def test_historical_median_falls_back_from_route_to_global():
    model = HistoricalMedianBaseline().fit(labelled())
    known, unknown = records()
    unknown["record_id"] = "x"
    unknown["features"]["departure_airport"] = "XXXX"
    result = model.predict([known, unknown])
    assert result[0]["taxi_out_min"] == 10.0
    assert result[1]["taxi_out_min"] == 15.0


def test_gradient_boosting_returns_four_finite_outputs():
    training = labelled() * 6
    training = [{**row, "record_id": f"{row['record_id']}-{index}"} for index, row in enumerate(training)]
    model = GradientBoostingBaseline(random_state=7).fit(training)
    prediction = model.predict(records()[:1])[0]
    assert prediction["record_id"] == "a"
    assert all(isinstance(prediction[name], float) for name in LABELS[0])


def test_gradient_boosting_serialization_round_trip():
    import pickle
    model = GradientBoostingBaseline(random_state=7).fit(labelled() * 3)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict(records()) == model.predict(records())


def test_weather_aware_baseline_responds_to_cutoff_metar():
    import copy
    from training.baselines import ZgggWeatherGradientBoostingBaseline
    seed = labelled()[0]
    seed['features']['prediction_cutoff_time'] = seed['features']['planned_off_block']
    seed['features']['departure_weather'] = {'metar': {'missing': False, 'stale': False,
        'issue_time': '2025-04-30T23:40:00Z', 'report_age_minutes': 20,
        'features': {'precipitation': False}}}
    wet = copy.deepcopy(seed)
    wet['record_id'] = 'wet'
    wet['features']['departure_weather']['metar']['features']['precipitation'] = True
    wet['labels']['off_block_delay_min'] = 30.0
    model = ZgggWeatherGradientBoostingBaseline().fit([seed, wet] * 5)
    dry_prediction, wet_prediction = model.predict([seed, wet])
    assert model.features.state()['version'] == 'zggg-cutoff-weather-features-v1'
    assert dry_prediction['off_block_delay_min'] < wet_prediction['off_block_delay_min']
