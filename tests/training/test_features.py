import copy

from training.features import FeaturePipeline


def records():
    return [
        {"record_id": "a", "features": {"flight_number": "CZ1", "tail_number": "B1", "aircraft_type": "A320", "flight_nature": "J", "departure_airport": "ZGGG", "arrival_airport": "ZBAA", "planned_off_block": "2025-05-01T00:00:00Z", "planned_on_block": "2025-05-01T02:00:00Z", "planned_distance_miles": 1000, "planned_flight_minutes": 120, "planned_takeoff_count": 10, "planned_landing_count": 10, "planned_total_flow": 20, "departure_metar": "18005KT 8000 20/10 Q1013", "arrival_metar": "20004KT 9999 18/08 Q1015"}, "labels": {}},
        {"record_id": "b", "features": {"flight_number": "CZ2", "tail_number": "B2", "aircraft_type": "B738", "flight_nature": "J", "departure_airport": "ZGGG", "arrival_airport": "ZSPD", "planned_off_block": "2025-05-02T01:00:00Z", "planned_on_block": "2025-05-02T03:00:00Z", "planned_distance_miles": 900, "planned_flight_minutes": 110, "planned_takeoff_count": 12, "planned_landing_count": 8, "planned_total_flow": 20, "departure_metar": "VRB03KT CAVOK 21/11 Q1012", "arrival_metar": "19006KT 7000 17/07 Q1014"}, "labels": {}}
    ]


def test_fit_state_is_train_only_and_transform_is_deterministic():
    pipeline = FeaturePipeline().fit(records())
    before = copy.deepcopy(pipeline.state())
    validation = copy.deepcopy(records()[0])
    validation["features"]["departure_airport"] = "NEVER_SEEN"
    validation["features"]["planned_total_flow"] = 999999
    first = pipeline.transform([validation])
    second = pipeline.transform([validation])
    assert pipeline.state() == before
    assert first.tolist() == second.tolist()
    assert first.shape[0] == 1


def test_metar_and_time_features_have_no_label_dependency():
    first = records()[0]
    second = copy.deepcopy(first)
    first["labels"] = {"off_block_delay_min": -1000}
    second["labels"] = {"off_block_delay_min": 1000}
    pipeline = FeaturePipeline().fit(records())
    assert pipeline.transform([first]).tolist() == pipeline.transform([second]).tolist()


def test_observation_metars_at_planned_events_do_not_change_features():
    first = records()[0]
    second = copy.deepcopy(first)
    second["features"]["arrival_metar"] = "99999KT 0000 M99/M99 Q0001"
    second["features"]["departure_metar"] = "99999KT 0000 M99/M99 Q0001"
    pipeline = FeaturePipeline().fit(records())
    assert pipeline.transform([first]).tolist() == pipeline.transform([second]).tolist()


def test_zggg_weather_pipeline_uses_only_cutoff_safe_airport_metar():
    import pytest
    from training.features import ZgggWeatherFeaturePipeline
    first = records()[0]
    first['features']['airport_direction'] = 'DEPARTURE'
    first['features']['prediction_cutoff_time'] = '2025-05-01T00:00:00Z'
    first['features']['departure_weather'] = {'metar': {
        'missing': False, 'stale': False, 'issue_time': '2025-04-30T23:40:00Z',
        'report_age_minutes': 20, 'features': {'wind_speed_kt': 5, 'visibility_m': 9999,
                                               'precipitation': False}}}
    pipeline = ZgggWeatherFeaturePipeline().fit([first])
    changed = copy.deepcopy(first)
    changed['features']['departure_weather']['metar']['features']['precipitation'] = True
    assert pipeline.transform([first]).tolist() != pipeline.transform([changed]).tolist()
    changed['features']['departure_weather']['metar']['issue_time'] = '2025-05-01T00:01:00Z'
    with pytest.raises(ValueError, match='future weather'):
        pipeline.transform([changed])
