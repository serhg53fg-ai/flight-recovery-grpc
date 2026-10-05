import pandas as pd


def test_feature_builder_excludes_future_actual_events():
    from training.flow_features import build_flow_features

    rows = pd.DataFrame([
        {'计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
         '计划离港时间': '2025-05-02T00:35:00Z', '计划到港时间': '2025-05-02T02:00:00Z',
         '实际起飞时间': '2025-05-02T00:40:00Z'},
        {'计划起飞站四字码': 'ZSPD', '计划到达站四字码': 'ZGGG',
         '计划离港时间': '2025-05-01T22:00:00Z', '计划到港时间': '2025-05-02T00:45:00Z',
         '实际落地时间': '2025-05-02T00:48:00Z'},
    ])
    cutoff = '2025-05-02T00:30:00Z'
    first = build_flow_features(rows, [cutoff])[0]
    assert first['planned_takeoff_15m'] == 1
    assert first['planned_landing_30m'] == 1
    assert first['completed_takeoff_15m'] == 0
    assert first['completed_landing_60m'] == 0
    rows.loc[0, '实际起飞时间'] = '2025-05-02T00:55:00Z'
    rows.loc[1, '实际落地时间'] = '2025-05-02T01:10:00Z'
    assert build_flow_features(rows, [cutoff])[0] == first


def test_feature_builder_sees_completed_event_before_cutoff():
    from training.flow_features import build_flow_features

    rows = pd.DataFrame([{'计划起飞站四字码': 'ZGGG', '计划离港时间': '2025-05-02T00:10:00Z',
                          '实际起飞时间': '2025-05-02T00:25:00Z'}])
    result = build_flow_features(rows, ['2025-05-02T00:30:00Z'])[0]
    assert result['completed_takeoff_15m'] == 1
    assert result['completed_takeoff_30m'] == 1
