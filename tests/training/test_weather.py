from datetime import datetime, timezone

import pytest


UTC = timezone.utc


def test_parses_zggg_metar_issue_time_and_structured_features():
    from training.weather import parse_weather_report

    report = parse_weather_report(
        "METAR ZGGG 010025Z 20012G20KT 165V255 6000 -RA BKN008CB 23/21 A2992=",
        datetime(2025, 5, 1, 1, 0, tzinfo=UTC),
        source="archive",
    )

    assert report.kind == "METAR"
    assert report.airport == "ZGGG"
    assert report.issue_time == datetime(2025, 5, 1, 0, 25, tzinfo=UTC)
    assert report.parse_status == "OK"
    assert report.feature_dict() == {
        "cavok": False,
        "ceiling_ft": 800.0,
        "cumulonimbus": True,
        "dewpoint_c": 21.0,
        "precipitation": True,
        "pressure_hpa": pytest.approx(1013.2075, abs=0.01),
        "temperature_c": 23.0,
        "thunderstorm": False,
        "visibility_m": 6000.0,
        "wind_direction_deg": 200.0,
        "wind_gust_kt": 20.0,
        "wind_speed_kt": 12.0,
    }


def test_parses_mps_wind_from_zggg_historical_metar():
    from training.weather import parse_weather_report

    report = parse_weather_report(
        "METAR ZGGG 010030Z 04002G05MPS 9999 -RA SCT010 24/23 Q1014",
        datetime(2025, 5, 1, 1, 0, tzinfo=UTC),
    )

    assert report.feature_dict()["wind_direction_deg"] == 40.0
    assert report.feature_dict()["wind_speed_kt"] == pytest.approx(3.88769, abs=0.0001)
    assert report.feature_dict()["wind_gust_kt"] == pytest.approx(9.71922, abs=0.0001)


def test_month_boundary_resolves_previous_month_without_inventing_future_month():
    from training.weather import parse_weather_report

    report = parse_weather_report(
        "METAR ZGGG 312350Z 01005KT 9999 CAVOK 20/10 Q1012=",
        datetime(2025, 6, 1, 0, 10, tzinfo=UTC),
    )

    assert report.issue_time == datetime(2025, 5, 31, 23, 50, tzinfo=UTC)


def test_nearest_prior_metar_is_selected_and_future_report_is_rejected():
    from training.weather import parse_weather_report, select_weather_snapshot

    reference = datetime(2025, 5, 1, 1, 0, tzinfo=UTC)
    reports = [
        parse_weather_report("METAR ZGGG 010030Z 01005KT 9999 CLR 20/10 Q1012=", reference),
        parse_weather_report("METAR ZGGG 010100Z 02006KT 9000 CLR 21/11 Q1011=", reference),
        parse_weather_report("METAR ZGGG 010105Z 03007KT 8000 CLR 22/12 Q1010=", reference),
    ]

    snapshot = select_weather_snapshot(
        reports, cutoff_time=reference, target_time=reference, kind="METAR",
    )

    assert snapshot.report is not None
    assert snapshot.report.issue_time == reference
    assert snapshot.report_age_minutes == 0.0
    assert snapshot.missing is False


def test_taf_requires_prior_issue_and_target_coverage():
    from training.weather import parse_weather_report, select_weather_snapshot

    cutoff = datetime(2025, 5, 1, 2, 0, tzinfo=UTC)
    target = datetime(2025, 5, 1, 8, 0, tzinfo=UTC)
    valid = parse_weather_report(
        "TAF ZGGG 010100Z 0103/0112 18008KT 9999 SCT020=", cutoff,
    )
    wrong_window = parse_weather_report(
        "TAF ZGGG 010130Z 0113/0200 20008KT 9999 SCT020=", cutoff,
    )
    future_issue = parse_weather_report(
        "TAF ZGGG 010205Z 0103/0112 22008KT 9999 SCT020=", cutoff,
    )

    snapshot = select_weather_snapshot(
        [wrong_window, valid, future_issue], cutoff_time=cutoff,
        target_time=target, kind="TAF",
    )

    assert snapshot.report == valid
    assert valid.valid_from == datetime(2025, 5, 1, 3, 0, tzinfo=UTC)
    assert valid.valid_to == datetime(2025, 5, 1, 12, 0, tzinfo=UTC)


def test_missing_and_malformed_weather_remain_explicit():
    from training.weather import parse_weather_report, select_weather_snapshot

    malformed = parse_weather_report(
        "METAR ZGGG missing-time bad-data",
        datetime(2025, 5, 1, tzinfo=UTC),
        source="broken-feed",
    )
    snapshot = select_weather_snapshot(
        [malformed], cutoff_time=datetime(2025, 5, 1, tzinfo=UTC),
        target_time=datetime(2025, 5, 1, tzinfo=UTC), kind="METAR",
    )

    assert malformed.parse_status == "INVALID"
    assert malformed.issue_time is None
    assert malformed.raw_text == "METAR ZGGG missing-time bad-data"
    assert snapshot.report is None
    assert snapshot.missing is True
    assert snapshot.feature_dict()["weather_missing"] is True


def test_naive_reference_or_unknown_kind_is_rejected():
    from training.weather import parse_weather_report, select_weather_snapshot

    with pytest.raises(ValueError, match="timezone"):
        parse_weather_report("METAR ZGGG 010000Z 00000KT CAVOK 20/10 Q1012=",
                             datetime(2025, 5, 1))
    with pytest.raises(ValueError, match="kind"):
        select_weather_snapshot([], datetime(2025, 5, 1, tzinfo=UTC),
                                datetime(2025, 5, 1, tzinfo=UTC), "SIGMET")


def test_metar_current_conditions_ignore_later_becmg_trend():
    from training.weather import parse_weather_report

    report = parse_weather_report(
        'METAR ZGGG 040730Z 12004MPS 9999 FEW015 BKN050 28/24 Q1005 '
        'BECMG AT0800 TSRA SCT026CB BKN033',
        datetime(2025, 5, 4, 7, 30, tzinfo=UTC),
        feature_version='aviation-weather-v2',
    )
    features = report.feature_dict()
    assert features['thunderstorm'] is False
    assert features['precipitation'] is False
    assert features['ceiling_ft'] == 5000
    assert 'BECMG' in report.raw_text


def test_metar_current_tsra_marks_both_thunderstorm_and_precipitation():
    from training.weather import parse_weather_report

    report = parse_weather_report(
        'METAR ZGGG 040800Z 10004MPS 9999 -TSRA FEW015 BKN033 29/24 Q1004 '
        'BECMG AT0810 +TSRA SCT026CB BKN033',
        datetime(2025, 5, 4, 8, 0, tzinfo=UTC),
        feature_version='aviation-weather-v2',
    )
    assert report.feature_dict()['thunderstorm'] is True
    assert report.feature_dict()['precipitation'] is True


def test_v1_weather_feature_semantics_remain_compatible():
    from training.weather import parse_weather_report

    report = parse_weather_report(
        'METAR ZGGG 040730Z 12004MPS 9999 BKN050 28/24 Q1005 '
        'BECMG AT0800 TSRA BKN033',
        datetime(2025, 5, 4, 7, 30, tzinfo=UTC),
    )
    assert report.feature_dict()['thunderstorm'] is True
    assert report.feature_dict()['precipitation'] is False
