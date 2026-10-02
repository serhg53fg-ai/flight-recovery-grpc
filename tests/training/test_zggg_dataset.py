import hashlib
import json
from datetime import timedelta

import pandas as pd
import pytest

from tests.training.test_dataset import read_jsonl, row, source
from training.dataset import DatasetConfig
from training.schema import PROHIBITED_FEATURE_COLUMNS


def all_rows(dataset):
    return sum((read_jsonl(dataset / f"{name}.jsonl")
                for name in ("train", "validation", "test")), [])


def prior_zggg_metar(day):
    issue_day = 30 if day == 1 else day - 1
    return f"METAR ZGGG {issue_day:02d}2350Z 18005KT 9999 CLR 20/10 Q1013="


def test_builds_only_zggg_arrivals_and_departures_with_cutoff_weather(tmp_path):
    from training.zggg_dataset import build_zggg_dataset

    rows = []
    for day in range(1, 13):
        departure = row(day)
        departure["起飞站METAR"] = prior_zggg_metar(day)
        departure["到达站METAR"] = f"METAR ZBAA {day:02d}1000Z 20004KT 8000 FEW020 18/08 Q1015="
        arrival = row(day, 1)
        arrival["计划起飞站四字码"] = "ZBAA"
        arrival["实际起飞站四字码"] = "ZBAA"
        arrival["计划到达站四字码"] = "ZGGG"
        arrival["实际到达站四字码"] = "ZGGG"
        arrival["起飞站METAR"] = f"METAR ZBAA {day:02d}0755Z 19004KT 9000 CLR 18/08 Q1015="
        arrival["到达站METAR"] = f"METAR ZGGG {day:02d}1000Z 21006KT 7000 -RA BKN010 22/20 Q1012="
        outside = row(day, 2)
        outside["计划起飞站四字码"] = outside["实际起飞站四字码"] = "ZBAA"
        outside["计划到达站四字码"] = outside["实际到达站四字码"] = "ZSPD"
        rows.extend((departure, arrival, outside))
    source_path = source(tmp_path, rows)
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()

    manifest = build_zggg_dataset(
        source_path, tmp_path / "zggg", DatasetConfig(min_test_rows=1),
    )
    records = all_rows(tmp_path / "zggg")

    assert manifest.accepted_rows == 24
    assert {item["features"]["airport_direction"] for item in records} == {"ARRIVAL", "DEPARTURE"}
    assert all("ZGGG" in (item["features"]["departure_airport"],
                           item["features"]["arrival_airport"]) for item in records)
    assert all(item["features"]["prediction_cutoff_time"] ==
               item["features"]["planned_off_block"] for item in records)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before

    first_departure = next(item for item in records
                           if item["features"]["airport_direction"] == "DEPARTURE")
    assert first_departure["features"]["departure_weather"]["metar"]["missing"] is False
    assert first_departure["features"]["arrival_weather"]["metar"]["missing"] is True
    encoded = json.dumps(records)
    assert not set(first_departure["features"]).intersection(PROHIBITED_FEATURE_COLUMNS)
    assert "actual_total_flow" not in encoded and "实际总流量" not in encoded


def test_archive_uses_explicit_prior_issue_time_and_keeps_taf_missing(tmp_path):
    from training.zggg_dataset import build_zggg_dataset

    rows = [row(day) for day in range(1, 13)]
    archive = tmp_path / "weather.csv"
    pd.DataFrame([{
        "kind": "METAR", "airport": "ZGGG",
        "issue_time": "2025-05-01T07:45:00+08:00",
        "report_text": "METAR ZGGG 302345Z 12003KT 5000 BR BKN006 19/18 Q1010=",
        "source": "caac-archive",
    }]).to_csv(archive, index=False)

    build_zggg_dataset(source(tmp_path, rows), tmp_path / "zggg",
                       DatasetConfig(min_test_rows=1), weather_archive=archive)
    manifest = json.loads((tmp_path / "zggg/manifest.json").read_text())
    records = all_rows(tmp_path / "zggg")
    weather = records[0]["features"]["departure_weather"]

    assert weather["metar"]["issue_time"] == "2025-04-30T23:45:00Z"
    assert weather["metar"]["source"] == "caac-archive"
    assert weather["taf"]["missing"] is True
    assert manifest["weather_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_v2_weather_features_get_distinct_dataset_identity(tmp_path):
    from training.zggg_dataset import build_zggg_dataset, verify_zggg_dataset

    rows = [row(day) for day in range(1, 13)]
    rows[0]['起飞站METAR'] = ('METAR ZGGG 010000Z 10004MPS 9999 -TSRA '
                              'BKN033 29/24 Q1004 BECMG AT0010 +TSRA BKN020')
    workbook = source(tmp_path, rows)
    first = build_zggg_dataset(workbook, tmp_path / 'v1', DatasetConfig(min_test_rows=1))
    second = build_zggg_dataset(workbook, tmp_path / 'v2', DatasetConfig(min_test_rows=1),
                                weather_feature_version='aviation-weather-v2')
    assert first.manifest_hash != second.manifest_hash
    v1 = json.loads((tmp_path / 'v1/manifest.json').read_text())
    v2 = json.loads((tmp_path / 'v2/manifest.json').read_text())
    assert v1['weather_feature_version'] == 'aviation-weather-v1'
    assert v2['weather_feature_version'] == 'aviation-weather-v2'
    assert verify_zggg_dataset(tmp_path / 'v2')['manifest_hash'] == second.manifest_hash
    assert all_rows(tmp_path / 'v2')[0]['features']['departure_weather']['metar']['features']['precipitation'] is True


def test_manifest_audits_scope_weather_and_chronological_splits(tmp_path):
    from training.zggg_dataset import build_zggg_dataset, verify_zggg_dataset

    dataset = tmp_path / "zggg"
    rows = [row(day) for day in range(1, 13)]
    for day, item in enumerate(rows, start=1):
        item["起飞站METAR"] = prior_zggg_metar(day)
    build_zggg_dataset(source(tmp_path, rows), dataset,
                       DatasetConfig(min_test_rows=1))
    manifest = verify_zggg_dataset(dataset)
    audit = json.loads((dataset / "audit.json").read_text())
    dates = [manifest["splits"][name]["dates"] for name in ("train", "validation", "test")]

    assert manifest["schema_version"] == "zggg-flight-weather-v1"
    assert manifest["airport"] == "ZGGG"
    assert dates[0][-1] < dates[1][0] < dates[2][0]
    assert audit["scope_rejections"] == 0
    assert audit["weather"]["departure_metar_available"] == 12
    assert audit["weather"]["taf_available"] == 0


def test_verifier_rejects_future_weather_or_tampered_split(tmp_path):
    from training.zggg_dataset import build_zggg_dataset, verify_zggg_dataset

    dataset = tmp_path / "zggg"
    build_zggg_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), dataset,
                       DatasetConfig(min_test_rows=1))
    records = read_jsonl(dataset / "test.jsonl")
    records[0]["features"]["departure_weather"]["metar"]["issue_time"] = (
        pd.Timestamp(records[0]["features"]["prediction_cutoff_time"]) + timedelta(minutes=1)
    ).isoformat().replace("+00:00", "Z")
    text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
    (dataset / "test.jsonl").write_text(text)
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["splits"]["test"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (dataset / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="cutoff"):
        verify_zggg_dataset(dataset)
