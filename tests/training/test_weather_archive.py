import hashlib

import pandas as pd


def test_workbook_archive_uses_report_time_and_deduplicates(tmp_path):
    from training.weather_archive import export_workbook_metar_archive

    report = "METAR ZGGG 010025Z 18005KT 9999 CLR 20/10 Q1013="
    source = tmp_path / "flights.xlsx"
    pd.DataFrame([
        {"计划离港时间": "2025-05-01 08:00:00", "计划起飞站四字码": "ZGGG",
         "计划到达站四字码": "ZBAA", "起飞站METAR": report,
         "到达站METAR": "METAR ZBAA 010025Z 18005KT 9999 CLR 20/10 Q1013="},
        {"计划离港时间": "2025-05-01 09:00:00", "计划起飞站四字码": "ZGGG",
         "计划到达站四字码": "ZBAA", "起飞站METAR": report,
         "到达站METAR": "METAR ZGGG 010025Z 18005KT 9999 CLR 20/10 Q1013="},
    ]).to_excel(source, index=False)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    audit = export_workbook_metar_archive(source, tmp_path / "archive")
    rows = pd.read_csv(tmp_path / "archive/reports.csv").to_dict("records")

    assert len(rows) == 2  # ZGGG and ZBAA; repeated ZGGG is one report.
    zggg = next(item for item in rows if item["airport"] == "ZGGG")
    assert zggg["issue_time"] == "2025-05-01T00:25:00Z"
    assert zggg["source"] == "workbook-derived-metar-v1"
    assert audit["unique_reports"] == 2
    assert audit["airport_mismatch"] == 1
    assert audit["future_relative_to_source_row"] == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash


def test_workbook_archive_refuses_existing_output(tmp_path):
    from training.weather_archive import export_workbook_metar_archive

    source = tmp_path / "flights.xlsx"
    pd.DataFrame([{"计划离港时间": "2025-05-01 09:00:00",
                   "计划起飞站四字码": "ZGGG", "计划到达站四字码": "ZBAA",
                   "起飞站METAR": "METAR ZGGG 010025Z 18005KT 9999 CLR 20/10 Q1013="}]).to_excel(source, index=False)
    output = tmp_path / "archive"
    output.mkdir()

    import pytest
    with pytest.raises(ValueError, match="new directory"):
        export_workbook_metar_archive(source, output)


def test_iem_archive_uses_independent_valid_time_and_rejects_bad_rows(tmp_path):
    from training.weather_archive import export_iem_metar_archive

    source = tmp_path / "iem.csv"
    pd.DataFrame([
        {"station": "ZGGG", "valid": "2025-05-01 00:30",
         "metar": "ZGGG 010030Z 04002MPS 9999 -RA SCT010 24/23 Q1014"},
        {"station": "ZGGG", "valid": "2025-05-01 00:31",
         "metar": "ZGGG 010030Z 04002MPS 9999 -RA SCT010 24/23 Q1014"},
        {"station": "ZBAA", "valid": "2025-05-01 00:30",
         "metar": "ZBAA 010030Z 04002MPS 9999 CLR 24/23 Q1014"},
    ]).to_csv(source, index=False)

    audit = export_iem_metar_archive(source, tmp_path / "archive", airport="ZGGG")
    rows = pd.read_csv(tmp_path / "archive/reports.csv").to_dict("records")

    assert len(rows) == 1
    assert rows[0]["issue_time"] == "2025-05-01T00:30:00Z"
    assert rows[0]["report_text"].startswith("METAR ZGGG 010030Z")
    assert rows[0]["source"] == "iem-asos-zggg-v1"
    assert audit["timestamp_mismatch"] == 1
    assert audit["outside_airport"] == 1
