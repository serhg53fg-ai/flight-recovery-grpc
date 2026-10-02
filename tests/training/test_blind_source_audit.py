from pathlib import Path

import pandas as pd

from training.blind_source_audit import audit_blind_source
from training.schema import FEATURE_COLUMNS, TARGET_COLUMNS


def workbook(path: Path, *, date='2025-06-01', missing_actual=False):
    row = {name: None for name in FEATURE_COLUMNS}
    row.update({name: f'{date} 08:00:00' for name in TARGET_COLUMNS})
    row.update({'计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZBAA',
                '计划离港时间': f'{date} 08:00:00', '计划到港时间': f'{date} 10:00:00'})
    if missing_actual:
        row['实际到港时间'] = None
    pd.DataFrame([row]).to_excel(path, index=False)


def test_blind_source_rejects_old_dates(tmp_path):
    path = tmp_path / 'old.xlsx'
    workbook(path, date='2025-05-30')
    audit = audit_blind_source(path, after_date='2025-05-30', min_rows=1)
    assert audit['eligible'] is False
    assert 'not_after_previous_dataset' in audit['blocks']
    assert audit['zggg_rows'] == 1


def test_blind_source_requires_complete_targets_and_minimum_rows(tmp_path):
    path = tmp_path / 'new.xlsx'
    workbook(path, missing_actual=True)
    audit = audit_blind_source(path, after_date='2025-05-30', min_rows=2)
    assert audit['eligible'] is False
    assert 'insufficient_zggg_rows' in audit['blocks']
    assert 'missing_actual_times' in audit['blocks']


def test_blind_source_accepts_new_labeled_rows(tmp_path):
    path = tmp_path / 'new.xlsx'
    workbook(path)
    audit = audit_blind_source(path, after_date='2025-05-30', min_rows=1)
    assert audit['eligible'] is True
    assert audit['date_min'] == '2025-06-01'
    assert audit['date_max'] == '2025-06-01'
    assert len(audit['source_sha256']) == 64


def test_reused_may_dates_not_accepted_as_new_blind_set(tmp_path):
    path = tmp_path / 'reused-may.xlsx'
    workbook(path, date='2025-05-25')
    audit = audit_blind_source(path, after_date='2025-05-30', min_rows=1)
    assert audit['eligible'] is False
    assert 'not_after_previous_dataset' in audit['blocks']
