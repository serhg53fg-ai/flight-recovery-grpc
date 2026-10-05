import pytest

from training.original_style_split import random_split, eligible_before


def _row(identity, time, delay=0):
    return {"record_id": identity, "features": {"planned_off_block": time},
            "labels": {"off_block_delay_min": delay, "taxi_out_min": 10,
                       "airborne_min": 40, "taxi_in_min": 10}}


def test_random_split_is_seeded_disjoint_and_has_fixed_fraction():
    rows = [_row(str(i), "2025-05-01T00:00:00Z") for i in range(10)]
    train, test = random_split(rows, seed=42, test_fraction=.2)
    assert len(train) == 8 and len(test) == 2
    assert {r["record_id"] for r in train}.isdisjoint({r["record_id"] for r in test})
    assert ([r["record_id"] for r in train], [r["record_id"] for r in test]) == (
        [r["record_id"] for r in random_split(rows, seed=42, test_fraction=.2)[0]],
        [r["record_id"] for r in random_split(rows, seed=42, test_fraction=.2)[1]])


def test_random_split_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        random_split([_row("x", "2025-05-01T00:00:00Z")] * 3)


def test_eligible_before_drops_labels_finished_after_boundary():
    rows = [_row("safe", "2025-05-08T00:00:00Z"),
            _row("late", "2025-05-09T23:30:00Z", delay=200),
            _row("future", "2025-05-10T00:00:00Z")]
    selected, audit = eligible_before(rows, "2025-05-10T00:00:00Z")
    assert [item["record_id"] for item in selected] == ["safe"]
    assert audit["excluded_rows"] == 2
