from datetime import datetime, timedelta, timezone

import pytest

from apps.flight.recovery.may_replay import select_rotations, build_may_replay, main
from apps.flight.recovery.domain import inputs


ZONE = timezone(timedelta(hours=8))


def row(identity, tail, origin, destination, departure, *, delay=0):
    start = datetime.fromisoformat(departure).astimezone(timezone.utc)
    end = start + timedelta(minutes=60)
    return {"record_id": identity, "features": {
        "tail_number": tail, "departure_airport": origin, "arrival_airport": destination,
        "planned_off_block": start.isoformat(), "planned_on_block": end.isoformat(),
    }, "labels": {"off_block_delay_min": delay, "taxi_out_min": 10,
                   "airborne_min": 40, "taxi_in_min": 10}}


def test_selection_requires_contiguous_within_day_rotations():
    rows = [row("a", "B1", "ZGGG", "ZBAA", "2025-05-10T08:00:00+08:00"),
            row("b", "B1", "ZBAA", "ZGGG", "2025-05-10T10:00:00+08:00"),
            row("c", "B1", "ZGGG", "ZBAA", "2025-05-10T12:00:00+08:00"),
            row("bad", "B2", "ZGGG", "ZBAA", "2025-05-10T08:00:00+08:00"),
            row("broken", "B2", "ZGGG", "ZBAA", "2025-05-10T10:00:00+08:00")]
    selected = select_rotations(rows, "2025-05-10", max_tails=1)
    assert [item["record_id"] for item in selected] == ["a", "b", "c"]


def test_replay_uses_only_prior_completed_labels_and_planned_features():
    prior = row("past", "P", "ZGGG", "ZBAA", "2025-05-08T08:00:00+08:00", delay=12)
    flights = [row("a", "B1", "ZGGG", "ZBAA", "2025-05-10T08:00:00+08:00", delay=999),
               row("b", "B1", "ZBAA", "ZGGG", "2025-05-10T10:00:00+08:00", delay=999),
               row("c", "B1", "ZGGG", "ZBAA", "2025-05-10T12:00:00+08:00", delay=999)]
    first = build_may_replay([prior, *flights], "2025-05-10", max_tails=1)
    changed = [{**item, "labels": {**item["labels"], "off_block_delay_min": -999}} for item in flights]
    second = build_may_replay([prior, *changed], "2025-05-10", max_tails=1)
    assert first["flights"] == second["flights"]
    assert first["source"] == "may_planned_flights_hypothetical_capacity"
    assert first["training_rows"] == 1
    assert len(first["flights"]) == 3
    inputs(first["flights"], first["config"])


def test_replay_normalizes_zulu_timestamps_for_scheduler():
    prior = row("past", "P", "ZGGG", "ZBAA", "2025-05-08T08:00:00+08:00")
    flights = [row("a", "B1", "ZGGG", "ZBAA", "2025-05-10T08:00:00+08:00"),
               row("b", "B1", "ZBAA", "ZGGG", "2025-05-10T10:00:00+08:00"),
               row("c", "B1", "ZGGG", "ZBAA", "2025-05-10T12:00:00+08:00")]
    for item in flights:
        for field in ("planned_off_block", "planned_on_block"):
            item["features"][field] = item["features"][field].replace("+00:00", "Z")
    replay = build_may_replay([prior, *flights], "2025-05-10", max_tails=1)
    inputs(replay["flights"], replay["config"])


def test_cli_refuses_to_overwrite_existing_report(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="new file"):
        main(["--dataset", str(tmp_path / "absent"), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "keep"


def test_replay_excludes_previous_day_label_completed_after_boundary():
    complete = row("complete", "P", "ZGGG", "ZBAA", "2025-05-08T08:00:00+08:00")
    late = row("late", "Q", "ZGGG", "ZBAA", "2025-05-09T23:30:00+08:00", delay=180)
    flights = [row("a", "B1", "ZGGG", "ZBAA", "2025-05-10T08:00:00+08:00"),
               row("b", "B1", "ZBAA", "ZGGG", "2025-05-10T10:00:00+08:00"),
               row("c", "B1", "ZGGG", "ZBAA", "2025-05-10T12:00:00+08:00")]
    replay = build_may_replay([complete, late, *flights], "2025-05-10", max_tails=1)
    assert replay["training_rows"] == 1
    assert replay["training_audit"]["reasons"]["label_not_available_at_boundary"] == 1
