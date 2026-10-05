import json

import pytest


def rows():
    candidate = [
        {"record_id": "a", "off_block_delay_min": 40, "taxi_out_min": 9,
         "airborne_min": 80, "taxi_in_min": 6, "elapsed_ms": 12},
        {"record_id": "b", "off_block_delay_min": 30, "taxi_out_min": 10,
         "airborne_min": 90, "taxi_in_min": 7, "elapsed_ms": 13},
    ]
    baseline = [
        {"record_id": "b", "off_block_delay_min": 2, "taxi_out_min": 20,
         "airborne_min": 100, "taxi_in_min": 8},
        {"record_id": "a", "off_block_delay_min": -1, "taxi_out_min": 21,
         "airborne_min": 101, "taxi_in_min": 9},
    ]
    return candidate, baseline


def test_hybrid_replaces_only_selected_components_and_preserves_order():
    from training.hybrid import compose_predictions

    candidate, baseline = rows()
    result = compose_predictions(candidate, baseline, ("off_block_delay_min",))

    assert result == [
        {"record_id": "a", "off_block_delay_min": -1.0, "taxi_out_min": 9.0,
         "airborne_min": 80.0, "taxi_in_min": 6.0},
        {"record_id": "b", "off_block_delay_min": 2.0, "taxi_out_min": 10.0,
         "airborne_min": 90.0, "taxi_in_min": 7.0},
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invalid_field", "non_finite"])
def test_hybrid_rejects_incomplete_or_invalid_inputs(mutation):
    from training.hybrid import compose_predictions

    candidate, baseline = rows()
    if mutation == "missing":
        baseline.pop()
    elif mutation == "duplicate":
        baseline[1]["record_id"] = "b"
    elif mutation == "invalid_field":
        with pytest.raises(ValueError, match="component"):
            compose_predictions(candidate, baseline, ("unknown",))
        return
    else:
        baseline[0]["off_block_delay_min"] = float("nan")
    with pytest.raises(ValueError):
        compose_predictions(candidate, baseline, ("off_block_delay_min",))


def test_compose_hybrid_cli_writes_new_prediction_file(tmp_path):
    from training.cli import main

    candidate, baseline = rows()
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    for path, values in ((candidate_path, candidate), (baseline_path, baseline)):
        path.write_text("".join(json.dumps(row) + "\n" for row in values))

    output = tmp_path / "hybrid.jsonl"
    assert main(["compose-hybrid", "--candidate", str(candidate_path),
                 "--baseline", str(baseline_path), "--output", str(output),
                 "--baseline-fields", "off_block_delay_min"]) == 0
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["off_block_delay_min"] for row in result] == [-1.0, 2.0]
    assert [row["airborne_min"] for row in result] == [80.0, 90.0]
