from training.schema import LABEL_FIELDS


def label(record_id, departure, arrival, total):
    return {
        "record_id": record_id,
        "features": {"departure_airport": departure, "arrival_airport": arrival},
        "labels": {"off_block_delay_min": total, "taxi_out_min": 0,
                   "airborne_min": 0, "taxi_in_min": 0},
    }


def prediction(record_id, total):
    return {"record_id": record_id, "off_block_delay_min": total,
            "taxi_out_min": 0, "airborne_min": 0, "taxi_in_min": 0}


def test_fit_policy_only_allows_groups_with_validation_evidence():
    from training.guarded import fit_policy

    labels = [label("a", "GOOD", "DEST", 10), label("b", "GOOD", "DEST", 12),
              label("c", "BAD1", "DEST", 10), label("d", "BAD1", "DEST", 10)]
    candidate = [prediction("a", 10), prediction("b", 12),
                 prediction("c", 30), prediction("d", 30)]
    baseline = [prediction(key, 0) for key in ("a", "b", "c", "d")]

    policy = fit_policy(labels, candidate, baseline, minimum_group_size=2)

    assert policy["eligible_departure_airports"] == ["GOOD"]
    assert policy["eligible_arrival_airports"] == ["DEST"]
    assert policy["validation_rows"] == 4


def test_apply_policy_requires_both_airports_to_be_eligible():
    from training.guarded import apply_policy

    rows = [label("a", "GOOD", "DEST", 0), label("b", "GOOD", "OTHER", 0)]
    candidate = [prediction("a", 7), prediction("b", 8)]
    baseline = [prediction("a", 1), prediction("b", 2)]
    policy = {"eligible_departure_airports": ["GOOD"],
              "eligible_arrival_airports": ["DEST"]}

    result = apply_policy(rows, candidate, baseline, policy)

    assert result[0]["off_block_delay_min"] == 7.0
    assert result[1]["off_block_delay_min"] == 2.0
    assert all(set(row) == {"record_id", *LABEL_FIELDS} for row in result)


def test_fit_policy_rejects_prediction_id_mismatch():
    from training.guarded import fit_policy

    rows = [label("a", "GOOD", "DEST", 0)]
    try:
        fit_policy(rows, [prediction("a", 0)], [prediction("b", 0)], 1)
    except ValueError as error:
        assert "record_id" in str(error)
    else:
        raise AssertionError("mismatched records were accepted")
