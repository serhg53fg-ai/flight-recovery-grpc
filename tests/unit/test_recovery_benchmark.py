from apps.flight.recovery.benchmark import build_scenarios, compare_policies
from apps.flight.recovery.domain import inputs, rotations


def test_fixed_scenarios_contain_complete_rotations_and_pressure_cases():
    cases = build_scenarios()
    assert {case["name"] for case in cases} == {"base", "tight_departure", "arrival_closure", "short_horizon"}
    for case in cases:
        flights, _ = inputs(case["flights"], case["config"])
        groups = rotations(flights)
        assert len(flights) >= 24
        assert len(groups) >= 8
        assert all(len(group) >= 3 for group in groups.values())
        assert {len(group) for group in groups.values()} == {3, 4}
        assert all(all(a.destination == b.origin for a, b in zip(group, group[1:]))
                   for group in groups.values())
    assert any(case["config"]["closures"] for case in cases)


def test_comparison_is_deterministic_and_independently_validated():
    first = compare_policies(build_scenarios())
    second = compare_policies(build_scenarios())
    assert [x["name"] for x in first["scenarios"]] == [x["name"] for x in second["scenarios"]]
    for left, right in zip(first["scenarios"], second["scenarios"]):
        for policy in ("earliest", "rotation_urgency"):
            assert left[policy]["validation_errors"] == []
            assert left[policy]["unassigned_count"] >= 0
            assert left[policy]["total_delay_minutes"] >= 0
            assert left[policy]["max_delay_minutes"] == right[policy]["max_delay_minutes"]
            assert left[policy]["total_delay_minutes"] == right[policy]["total_delay_minutes"]
    by_name = {item["name"]: item for item in first["scenarios"]}
    assert by_name["arrival_closure"]["earliest"]["total_delay_minutes"] > by_name["base"]["earliest"]["total_delay_minutes"]
    assert any(item["earliest"]["total_delay_minutes"] != item["rotation_urgency"]["total_delay_minutes"]
               for item in first["scenarios"])
    assert by_name["short_horizon"]["earliest"]["unassigned_count"] > 0
