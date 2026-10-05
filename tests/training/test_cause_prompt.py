import json

import pytest

from training.prompt import (parse_cause_duration_output, parse_duration_output,
                             render_cause_duration_prompt)


COMPONENTS = {"off_block_delay_min": 5, "taxi_out_min": 20,
              "airborne_min": 100, "taxi_in_min": 10}


def test_experimental_parser_keeps_legacy_contract_separate():
    assert parse_duration_output(json.dumps(COMPONENTS))["taxi_out_min"] == 20
    values, cause = parse_cause_duration_output(json.dumps({**COMPONENTS, "reason_code": "unknown"}))
    assert values["taxi_out_min"] == 20
    assert cause == "unknown"
    with pytest.raises(ValueError):
        parse_duration_output(json.dumps({**COMPONENTS, "reason_code": "unknown"}))


@pytest.mark.parametrize("extra", [{"reason_code": "bad"}, {"reason_code": "weather", "other": 1},
                                    {"reason_code": ["weather"]},
                                    {"reason_code": "weather", "taxi_in_min": float("nan")}])
def test_experimental_parser_rejects_invalid_fields(extra):
    with pytest.raises(ValueError):
        parse_cause_duration_output(json.dumps({**COMPONENTS, **extra}))


def test_prompt_uses_cutoff_safe_fields_and_does_not_assert_previous_flight():
    prompt = render_cause_duration_prompt({
        "planned_off_block": "2025-05-01T00:00:00Z", "prediction_cutoff_time": "2025-04-30T23:00:00Z",
        "actual_off_block": "2025-05-01T03:00:00Z", "previous_flight_delay": 120,
        "departure_weather": {"metar": {"missing": True}, "taf": {"missing": True}},
    })
    assert "actual_off_block" not in prompt
    assert "previous_flight_delay" not in prompt
    assert "previous_flight is unavailable" in prompt


def test_prompt_rejects_future_weather():
    with pytest.raises(ValueError, match="future weather"):
        render_cause_duration_prompt({
            "prediction_cutoff_time": "2025-04-30T23:00:00Z",
            "departure_weather": {"metar": {"missing": False, "issue_time": "2025-05-01T00:00:00Z"}},
        })


def test_parser_cannot_assert_previous_flight_without_evidence():
    with pytest.raises(ValueError, match="previous-flight evidence"):
        parse_cause_duration_output(json.dumps({**COMPONENTS, "reason_code": "previous_flight"}))
    _, reason = parse_cause_duration_output(json.dumps({**COMPONENTS, "reason_code": "previous_flight"}),
                                            previous_flight_evidence=True)
    assert reason == "previous_flight"
