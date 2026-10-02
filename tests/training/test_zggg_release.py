import hashlib
import json
from pathlib import Path

import pytest


def _artifact(path: Path, content: bytes = b"artifact") -> dict:
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def release_input(tmp_path: Path) -> dict:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({
        "airport": "ZGGG", "manifest_hash": "dataset-v1", "publishable": True,
        "causality_audit": {"future_weather_rows": 0},
    }), encoding="utf-8")
    return {
        "release_version": "zggg-20260923-01",
        "dataset_manifest": str(dataset / "manifest.json"),
        "flight_gate": {"passed": True},
        "flow_gate": {"passed": True},
        "artifacts": {
            name: _artifact(tmp_path / f"{name}.bin", name.encode())
            for name in ("model", "adapter", "flow_model", "historical_model", "feature_contract")
        },
    }


def test_stages_immutable_candidate_with_verified_hashes(tmp_path):
    from scripts.zggg_release import stage_candidate

    output = tmp_path / "release"
    result = stage_candidate(release_input(tmp_path), output)

    manifest = json.loads((output / "candidate.json").read_text())
    assert result == manifest
    assert manifest["airport"] == "ZGGG"
    assert manifest["dataset_manifest_hash"] == "dataset-v1"
    assert manifest["flight_gate_passed"] is True
    assert manifest["flow_gate_passed"] is True


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.update(release_version=""), "version"),
    (lambda value: value["artifacts"]["model"].update(sha256=""), "hash"),
    (lambda value: value["flight_gate"].update(passed=False), "flight gate"),
    (lambda value: value["flow_gate"].update(passed=False), "flow gate"),
])
def test_refuses_incomplete_or_failed_release(tmp_path, mutation, match):
    from scripts.zggg_release import stage_candidate

    value = release_input(tmp_path)
    mutation(value)
    with pytest.raises(ValueError, match=match):
        stage_candidate(value, tmp_path / "release")


def test_refuses_non_zggg_future_weather_and_existing_output(tmp_path):
    from scripts.zggg_release import stage_candidate

    value = release_input(tmp_path)
    dataset = Path(value["dataset_manifest"])
    manifest = json.loads(dataset.read_text())
    manifest["airport"] = "ZBAA"
    dataset.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ZGGG"):
        stage_candidate(value, tmp_path / "release-a")

    manifest.update(airport="ZGGG", causality_audit={"future_weather_rows": 1})
    dataset.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="future weather"):
        stage_candidate(value, tmp_path / "release-b")

    manifest["causality_audit"] = {"future_weather_rows": 0}
    dataset.write_text(json.dumps(manifest))
    output = tmp_path / "release-c"
    output.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        stage_candidate(value, output)


def test_refuses_artifact_content_mismatch(tmp_path):
    from scripts.zggg_release import stage_candidate

    value = release_input(tmp_path)
    Path(value["artifacts"]["flow_model"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        stage_candidate(value, tmp_path / "release")


def independent_evidence():
    return {
        'previous_date_max': '2025-05-30',
        'blind_audit': {'eligible': True, 'date_min': '2025-06-01',
                        'date_max': '2025-06-05', 'source_sha256': 'a' * 64,
                        'blocks': []},
        'frozen_config_sha256': 'b' * 64,
        'frozen_at': '2025-05-31T10:00:00+08:00',
        'evaluated_at': '2025-06-06T10:00:00+08:00',
        'accuracy_gate': {'passed': True, 'report_sha256': 'c' * 64},
        'service_gate': {'passed': True, 'report_sha256': 'd' * 64},
    }


def independent_release_input(tmp_path):
    value = release_input(tmp_path)
    value.update(flight_mode='historical',
                 flight_model_version='historical-flight-v1',
                 flow_model_version='gradient_boosting',
                 feature_contract_version='zggg-airport-flow-v1',
                 prompt_version='historical-duration-v1')
    return value


def test_release_requires_service_and_accuracy_evidence(tmp_path):
    from scripts.zggg_release import stage_independent_candidate

    value = independent_release_input(tmp_path)
    evidence = independent_evidence()
    evidence['service_gate']['passed'] = False
    with pytest.raises(ValueError, match='service'):
        stage_independent_candidate(value, evidence, tmp_path / 'new-release')
    evidence = independent_evidence()
    evidence['accuracy_gate']['passed'] = False
    with pytest.raises(ValueError, match='accuracy'):
        stage_independent_candidate(value, evidence, tmp_path / 'new-release')
    evidence = independent_evidence()
    evidence['blind_audit']['date_min'] = '2025-05-29'
    with pytest.raises(ValueError, match='new dates'):
        stage_independent_candidate(value, evidence, tmp_path / 'new-release')


def test_failed_candidate_keeps_current_release(tmp_path):
    from scripts.zggg_release import stage_independent_candidate

    current = tmp_path / 'current'
    current.mkdir()
    (current / 'candidate.json').write_text('existing release')
    proposed = tmp_path / 'proposed'
    evidence = independent_evidence()
    evidence['blind_audit']['eligible'] = False
    with pytest.raises(ValueError, match='blind'):
        stage_independent_candidate(independent_release_input(tmp_path), evidence, proposed)
    assert (current / 'candidate.json').read_text() == 'existing release'
    assert not proposed.exists()


def test_independent_stage_preserves_evidence_in_immutable_candidate(tmp_path):
    from scripts.zggg_release import stage_independent_candidate

    output = tmp_path / 'new-release'
    value = independent_release_input(tmp_path)
    result = stage_independent_candidate(value, independent_evidence(), output)
    assert result['independent_evidence']['blind_audit']['date_min'] == '2025-06-01'
    assert json.loads((output / 'candidate.json').read_text()) == result
    with pytest.raises(ValueError, match='new directory'):
        stage_independent_candidate(value, independent_evidence(), output)


def test_independent_release_cannot_stage_legacy_v1(tmp_path):
    from scripts.zggg_release import stage_independent_candidate

    with pytest.raises(ValueError, match='serving flight mode'):
        stage_independent_candidate(release_input(tmp_path), independent_evidence(),
                                    tmp_path / 'new-release')
