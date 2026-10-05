"""Stage an immutable ZGGG weather/flow release candidate after evidence checks."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re


_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_independent_evidence(evidence: dict) -> dict:
    """Require new dates and both gates before staging an independently tested release."""
    if not isinstance(evidence, dict):
        raise ValueError('independent evidence is required')
    blind = evidence.get('blind_audit')
    if not isinstance(blind, dict) or blind.get('eligible') is not True or blind.get('blocks') != []:
        raise ValueError('blind source audit did not pass')
    try:
        previous = date.fromisoformat(evidence['previous_date_max'])
        first = date.fromisoformat(blind['date_min'])
        last = date.fromisoformat(blind['date_max'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('independent new dates are required') from error
    if first <= previous or last < first:
        raise ValueError('independent new dates must follow the previous dataset')
    for value, label in ((blind.get('source_sha256'), 'blind source'),
                         (evidence.get('frozen_config_sha256'), 'frozen config')):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f'{label} hash is invalid')
    try:
        frozen = datetime.fromisoformat(evidence['frozen_at'].replace('Z', '+00:00'))
        evaluated = datetime.fromisoformat(evidence['evaluated_at'].replace('Z', '+00:00'))
        if (frozen.utcoffset() is None or evaluated.utcoffset() is None or
                frozen >= evaluated):
            raise ValueError('freeze must precede evaluation')
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise ValueError('frozen config must precede independent evaluation') from error
    for name in ('accuracy', 'service'):
        gate = evidence.get(f'{name}_gate')
        if (not isinstance(gate, dict) or gate.get('passed') is not True or
                not isinstance(gate.get('report_sha256'), str) or
                not _SHA256.fullmatch(gate['report_sha256'])):
            raise ValueError(f'{name} evidence gate did not pass')
    return json.loads(json.dumps(evidence, sort_keys=True))


def sha256_path(path: Path) -> str:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"artifact is missing: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _dataset_manifest(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid dataset manifest") from error
    if value.get("airport") != "ZGGG":
        raise ValueError("release dataset must be ZGGG")
    if not value.get("publishable") or not value.get("manifest_hash"):
        raise ValueError("dataset is not publishable or lacks a hash")
    if value.get("causality_audit", {}).get("future_weather_rows", 0) != 0:
        raise ValueError("future weather is forbidden")
    if value.get("schema_version") == "zggg-flight-weather-v1":
        from training.zggg_dataset import verify_zggg_dataset
        value = verify_zggg_dataset(Path(path).parent)
    return value


def validate_candidate(value: dict) -> dict:
    version = value.get("release_version")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("release version is invalid")
    dataset = _dataset_manifest(Path(value.get("dataset_manifest", "")))
    experimental = value.get('deployment_stage') == 'experimental'
    if value.get('deployment_stage', 'validated') not in ('validated', 'experimental'):
        raise ValueError('invalid deployment stage')
    for name in ("flight", "flow"):
        gate = value.get(f"{name}_gate")
        if not isinstance(gate, dict) or (gate.get("passed") is not True and
                not (experimental and name == 'flight' and gate.get('passed') is False)):
            raise ValueError(f"{name} gate did not pass")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifact hashes are required")
    normalized = {}
    for name, record in sorted(artifacts.items()):
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError(f"artifact hash is missing: {name}")
        path = Path(record.get("path", "")).resolve()
        actual = sha256_path(path)
        if actual != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
        normalized[name] = {"path": str(path), "sha256": actual}
    candidate = {
        "schema_version": "zggg-release-v1", "release_version": version,
        "airport": "ZGGG", "dataset_manifest_hash": dataset["manifest_hash"],
        "flight_gate_passed": value['flight_gate']['passed'], "flow_gate_passed": True,
        "artifacts": normalized,
    }
    if 'flight_mode' in value:
        mode = value['flight_mode']
        if mode not in ('historical', 'qwen'):
            raise ValueError('invalid flight mode')
        required = {'historical_model', 'flow_model', 'feature_contract'}
        if mode == 'qwen':
            required.add('model')
        if not required <= normalized.keys():
            raise ValueError('release artifacts are incomplete')
        candidate.update(schema_version='zggg-release-v2', flight_mode=mode,
                         source='BASELINE' if mode == 'historical' else 'LLM')
        for name in ('flight_model_version', 'flow_model_version', 'feature_contract_version',
                     'prompt_version'):
            candidate[name] = value.get(name)
        candidate['model_version'] = f"{candidate['flight_model_version']}+{candidate['flow_model_version']}"
        if 'fallback_identity' in value:
            candidate['fallback_identity'] = value['fallback_identity']
        from deploy.distributed.release import release_identity
        release_identity(candidate)
    if experimental:
        candidate['deployment_stage'] = 'experimental'
        from deploy.distributed.release import validate_experimental_release
        validate_experimental_release(candidate)
    if 'independent_evidence' in value:
        candidate['independent_evidence'] = validate_independent_evidence(
            value['independent_evidence'])
    return candidate


def stage_candidate(value: dict, output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("release output must be a new directory")
    candidate = validate_candidate(value)
    output.mkdir(parents=True)
    temporary = output / "candidate.json.tmp"
    temporary.write_text(json.dumps(candidate, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "candidate.json")
    return candidate


def stage_independent_candidate(value: dict, evidence: dict, output: Path) -> dict:
    """Stage the A03 candidate only after independent accuracy and service evidence."""
    checked = validate_independent_evidence(evidence)
    if value.get('deployment_stage') == 'experimental':
        raise ValueError('experimental candidate cannot claim independent approval')
    if value.get('flight_mode') not in ('historical', 'qwen'):
        raise ValueError('independent release requires a serving flight mode')
    return stage_candidate({**value, 'independent_evidence': checked}, output)


def stage_experimental_candidate(value: dict, output: Path) -> dict:
    """Explicitly stage Qwen-first operation while retaining failed accuracy evidence."""
    if 'independent_evidence' in value:
        raise ValueError('experimental candidate cannot claim independent approval')
    return stage_candidate({**value, 'deployment_stage': 'experimental'}, output)


def migrate_candidate(source: Path, output: Path, *, flight_mode: str,
                      flight_model_version: str, flow_model_version: str,
                      feature_contract_version: str, prompt_version: str) -> dict:
    """Create a new v2 file from an already approved v1 release; never edit v1."""
    old = json.loads(Path(source).read_text(encoding='utf-8'))
    if old.get('schema_version') != 'zggg-release-v1' or old.get('airport') != 'ZGGG' or old.get('flight_gate_passed') is not True or old.get('flow_gate_passed') is not True:
        raise ValueError('migration requires an approved ZGGG release v1')
    if flight_mode not in ('historical', 'qwen'):
        raise ValueError('migration requires explicit flight mode')
    artifacts = old.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('migration artifacts are missing')
    required = {'historical_model', 'flow_model', 'feature_contract'}
    if flight_mode == 'qwen':
        required.add('model')
        if 'adapter' in artifacts:
            required.add('adapter')
    if not required <= artifacts.keys():
        raise ValueError('migration artifacts are incomplete')
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not record.get('sha256') or sha256_path(Path(record.get('path', ''))) != record['sha256']:
            raise ValueError(f'artifact hash mismatch: {name}')
    candidate = {**old, 'schema_version': 'zggg-release-v2', 'flight_mode': flight_mode,
                 'source': 'BASELINE' if flight_mode == 'historical' else 'LLM',
                 'artifacts': {name: artifacts[name] for name in sorted(required)},
                 'flight_model_version': flight_model_version,
                 'flow_model_version': flow_model_version,
                 'feature_contract_version': feature_contract_version,
                 'prompt_version': prompt_version,
                 'model_version': f'{flight_model_version}+{flow_model_version}'}
    from deploy.distributed.release import release_identity
    release_identity(candidate)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('release output must be a new directory')
    output.mkdir(parents=True)
    temporary = output / 'candidate.json.tmp'
    temporary.write_text(json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(output / 'candidate.json')
    return candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage a verified ZGGG release candidate")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--migrate-from", type=Path)
    parser.add_argument('--independent-evidence', type=Path)
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--flight-mode', choices=('historical', 'qwen'))
    for field in ('flight-model-version', 'flow-model-version', 'feature-contract-version', 'prompt-version'):
        parser.add_argument('--' + field)
    args = parser.parse_args(argv)
    try:
        if args.experimental and (args.migrate_from or args.independent_evidence):
            raise ValueError('experimental staging cannot migrate or claim independent approval')
        if args.migrate_from:
            value = migrate_candidate(args.migrate_from, args.output, flight_mode=args.flight_mode,
                                      flight_model_version=args.flight_model_version,
                                      flow_model_version=args.flow_model_version,
                                      feature_contract_version=args.feature_contract_version,
                                      prompt_version=args.prompt_version)
        else:
            proposed = json.loads(args.input.read_text(encoding='utf-8'))
            value = (stage_independent_candidate(
                proposed, json.loads(args.independent_evidence.read_text(encoding='utf-8')),
                args.output) if args.independent_evidence else
                (stage_experimental_candidate(proposed, args.output) if args.experimental
                 else stage_candidate(proposed, args.output)))
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"release staging failed: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
