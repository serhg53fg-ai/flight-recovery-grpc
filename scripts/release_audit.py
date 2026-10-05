"""Audit candidate delivery files without echoing suspected secret contents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


SECRET_RULES = (
    ('private_key_material', re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----')),
    ('embedded_url_credentials', re.compile(rb'https?://[^/\s:@]+:[^@\s/]+@')),
    ('github_token', re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{20,}\b')),
    ('openai_key', re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}\b')),
    ('aws_access_key', re.compile(rb'\bAKIA[0-9A-Z]{16}\b')),
    ('sshpass_password', re.compile(rb'\bsshpass\s+-p\b')),
)
LITERAL_CREDENTIAL = re.compile(
    rb'(?i)\b(?:password|passwd|secret|token)\b\s*[:=]\s*["\']([^"\']{4,})["\']')
BLOCKED_PARTS = frozenset(('runtime', 'models', 'model-assets', '.venv', 'build'))
BLOCKED_SUFFIXES = frozenset(('.pem', '.p12', '.pfx'))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def audit_files(root: Path, paths) -> dict:
    root = Path(root).resolve()
    findings = set()
    checked = 0
    for supplied in paths:
        path = Path(supplied)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            continue
        relative = _relative(root, path)
        checked += 1
        parts = Path(relative).parts
        if any(part in BLOCKED_PARTS for part in parts) or path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.add((relative, 'runtime_or_model_artifact'))
        if path.stat().st_size > 10 * 1024 * 1024:
            findings.add((relative, 'oversized_delivery_file'))
        data = path.read_bytes()
        for rule, pattern in SECRET_RULES:
            if pattern.search(data):
                findings.add((relative, rule))
        is_fixture = ('tests' in parts or 'docs' in parts or '.example.' in path.name)
        if not is_fixture:
            for match in LITERAL_CREDENTIAL.finditer(data):
                value = match.group(1).lower()
                if not any(marker in value for marker in
                           (b'example', b'dummy', b'changeme', b'placeholder', b'/path/to/')):
                    findings.add((relative, 'literal_credential'))
                    break
    rows = [{'path': path, 'rule': rule} for path, rule in sorted(findings)]
    return {'status': 'FAIL' if rows else 'PASS', 'checked_files': checked,
            'findings': rows}


def candidate_files(root: Path, paths_file: Path | None = None) -> list[Path]:
    if paths_file is not None:
        values = json.loads(Path(paths_file).read_text())
        if not isinstance(values, list) or not all(isinstance(value, str) and value and
                                                   not Path(value).is_absolute() and '..' not in Path(value).parts
                                                   for value in values):
            raise ValueError('invalid delivery paths file')
        return [root / value for value in values]
    result = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=root, capture_output=True, check=True)
    return [root / value.decode('utf-8', 'surrogateescape')
            for value in result.stdout.split(b'\0') if value]


def write_report(output: Path, report: dict) -> None:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('audit report already exists')
    if not output.parent.is_dir():
        raise ValueError('audit report parent is missing')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Secret-safe delivery audit')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paths-file', type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        paths = candidate_files(root, args.paths_file)
        report = audit_files(root, paths)
        report['tree_digest'] = hashlib.sha256(
            '\n'.join(sorted(path.relative_to(root).as_posix()
                             for path in paths)).encode()).hexdigest()
        write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report['status'] == 'PASS' else 1
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'release audit failed: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
