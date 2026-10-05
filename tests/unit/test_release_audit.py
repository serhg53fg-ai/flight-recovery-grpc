from pathlib import Path


def test_audit_rejects_private_key_and_reports_no_secret_text(tmp_path):
    from scripts.release_audit import audit_files

    secret = '-----BEGIN OPENSSH ' + 'PRIVATE KEY-----\ndo-not-print-this\n'
    path = tmp_path / 'id_key'; path.write_text(secret)
    report = audit_files(tmp_path, [path])

    assert report['status'] == 'FAIL'
    assert report['findings'] == [{'path': 'id_key', 'rule': 'private_key_material'}]
    assert 'do-not-print-this' not in str(report)


def test_audit_rejects_runtime_models_and_embedded_credentials(tmp_path):
    from scripts.release_audit import audit_files

    runtime = tmp_path / 'runtime' / 'config.json'; runtime.parent.mkdir(); runtime.write_text('{}')
    credential_url = 'https://alice:' + 'real-pass@example.test/api'
    source = tmp_path / 'deploy.py'; source.write_text(f'URL = "{credential_url}"')
    report = audit_files(tmp_path, [runtime, source])

    assert {'path': 'runtime/config.json', 'rule': 'runtime_or_model_artifact'} in report['findings']
    assert {'path': 'deploy.py', 'rule': 'embedded_url_credentials'} in report['findings']


def test_audit_accepts_examples_and_source_files(tmp_path):
    from scripts.release_audit import audit_files, write_report

    source = tmp_path / 'app.py'; source.write_text("password = os.environ.get('PASSWORD', '')")
    example = tmp_path / 'cluster.example.json'; example.write_text('{"private_key":"/path/to/key"}')
    report = audit_files(tmp_path, [source, example])
    assert report['status'] == 'PASS'
    output = tmp_path / 'audit.json'; write_report(output, report)
    assert output.is_file()


def test_candidate_files_accepts_valid_external_delivery_manifest(tmp_path):
    import json
    import pytest
    from scripts.release_audit import candidate_files

    manifest = tmp_path / 'paths.json'; manifest.write_text(json.dumps(['app.py', 'docs/run.md']))
    assert candidate_files(tmp_path, manifest) == [tmp_path / 'app.py', tmp_path / 'docs/run.md']
    manifest.write_text(json.dumps(['../outside']))
    with pytest.raises(ValueError, match='invalid'):
        candidate_files(tmp_path, manifest)
