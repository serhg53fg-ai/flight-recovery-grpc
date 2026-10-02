from configparser import ConfigParser
from pathlib import Path
import pytest

from deploy.autodl.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_cluster_example_is_valid():
    manifest = load_manifest(ROOT / "deploy/autodl/examples/cluster.example.json")
    assert len(manifest.workers) == 2


@pytest.mark.parametrize('project_root', [ROOT, Path('/srv/flight-example/project')])
def test_supervisor_examples_use_repository_entrypoints_and_safe_restart(project_root):
    expected = {
        "control.conf.example": "-m deploy.autodl.control",
        "worker.conf.example": "-m deploy.autodl.worker",
    }
    for filename, entrypoint in expected.items():
        path = ROOT / "deploy/autodl/supervisor" / filename
        parser = ConfigParser(defaults={'ENV_FLIGHT_PROJECT_ROOT': str(project_root)})
        parser.read(path, encoding="utf-8")
        section = next(name for name in parser.sections() if name.startswith("program:"))
        assert entrypoint in parser[section]["command"]
        assert parser[section]["directory"] == str(project_root)
        assert parser[section]['command'].startswith(str(project_root / '.venv/bin/python'))
        assert parser[section].getboolean("autostart") is True
        assert parser[section]["autorestart"] == "unexpected"
        assert parser[section].getint("startretries") >= 3


def test_deploy_examples_and_docs_contain_no_credential_or_host_key_bypass():
    paths = list((ROOT / "deploy/autodl").rglob("*")) + list((ROOT / "docs/autodl").glob("*.md"))
    forbidden = ("strict" + "hostkeychecking=no", "begin " + "private key", "sshpass", "password=" )
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower().replace(" ", "")
        assert not any(term.replace(" ", "") in text for term in forbidden), path
