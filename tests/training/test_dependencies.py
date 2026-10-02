from pathlib import Path


def test_trl_transformers_pins_are_compatible():
    requirements = (Path(__file__).parents[2] / "requirements-worker.txt").read_text()
    assert "trl==0.21.0" in requirements
    assert "transformers==4.55.4" in requirements
