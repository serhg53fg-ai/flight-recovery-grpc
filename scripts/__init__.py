"""Make generated RPC modules available to project operational commands."""
from pathlib import Path
import sys

_generated = str(Path(__file__).resolve().parents[1] / 'generated')
if _generated not in sys.path:
    sys.path.insert(0, _generated)
