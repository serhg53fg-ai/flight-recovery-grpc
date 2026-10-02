"""Create an untracked spreadsheet from public synthetic JSON inputs."""
import json
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    target = root / 'runtime/demo/synthetic-flights.xlsx'
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(json.loads((root / 'examples/synthetic-batch.json').read_text())).to_excel(target, index=False)
    print(target)
