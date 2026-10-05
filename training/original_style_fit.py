"""Fit one original-style model arm in an environment with Torch and sklearn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .original_style_model import OriginalStyleEnsemble


def _read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.metadata.exists():
        raise ValueError("model output and metadata must be new files")
    train, test = _read(args.train), _read(args.test)
    begin = perf_counter()
    model = OriginalStyleEnsemble().fit(train)
    fitted = perf_counter()
    predictions = model.predict(test)
    predicted = perf_counter()
    import sklearn
    import torch
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    args.metadata.write_text(json.dumps({"train_rows": len(train), "test_rows": len(test),
                                         "fit_seconds": fitted - begin,
                                         "predict_seconds": predicted - fitted,
                                         "sklearn_version": sklearn.__version__,
                                         "torch_version": torch.__version__}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "train_rows": len(train), "test_rows": len(test)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
