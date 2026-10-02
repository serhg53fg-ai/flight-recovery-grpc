#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-$project_root/.venv/bin/python}"
mkdir -p "$project_root/generated"
"$python_bin" -m grpc_tools.protoc -I "$project_root/proto" \
  --python_out="$project_root/generated" --grpc_python_out="$project_root/generated" \
  "$project_root/proto/flight/v1/prediction.proto"
