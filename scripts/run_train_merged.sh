#!/usr/bin/env bash
# Run: autoconfig train-merged using the project venv (no hardcoded project path).
# Merged 53-d layout: X = features 4..53 (static..conf); Y = --target or --y-axis (0=price,1=time,2=cost).
# Usage:
#   ./scripts/run_train_merged.sh
#   ./scripts/run_train_merged.sh --data-dir out/train --output out/models
#   ./scripts/run_train_merged.sh --y-axis 1 --test-split 0
#   ./scripts/run_train_merged.sh --test-split 0 --pattern '*_gridgraph_wcc.yaml'
#   ./scripts/run_train_merged.sh --n-iter 500

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  set -- --data-dir out/train --output out/models
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" train-merged "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
