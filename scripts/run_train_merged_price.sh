#!/usr/bin/env bash
# Train with Y = **price** (``--target price``). Merged files are 53-D: price, time, cost, …
# Forwards all args to: autoconfig train-merged --target price
#
# Usage:
#   ./scripts/run_train_merged_price.sh
#   ./scripts/run_train_merged_price.sh --data-dir out/train/price --output out/models/price
#   ./scripts/run_train_merged_price.sh --data-dir out/train/price --output out/models/price --test-split 0
#   ./scripts/run_train_merged_price.sh --n-iter 500

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  set -- --data-dir out/train/price --output out/models/price --test-split 0.2 --model-basename bayesian_price_merged
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" train-merged --target price "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --target price "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
