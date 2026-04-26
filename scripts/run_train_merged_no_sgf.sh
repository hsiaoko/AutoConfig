#!/usr/bin/env bash
# Same as run_train_merged.sh but drops graph-parameterized symbolic features (sym_*), i.e. SGF.
# Usage:
#   ./scripts/run_train_merged_no_sgf.sh
#   ./scripts/run_train_merged_no_sgf.sh --data-dir out/train --output out/models --test-split 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  set -- --data-dir out/train --output out/models --test-split 0
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" train-merged --exclude-symbolic --model-basename bayesian_cost_merged_no_sgf "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --exclude-symbolic --model-basename bayesian_cost_merged_no_sgf "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
