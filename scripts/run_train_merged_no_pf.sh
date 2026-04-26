#!/usr/bin/env bash
# Same as run_train_merged.sh but drops all program features: static (static_*) and symbolic (sym_*).
# Only graph + config columns remain. Output model basename: bayesian_cost_merged_no_pf
# Default Y = **time** (``--y-axis 1``); override e.g. ``--y-axis 0`` for price.
# Usage:
#   ./scripts/run_train_merged_no_pf.sh
#   ./scripts/run_train_merged_no_pf.sh --data-dir out/train --output out/models --test-split 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  set -- --data-dir out/train --output out/models --test-split 0
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" train-merged --y-axis 1 --exclude-static --exclude-symbolic --model-basename bayesian_cost_merged_no_pf "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 1 --exclude-static --exclude-symbolic --model-basename bayesian_cost_merged_no_pf "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
