#!/usr/bin/env bash
# Train with Y = **price** (`y=0`, i.e. ``--y-axis 0``). Merged files are 53-D.
# **Required:** ``--model-basename <stem>`` (pick a name that reflects backend + experiment).
# Forwards all args: ``autoconfig train-merged --y-axis 0 …`` (add ``--model-kind nn`` 等).
#
# Usage:
#   ./scripts/run_train_merged_price.sh --model-basename bayesian_price_merged \\
#     --data-dir out/train/price --output out/models/price --test-split 0.2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_require_model_basename.inc.sh"
_require_model_basename_or_exit "$@"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" train-merged --y-axis 0 "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 0 "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
