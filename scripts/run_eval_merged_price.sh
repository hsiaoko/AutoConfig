#!/usr/bin/env bash
# Evaluate a model trained with Y = **price** (``--y-axis 0``).
# Invokes: ``eval-merged --y-axis 0`` (a later ``-y`` on the command line overrides).
#
# Usage:
#   ./scripts/run_eval_merged_price.sh -m out/models/price/bayesian_price_merged.pkl -d out/test/price
#   ./scripts/run_eval_merged_price.sh -m ... -d ... -o out/eval_price_report.yaml
#
# With no args, defaults to a price layout under out/models/price and out/test/price (edit as needed).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  set -- -m out/models/price/bayesian_price_merged.pkl -d out/test/price
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" eval-merged --y-axis 0 "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli eval-merged --y-axis 0 "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
