#!/usr/bin/env bash
# Evaluate: model (.pkl) + test folder of merged feature YAMLs → metrics to stdout.
# Usage:
#   ./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test
#   ./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/train -o out/eval_result.yaml
#   ./scripts/run_eval_merged.sh   # uses defaults: model + data-dir out/test (edit if you need)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  set -- -m out/models/bayesian_cost_merged.pkl -d out/test
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" eval-merged "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli eval-merged "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
