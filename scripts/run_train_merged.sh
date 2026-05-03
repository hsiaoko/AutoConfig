#!/usr/bin/env bash
# Run: autoconfig train-merged using the project venv (no hardcoded project path).
# Merged 53-d layout: X = features 4..53 (static..conf).
# Default Y is **time** (`y=1`) via ``--y-axis 1`` below; override with ``-y 0|1|2`` (see docs/TRAIN_TEST_MERGED.md).
# **Required:** ``--model-basename <stem>`` — output ``<output>/<stem>.pkl`` + ``<stem>_meta.yaml`` (name should reflect backend).
# **Model backend** (default: ``--model-kind bayesian``). Pass-through args; all flags go to ``autoconfig train-merged``.
# Full table + RL notes: ``docs/TRAIN_TEST_MERGED.md`` (anchor ``#model-kind``).
# ``nn`` / ``mlp`` = 神经网络；``eval-merged`` 只读 ``*.pkl`` 与旁路 ``*_meta.yaml``，无需再指定 kind。
#
# Usage:
#   ./scripts/run_train_merged.sh --model-basename bayesian_cost_full --data-dir out/train --output out/models
#   ./scripts/run_train_merged.sh --model-basename nn_cost_run1 --data-dir out/train --output out/models --test-split 0 --model-kind nn
#   ./scripts/run_train_merged.sh --model-basename nn_cost_run1 --model-kind nn --model-options '{"max_iter":500,"early_stopping":false}' ...

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
  exec "$VENV_AC" train-merged --y-axis 1 "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 1 "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
