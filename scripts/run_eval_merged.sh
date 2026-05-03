#!/usr/bin/env bash
# Evaluate: model (.pkl) + test folder of merged feature YAMLs → metrics to stdout.
# Backend (贝叶斯 / ``nn`` / …) 由 ``*.pkl`` 同目录的 ``*_meta.yaml`` 中 ``model_kind`` 决定，无需传 ``--model-kind``。
#
# **Y (label):** Omit ``-y`` / ``--y-axis`` → ``eval-merged`` uses ``y_axis`` (or legacy ``target`` string)
# from ``<model_stem>_meta.yaml``; if missing, defaults to **y=1** (time). Override: ``-y 0|1|2``.
#
# Usage:
#   ./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test
#   ./scripts/run_eval_merged.sh -m model.pkl -d out/test --y-axis 1   # force label = time
#   ./scripts/run_eval_merged.sh   # no args → default paths below (Y from meta is typical)

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
