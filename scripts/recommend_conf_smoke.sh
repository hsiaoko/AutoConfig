#!/usr/bin/env bash
# End-to-end smoke: synthetic merged training dir + tiny model + :func:`recommend_top_k`.
# Safe to run without real graph/query YAMLs. Requires editable install: pip install -e .
# 若需试导出，可在本脚本后加参数:  见 ``run_recommend_conf.sh -o <目录>``（会写出 recommend_rank01.yaml 等）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Need ${VENV_PY}; run: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

exec "$VENV_PY" "${ROOT}/scripts/recommend_conf_smoke.py" "$@"
