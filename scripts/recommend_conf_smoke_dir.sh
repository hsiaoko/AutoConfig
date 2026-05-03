#!/usr/bin/env bash
# Smoke: conf 候选为**一整个目录**（与 ``data/conf/`` 每文件一配置相同），并打印 ``source_path``。
# 与 ``recommend_conf_smoke.sh`` 对照：后者用内联多 conf，本脚本专门测目录模式。
# 导出示例:  ``./scripts/run_recommend_conf.sh ... -c data/conf/gpu/ -o out/picked`` → recommend_rank01.yaml …

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Need ${VENV_PY}; run: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

exec "$VENV_PY" "${ROOT}/scripts/recommend_conf_smoke_dir.py" "$@"
