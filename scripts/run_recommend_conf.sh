#!/usr/bin/env bash
# Call ``autoconfig recommend-conf``: rank config candidates with a train-merged model.
#
# Required:  -m model.pkl, -q query.yaml, -g graph.yaml, -c <config>  其中
#   ``-c`` 可以是**单个** YAML，或**文件夹**（每个 ``*.yaml`` 一个 ``configuration``，与 ``data/conf/`` 里每文件一配置相同）。
# Optional:  -k N  --maximize  --batch-size N
#            -o <目录>  每个推荐配置写一个 YAML（recommend_rank01.yaml …），与 ``data/conf`` 单文件格式一致
#            --export-prefix P  --full-report  # 目录下另存完整 to_dict
#            --refine  --refine-max-iterations N  --refine-seed S
#   ``--refine``: top-k 后对 cpu/memory/GPU 做随机扰动并反复预测，直到无更优（见 :mod:`autoconfig.conf_recommend.refine_search`）
#
# Examples:
#   ./scripts/run_recommend_conf.sh -m M -q Q -g G -c data/conf/gpu/ -k 5 \\
#     -o out/recommended_confs
#   # ``-o`` 为**目录**时：写出 recommend_rank01.yaml …（每文件一个 configuration，可 merge）
#   # 路径以 .yaml/.yml 结尾则只写**一个**汇总文件（to_dict）
#   ./scripts/run_recommend_conf.sh -m M -q Q -g G -c /path/to/candidates.yaml \\
#     -k 3 --refine -o out/rec_picked --export-prefix best
#
# One-shot self-test (synthetic data + train + recommend, no hand-made YAMLs):
#   ./scripts/recommend_conf_smoke.sh
#
# Extra args are forwarded:  ./scripts/run_recommend_conf.sh -m M -q Q -g G -c C -k 3 "$@"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_AC="${ROOT}/.venv/bin/autoconfig"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 -m <model.pkl> -q <query.yaml> -g <graph.yaml> -c <config> [recommend-conf options...]" >&2
  echo "  -c: 单个 YAML（多行 configurations）**或** 目录（每文件一个 conf，如 data/conf/gpu/）" >&2
  echo "  -o <DIR>: 导出每个推荐为独立 YAML（默认名 recommend_rankNN.yaml）；见 autoconfig recommend-conf -h" >&2
  echo "便捷:  ${ROOT}/scripts/run_recommend_conf_dir.sh  （支持环境变量 RECOMMEND_EXPORT_DIR）" >&2
  echo "自测:  ${ROOT}/scripts/recommend_conf_smoke.sh  |  ${ROOT}/scripts/recommend_conf_smoke_dir.sh" >&2
  exit 1
fi

if [[ -x "$VENV_AC" ]]; then
  exec "$VENV_AC" recommend-conf "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli recommend-conf "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
