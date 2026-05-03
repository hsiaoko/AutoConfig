#!/usr/bin/env bash
# 使用 **``-c`` 为「配置目录」** 跑 ``recommend-conf``：目录下每个 ``*.yaml`` 一个 ``configuration``（同 ``data/conf/gpu/`` 那种）。
# 不解析子目录，仅顶层 ``*.yaml``；排序见代码 ``_load_config_dir``。
#
# 用法 1（显式全参数，与 ``run_recommend_conf.sh`` 相同，只是明确 ``-c`` 为目录）:
#   ./scripts/run_recommend_conf_dir.sh -m out/models/xxx.pkl -q out/query_features/kernel_bfs.yaml \\
#     -g /path/to/graph_features.yaml -c data/conf/gpu/ -k 5 \\
#     -o out/recommended_confs
#
# 用法 2（用环境变量省写字；未设置的项仍须通过环境或命令行提供）:
#   export MERGED_MODEL=out/models/bayesian_cost_merged.pkl
#   export QUERY_YAML=out/query_features/kernel_bfs.yaml
#   export GRAPH_YAML=/path/to/graph_features.yaml
#   export CONF_DIR=data/conf/gpu
#   export RECOMMEND_EXPORT_DIR=out/recommended_confs   # 可选：自动加 ``-o``，每配置一个 YAML
#   ./scripts/run_recommend_conf_dir.sh
#
# 有任意 CLI 参数时，整段透传给 ``run_recommend_conf.sh``。
# 若设置了 ``RECOMMEND_EXPORT_DIR`` 且命令行里**没有** ``-o``/``--output``，则自动追 ``-o``。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RRC="${ROOT}/scripts/run_recommend_conf.sh"

_has_output_flag() {
  local a
  for a in "$@"; do
    if [[ "$a" == "-o" || "$a" == "--output" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ $# -gt 0 ]]; then
  if [[ -n "${RECOMMEND_EXPORT_DIR:-}" ]] && ! _has_output_flag "$@"; then
    mkdir -p "${RECOMMEND_EXPORT_DIR}"
    exec "$RRC" "$@" -o "${RECOMMEND_EXPORT_DIR}"
  fi
  exec "$RRC" "$@"
fi

: "${MERGED_MODEL:=out/models/bayesian_cost_merged.pkl}"
: "${QUERY_YAML:=out/query_features/kernel_bfs.yaml}"
: "${GRAPH_YAML:=}"
: "${CONF_DIR:=data/conf/gpu}"
: "${K:=5}"
: "${RECOMMEND_EXPORT_DIR:=}"

if [[ -z "${GRAPH_YAML}" ]]; then
  echo "当无参数时，请设置 GRAPH_YAML=你的 graph_features.yaml 路径" >&2
  echo "或显式传参:  $0 -m M -q Q -g G -c <conf目录或文件> -k N" >&2
  exit 1
fi

if [[ ! -d "${CONF_DIR}" ]]; then
  echo "CONF_DIR 不是目录: ${CONF_DIR}" >&2
  echo "可改为: export CONF_DIR=data/conf/gpu  或 直接传: $0 -m ... -c /path/to/conf_dir" >&2
  exit 1
fi

EXTRA=()
if [[ -n "${RECOMMEND_EXPORT_DIR}" ]]; then
  mkdir -p "${RECOMMEND_EXPORT_DIR}"
  EXTRA+=(-o "${RECOMMEND_EXPORT_DIR}")
fi

exec "$RRC" \
  -m "${MERGED_MODEL}" \
  -q "${QUERY_YAML}" \
  -g "${GRAPH_YAML}" \
  -c "${CONF_DIR}/" \
  -k "${K}" \
  "${EXTRA[@]}"
