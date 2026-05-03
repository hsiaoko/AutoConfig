#!/usr/bin/env bash
# Run GPU-related experiment configs from exp/conf/gpu against GridGraph (wcc, bfs, pagerank)
# and MatrixGraph (subiso_exec). Outputs under exp/gpu/:
#   gridgraph/<graph>_<task>/conf_NN.output
#   subiso/*.log, subiso/conf_*/output/, subiso_autoconfig_summary.csv
#
# Env:
#   AC_CONF          AutoConfig conf dir (default: repo/exp/conf/gpu)
#   GRIDGRAPH_ROOT   GridGraph checkout (default: sibling GridGraph/)
#   MG_ROOT          MatrixGraph checkout (default: sibling MatrixGraph/)
#   GRIDGRAPH_WORKSPACE  Graph datasets root (default: /ssd_data/.../GridGraph_workspace)
#   GRAPHS           Space-separated dataset names under workspace (default: four graphs)
#   MIN_CONF MAX_CONF   conf id range inclusive (default 1 50)
#   SKIP_GRIDGRAPH   if set (non-empty), skip wcc/bfs/pr
#   SKIP_SUBISO      if set (non-empty), skip MatrixGraph subiso_exec
#   MG_PATTERN MG_DATA  CSR dirs for subiso_exec (-p / -g); see MatrixGraph script defaults
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]] && [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
AC_CONF="${AC_CONF:-$ROOT/exp/conf/gpu}"
GRIDGRAPH_ROOT="${GRIDGRAPH_ROOT:-/home/zhuxk/project/graph/GridGraph}"
MG_ROOT="${MG_ROOT:-/home/zhuxk/project/graph/MatrixGraph}"
GRIDGRAPH_WORKSPACE="${GRIDGRAPH_WORKSPACE:-/ssd_data/zhuxk/workspace/GridGraph_workspace}"
GRAPHS="${GRAPHS:-friendster livejournal patents web-sk}"
MIN_CONF="${MIN_CONF:-1}"
MAX_CONF="${MAX_CONF:-50}"
OUT_ROOT="${OUT_ROOT:-$ROOT/exp/gpu}"
RESTORE_PARALLELISM="${RESTORE_PARALLELISM:-44}"

mkdir -p "$OUT_ROOT/gridgraph"
mkdir -p "$OUT_ROOT/subiso"

parse_conf_cores_mem() {
  "${PYTHON}" - "$1" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    doc = yaml.safe_load(f)
for item in doc.get("catalog") or []:
    if isinstance(item, dict) and "cpu_cores" in item:
        c = int(float(item["cpu_cores"]))
        m = int(float(item["memory_gb"]))
        print(c, m)
        sys.exit(0)
sys.exit(1)
PY
}

set_parallelism_gg() {
  local p="$1"
  sed -i "s/^[[:space:]]*parallelism = [0-9][0-9]*;/\t\tparallelism = ${p};/" "${GRIDGRAPH_ROOT}/core/graph.hpp"
}

run_gridgraph_one() {
  local nn="$1" graph="$2" task="$3" mem="$4"
  local gpath="${GRIDGRAPH_WORKSPACE%/}/${graph}/"
  local sub="${OUT_ROOT}/gridgraph/${graph}_${task}"
  mkdir -p "$sub"
  local out="${sub}/conf_${nn}.output"
  cd "$GRIDGRAPH_ROOT"
  case "$task" in
    wcc)
      stdbuf -oL -eL /usr/bin/time -p ./bin/wcc "$gpath" "$mem" >"$out" 2>&1
      ;;
    bfs)
      stdbuf -oL -eL /usr/bin/time -p ./bin/bfs "$gpath" 0 "$mem" >"$out" 2>&1
      ;;
    pr)
      stdbuf -oL -eL /usr/bin/time -p ./bin/pagerank "$gpath" 5 "$mem" >"$out" 2>&1
      ;;
    *)
      echo "bad task: $task" >&2
      return 1
      ;;
  esac
}

echo "=== $(date -Is) run_exp_conf_gpu_benchmarks START ==="
echo "AC_CONF=$AC_CONF OUT_ROOT=$OUT_ROOT conf ${MIN_CONF}-${MAX_CONF}"

if [[ -z "${SKIP_GRIDGRAPH:-}" ]]; then
  for c in $(seq "$MIN_CONF" "$MAX_CONF"); do
    printf -v nn "%02d" "$c"
    yfile="${AC_CONF}/conf_${nn}.yaml"
    if [[ ! -f "$yfile" ]]; then
      echo "missing $yfile" >&2
      exit 1
    fi
    read -r CORES MEM <<< "$(parse_conf_cores_mem "$yfile")"
    echo "--- $(date -Is) GridGraph conf_${nn} cores=${CORES} mem_gb=${MEM} ---"
    set_parallelism_gg "$CORES"
    make -j"$(nproc)" -C "$GRIDGRAPH_ROOT" bin/wcc bin/bfs bin/pagerank
    for g in $GRAPHS; do
      for task in wcc bfs pr; do
        echo ">>> $(date -Is) ${task} ${g} conf_${nn}"
        run_gridgraph_one "$nn" "$g" "$task" "$MEM" || {
          echo "FAILED conf_${nn} ${g} ${task}" >&2
          exit 1
        }
      done
    done
  done
  set_parallelism_gg "$RESTORE_PARALLELISM"
  make -j"$(nproc)" -C "$GRIDGRAPH_ROOT" bin/wcc bin/bfs bin/pagerank >/dev/null 2>&1 || true
  echo "Restored graph.hpp parallelism to ${RESTORE_PARALLELISM}"
else
  echo "SKIP_GRIDGRAPH set — not running wcc/bfs/pr"
fi

if [[ -z "${SKIP_SUBISO:-}" ]]; then
  echo "--- $(date -Is) MatrixGraph subiso_exec ---"
  PY=(
    "${PYTHON}"
    "${MG_ROOT}/scripts/run_autoconfig_subiso_exp.py"
    "--conf-dir"
    "$AC_CONF"
    "--exp-root"
    "$ROOT/exp"
  )
  if [[ -n "${MG_PATTERN:-}" ]]; then
    PY+=("-p" "$MG_PATTERN")
  fi
  if [[ -n "${MG_DATA:-}" ]]; then
    PY+=("-g" "$MG_DATA")
  fi
  "${PY[@]}"
else
  echo "SKIP_SUBISO set — not running subiso_exec"
fi

if [[ -f "${ROOT}/exp/subiso_autoconfig_summary.csv" ]]; then
  cp -a "${ROOT}/exp/subiso_autoconfig_summary.csv" "${OUT_ROOT}/"
fi

echo "=== $(date -Is) run_exp_conf_gpu_benchmarks DONE ==="
