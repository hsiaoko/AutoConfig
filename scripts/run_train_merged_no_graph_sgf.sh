#!/usr/bin/env bash
# no_graph + no SGF: graph block is only |V| and |E| (graph_num_vertices, graph_num_edges);
# drop all sym_* (graph-parameterized symbolic / SGF). Keep static_*, all conf_*.
# Drops diameter, skew, partition_*, and other graph_* except the two size scalars.
# **Required:** ``--model-basename <stem>``.
# Default Y = **time** (``--y-axis 1``); override e.g. ``--y-axis 0`` for price.
# Usage:
#   ./scripts/run_train_merged_no_graph_sgf.sh --model-basename nn_cost_no_graph_sgf \\
#     --data-dir out/train --output out/models --test-split 0

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
  exec "$VENV_AC" train-merged --y-axis 1 --graph-ve-only --exclude-symbolic "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 1 --graph-ve-only --exclude-symbolic "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
