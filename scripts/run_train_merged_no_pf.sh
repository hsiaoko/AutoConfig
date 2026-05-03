#!/usr/bin/env bash
# Same as run_train_merged.sh but drops all program features: static (static_*) and symbolic (sym_*).
# Graph columns: only |V| and |E| (graph_num_vertices, graph_num_edges) plus all conf_*.
# For legacy no_pf (all graph_*/partition_*), add: --full-graph-with-no-pf
# **Required:** ``--model-basename <stem>``.
# Default Y = **time** (``--y-axis 1``); override e.g. ``--y-axis 0`` for price.
# Usage:
#   ./scripts/run_train_merged_no_pf.sh --model-basename bayesian_cost_no_pf \\
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
  exec "$VENV_AC" train-merged --y-axis 1 --exclude-static --exclude-symbolic "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 1 --exclude-static --exclude-symbolic "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
