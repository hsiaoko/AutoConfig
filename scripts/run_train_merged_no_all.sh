#!/usr/bin/env bash
# no_all: X = conf_* (all 13) + graph_num_edges (|E|) only. No |V|, no SPF/SGF, no other graph.
# Flags: --exclude-static --exclude-symbolic --graph-ve-only --graph-e-only
# Same as run_train_merged_no_pf.sh except model basename (you choose via --model-basename) and --graph-e-only.
#   no_pf = conf + |V| + |E|; no_all = conf + |E| only.
# Legacy: full graph with program features removed -> add --full-graph-with-no-pf; drop
#   --graph-e-only to get |V|+|E|+conf.
# **Required:** ``--model-basename <stem>``.
# Default Y = **time** (``--y-axis 1``).
# Usage:
#   ./scripts/run_train_merged_no_all.sh --model-basename bayesian_cost_no_all \\
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
  exec "$VENV_AC" train-merged --y-axis 1 --exclude-static --exclude-symbolic --graph-ve-only \
    --graph-e-only "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 1 --exclude-static --exclude-symbolic --graph-ve-only \
    --graph-e-only "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
