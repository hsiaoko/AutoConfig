#!/usr/bin/env bash
#
# Train merged-tabular regressor with **no_graph** graph ablation:
#   Keep query program features (static_*, sym_*) and all conf_*;
#   graph block is only |V| and |E| (graph_num_vertices, graph_num_edges).
#   Drops diameter, skew, clustering, and all partition_* scalars.
#
# This wrapper always passes ``--graph-ve-only`` (see ``autoconfig train-merged --help``).
# **Required:** ``--model-basename <stem>`` so filenames match backend/experiment (e.g. ``nn_cost_no_graph``).
# Default before your args: ``--y-axis 1``. Override with ``-y`` / ``--y-axis``.
#
# Required / common passthrough (same as ``autoconfig train-merged``):
#   --data-dir, -d   Directory of merged feature YAMLs (*.yaml unless --pattern says otherwise)
#   --output, -o     Model output directory (*.pkl + *_meta.yaml)
#   --test-split     Float in [0,1]; 0 = train on all files (no internal test metrics); default in CLI 0.2
#
# Label / target (merged ``feature_vector`` slot 0..2):
#   -y N, --y-axis N   Allowed ONLY: 0 = price, 1 = time (default here), 2 = cost
#
# Model backend:
#   --model-kind NAME   Allowed (case-insensitive): bayesian | nn | mlp | rl
#   --model-options JSON  One JSON object; keys depend on backend (see TRAIN_TEST_MERGED.md).
#   --n-iter N          Positive int; used when model-kind=bayesian (default in CLI: 300).
#
# Boolean flags (omit = false, pass flag = true), same as train-merged:
#   --exclude-static  --exclude-symbolic  --full-graph-with-no-pf  --graph-e-only
#   (--graph-ve-only is always ON from this wrapper; --full-graph-with-no-pf overrides it.)
#
# Other passthrough:
#   --pattern '*.yaml'   --seed INT   --batch-size FLOAT
#
# Examples:
#   ./scripts/run_train_merged_no_graph.sh --model-basename nn_cost_no_graph \\
#     --data-dir exp/train/gpu/seen_tasks/ --output exp/models/gpu/seen_tasks/ \\
#     --test-split 0.1 -y 2 --model-kind nn \\
#     --model-options '{"hidden_layer_sizes":[256,128],"max_iter":800}'
#

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
  exec "$VENV_AC" train-merged --y-axis 1 --graph-ve-only "$@"
fi
if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m autoconfig.cli train-merged --y-axis 1 --graph-ve-only "$@"
fi

echo "Cannot find .venv. From repo root run:" >&2
echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
exit 1
