#!/usr/bin/env bash
# shellcheck shell=bash
# Source-only: merged train wrappers must receive --model-basename so output filenames match intent.

_require_model_basename_in_args() {
  local i a v j
  i=1
  while [[ $i -le $# ]]; do
    eval "a=\${$i}"
    case "$a" in
      --model-basename=*)
        v="${a#*=}"
        if [[ -n "$v" ]]; then
          return 0
        fi
        echo "error: empty --model-basename=" >&2
        return 2
        ;;
      --model-basename)
        j=$((i + 1))
        if [[ $j -le $# ]]; then
          eval "v=\${$j}"
          if [[ -n "$v" && "${v:0:1}" != "-" ]]; then
            return 0
          fi
        fi
        echo "error: --model-basename requires a non-empty value" >&2
        return 2
        ;;
    esac
    i=$((i + 1))
  done
  return 1
}

_require_model_basename_or_exit() {
  local rc=0
  _require_model_basename_in_args "$@" || rc=$?
  if [[ $rc -eq 0 ]]; then
    return 0
  fi
  if [[ $rc -eq 2 ]]; then
    exit 1
  fi
  cat >&2 <<'EOF'
error: missing required --model-basename <stem>
  Writes <output>/<stem>.pkl and <output>/<stem>_meta.yaml . Choose a name that matches backend + experiment
  (e.g. bayesian_cost_full_merged, nn_cost_no_sgf).

Example:
  ./scripts/run_train_merged_no_sgf.sh \
    --model-basename nn_cost_no_sgf \
    --data-dir exp/train/gpu/seen_tasks/ \
    --output exp/models/gpu/seen_tasks/ \
    --test-split 0.1 -y 2 --model-kind nn
EOF
  exit 1
}
