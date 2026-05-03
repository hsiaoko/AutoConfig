#!/usr/bin/env bash
# Generate merge-ready system config YAMLs (one per file), LHS or product grid in unit 8-cube.
# Reuses :file:`data/conf/build_ten_conf.py` for resource mapping and GPU constraints.
#
# Examples:
#   ./scripts/generate_confs.sh -n 50 -o exp/conf/out-of-core --method lhs
#   ./scripts/generate_confs.sh -n 20 -o /tmp/cfgs --method grid --cpu-max 64 --mem-max 64
#
# See:  .venv/bin/python scripts/generate_confs.py -h

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv/bin/python"
if [[ -x "$VENV" ]]; then
  exec "$VENV" "${ROOT}/scripts/generate_confs.py" "$@"
fi
exec python3 "${ROOT}/scripts/generate_confs.py" "$@"
