#!/usr/bin/env bash
# Alias for ``run_recommend_conf_dir.sh`` (config **directory** + optional env vars). See ``docs/CONF_RECOMMEND.md``.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run_recommend_conf_dir.sh" "$@"
