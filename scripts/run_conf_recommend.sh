#!/usr/bin/env bash
# Alias for ``run_recommend_conf.sh`` — CLI matches ``autoconfig recommend-conf`` / module ``autoconfig.conf_recommend``.
# Doc: ``docs/CONF_RECOMMEND.md``

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run_recommend_conf.sh" "$@"
