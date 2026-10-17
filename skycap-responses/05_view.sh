#!/usr/bin/env bash
# skycap-viewer on this kit's records: every server's record directory shows up as its own run.
# Long-running: leave it in a terminal, or nohup it (README). Needs Node 22.15+ and the viewer (_tools.sh).
#
#   bash skyrl-test/skycap-responses/05_view.sh            serve on 127.0.0.1:8765
#   bash skyrl-test/skycap-responses/05_view.sh summary    a text summary of each run
#   bash skyrl-test/skycap-responses/05_view.sh summary --json
#
# A remote node's viewer is reached through a port forward, e.g. `ssh -L 8765:127.0.0.1:8765 <node>`.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
viewer="$SKYCAP_TOOLS/skycap-viewer/bin/skycap-viewer.js"
command -v node >/dev/null || { echo "node is not on PATH: bash $KIT_DIR/_tools.sh"; exit 1; }
[[ -f "$viewer" ]] || { echo "no skycap-viewer at $viewer: bash $KIT_DIR/_tools.sh"; exit 1; }
if [[ "${1:-}" == summary || "${1:-}" == check ]]; then
  command="$1"; shift
  exec node "$viewer" "$command" "$SKYCAP_RECORDS" "$@"
fi
exec node "$viewer" "$SKYCAP_RECORDS" --host 127.0.0.1 --port "${SKYCAP_VIEWER_PORT:-8765}" "$@"
