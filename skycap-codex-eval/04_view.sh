#!/usr/bin/env bash
# Run ON the VM. skycap-viewer on the eval's records (every run is its own directory there), on :8766, which
# gcp/tunnel.sh forwards to your Mac. Long-running: start it with nohup (README).
#   bash skyrl-test/skycap-codex-eval/04_view.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
viewer="$SKYCAP_TOOLS/skycap-viewer/bin/skycap-viewer.js"
command -v node >/dev/null && [[ -f "$viewer" ]] || { echo "no skycap-viewer: bash $KIT_DIR/_tools.sh" >&2; exit 1; }
exec node "$viewer" "$EVAL_RECORDS" --host 127.0.0.1 --port "$EVAL_VIEWER_PORT"
