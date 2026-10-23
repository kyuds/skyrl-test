#!/usr/bin/env bash
# The graph-structure analysis of one eval run (02_analyze.py), then skycap-viewer's own format check and summary.
# Runs on the VM, or on the Mac after pull.sh. Writes runs/<run>/analysis.md and analysis.json; exits 1 when an
# invariant fails or the viewer's check does.
#   bash skyrl-test/skycap-codex-eval/02_analyze.sh <run>
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
run="${1:?usage: 02_analyze.sh <run> (a directory under records/)}"
records="$EVAL_RECORDS/$run"
[[ -d "$records" ]] || { echo "no records/$run here (on the Mac: bash skyrl-test/skycap-codex-eval/pull.sh $run)" >&2; exit 1; }
status=0
# skycap's environment, for skycap.record and skycap.samples.
(cd "$SKYRL_DIR/skycap" && uv run --isolated --extra tokens python "$EVAL_DIR/02_analyze.py" "$records" --out "$EVAL_DIR/runs/$run") || status=$?
viewer="$SKYCAP_TOOLS/skycap-viewer/bin/skycap-viewer.js"
if command -v node >/dev/null && [[ -f "$viewer" ]]; then
  echo "--- skycap-viewer check"
  node "$viewer" check "$records" || status=1
  echo "--- skycap-viewer summary"
  node "$viewer" summary "$records" || true
else
  echo "(skycap-viewer not installed here: its check and summary are skipped)"
fi
exit $status
