#!/usr/bin/env bash
# Run on your Mac and leave it open. Forwards the VM's skycap servers (token mode :8080, text mode :8090),
# skycap-viewer (:8765) and skycap-codex-eval's viewer (:8766) to the same ports here, so 03_drive.py runs from
# the Mac against the VM (a trajectory's base_url is built from the address the request came in on, so it points
# through the tunnel too) and the viewers open at http://127.0.0.1:8765 (03_serve.sh starts it) and :8766
# (skycap-codex-eval/04_view.sh).
#   bash skyrl-test/skycap-responses/gcp/tunnel.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
echo "forwarding :8080 :8090 :8765 :8766 from $GCP_VM; Ctrl-C to stop"
gssh -- -N -o ExitOnForwardFailure=yes \
  -L 8080:127.0.0.1:8080 -L 8090:127.0.0.1:8090 -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766
