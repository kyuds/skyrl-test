#!/usr/bin/env bash
# Run on your Mac and leave it open. Forwards the VM's skycap servers (:8080, :8081) and skycap-viewer (:8765) to
# the same ports here, so 03_drive.py runs from the Mac against the VM (a trajectory's base_url is built from the
# address the request came in on, so it points through the tunnel too) and the viewer opens at
# http://127.0.0.1:8765 (03_serve.sh starts the viewer on the VM).
#   bash skyrl-test/skycap-responses/gcp/tunnel.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
echo "forwarding :8080 :8081 :8765 from $GCP_VM; Ctrl-C to stop"
gssh -- -N -o ExitOnForwardFailure=yes \
  -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 -L 8765:127.0.0.1:8765
