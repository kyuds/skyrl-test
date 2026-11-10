#!/usr/bin/env bash
# Run on your Mac. Copies the VM's records (and the driver's results) into this kit's records/ and runs/ here,
# under the VM's name, so they can be kept or viewed on the Mac after the VM is gone:
#   records/<vm>/<server>/...   runs/<vm>/*.jsonl   runs/<vm>/codex-*/
#   bash skyrl-test/skycap-responses/gcp/pull_records.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
kit="$(cd "$GCP_DIR/.." && pwd)"
mkdir -p "$kit/records/$GCP_VM" "$kit/runs/$GCP_VM"
gscp --recurse "$GCP_VM:~/SkyRL/skyrl-test/skycap-responses/records/*" "$kit/records/$GCP_VM/"
gscp "$GCP_VM:~/SkyRL/skyrl-test/skycap-responses/runs/*.jsonl" "$kit/runs/$GCP_VM/" 2>/dev/null || true
# 04_codex.sh's runs: Codex's own log and events, one directory each.
gscp --recurse "$GCP_VM:~/SkyRL/skyrl-test/skycap-responses/runs/codex-*" "$kit/runs/$GCP_VM/" 2>/dev/null || true
echo "records in $kit/records/$GCP_VM; view them with: node <skycap-viewer>/bin/skycap-viewer.js $kit/records/$GCP_VM"
