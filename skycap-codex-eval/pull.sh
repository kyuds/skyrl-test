#!/usr/bin/env bash
# Run on your Mac. Copies one eval run from the VM: its records to records/<run>/ and its results, analysis, Codex
# logs and workspaces to runs/<run>/, both under this kit (git-ignored).
#   bash skyrl-test/skycap-codex-eval/pull.sh <run>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../skycap-responses/gcp/config.sh"
need_vm
run="${1:?usage: pull.sh <run>}"
mkdir -p "$here/records" "$here/runs"
gscp --recurse "$GCP_VM:~/SkyRL/skyrl-test/skycap-codex-eval/records/$run" "$here/records/"
gscp --recurse "$GCP_VM:~/SkyRL/skyrl-test/skycap-codex-eval/runs/$run" "$here/runs/"
echo "records in $here/records/$run; results and analysis in $here/runs/$run"
