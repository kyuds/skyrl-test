#!/usr/bin/env bash
# Run on your Mac. Copies one eval run's trajectories and analysis from the VM into this kit (git-ignored):
#   records/<run>/          the skycap records (what skycap-viewer and 02_analyze.sh read)
#   runs/<run>/             results.jsonl, config.json, analysis.md and analysis.json (when 02_analyze.sh has run),
#                           skycap.log, screenshots/ (when taken there)
#   runs/<run>.log          the run's own log
# Not the Codex homes and workspaces (runs/<run>/codex, runs/<run>/work): gigabytes, mostly Codex's own state. Get
# one rollout's with gcp/ssh.sh if needed.
#   bash skyrl-test/skycap-codex-eval/pull.sh <run>
# The VM tars what exists of these into one stream over a single ssh connection, unpacked here as it arrives.
# Pulling again overwrites what is here.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../skycap-responses/gcp/config.sh"
need_vm
run="${1:?usage: pull.sh <run>}"
remote="SkyRL/skyrl-test/skycap-codex-eval"
wanted="records/$run runs/$run/results.jsonl runs/$run/config.json runs/$run/analysis.md runs/$run/analysis.json"
wanted="$wanted runs/$run/skycap.log runs/$run/screenshots runs/$run.log"
mkdir -p "$here/records" "$here/runs"
echo "pulling $run's records and analysis from $GCP_VM"
# Only the paths that exist, so a run not yet analysed still pulls. gzip -1: the records are zstd already.
gssh --command "cd ~/$remote && [ -d records/$run ] || { echo 'no run $run on the VM' >&2; exit 1; }
  tar -cf - \$(for p in $wanted; do [ -e \"\$p\" ] && echo \"\$p\"; done) | gzip -1" | tar -xzf - -C "$here"
echo "records:  $here/records/$run ($(find "$here/records/$run" -name '*.json.zst' | wc -l | tr -d ' ') trajectories)"
echo "analysis: $(ls "$here/runs/$run" | tr '\n' ' ')"
