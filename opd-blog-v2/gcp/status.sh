#!/usr/bin/env bash
# Run on your Mac. Liveness check: which nodes exist, in what state, and whether they answer.
#   bash skyrl-test/opd-blog-v2/gcp/status.sh              # every node and its state, and the GPUs held
#   bash skyrl-test/opd-blog-v2/gcp/status.sh alive        # also ssh into every running node and count its GPUs
#   bash skyrl-test/opd-blog-v2/gcp/status.sh h100-1       # the same check for one node
# TERMINATED means preempted or stopped: the node is not billing for GPUs and its boot disk is still there.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"

if [[ -n "${1:-}" && "$1" != "alive" ]]; then
  resolve_node "$1"
  if [[ "$GCP_VM_STATUS" != RUNNING ]]; then echo "$GCP_VM is $GCP_VM_STATUS in $GCP_ZONE"; exit 1; fi
  gssh --ssh-flag="-o ConnectTimeout=20" --command \
    'echo "gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed "s/^ *//" | tr "\n" " ")  driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)  up: $(uptime -p)"'
  exit 0
fi

rows="$(nodes_rows)"
if [[ -z "$rows" ]]; then
  echo "no nodes in $GCP_PROJECT"
else
  { echo "NODE ZONE MACHINE STATUS"; echo "$rows"; } | column -t
  gpus=0
  for kind in $GCP_KINDS; do
    have="$(awk -v k="-$kind-" 'index($1, k) && $4 == "RUNNING"' <<<"$rows" | wc -l | tr -d ' ')"
    echo "$kind: $have of $(kind_count "$kind") running"
    gpus=$((gpus + have * 8))
  done
  echo "GPUs held: $gpus"
fi
if acquire_running; then
  echo "01_create_nodes.sh is running; it stops asking at $(date -r "$(cat "$GCP_STATE_DIR/deadline")" +%H:%M)"
fi
if [[ "${1:-}" == "alive" && -n "$rows" ]]; then
  echo
  on_all_nodes "$0"
fi
