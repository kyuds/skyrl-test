#!/usr/bin/env bash
# Run on your Mac. Stop paying for the GPUs.
#   bash skyrl-test/opd-blog-v2/gcp/down.sh stop all        # keeps every boot disk; you pay for the disks only
#   bash skyrl-test/opd-blog-v2/gcp/down.sh stop h100-1     # one node
#   bash skyrl-test/opd-blog-v2/gcp/down.sh delete all      # deletes the VMs AND their boot disks; asks first
#   bash skyrl-test/opd-blog-v2/gcp/down.sh delete h100-1   # one node; asks for its name
# Stopped nodes come back through 01_create_nodes.sh, in the zone they were in. It restarts any it finds stopped,
# so this script refuses to run while 01_create_nodes.sh is running.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
action="${1:-}"; target="${2:-}"
case "$action" in stop|delete) ;; *) echo "usage: $0 stop|delete <node>|all" >&2; exit 2 ;; esac
[[ -n "$target" ]] || { echo "usage: $0 stop|delete <node>|all" >&2; exit 2; }
if acquire_running; then
  echo "01_create_nodes.sh is still running and would bring the node back. Stop it first:" >&2
  echo "  touch $GCP_STATE_DIR/stop" >&2
  exit 1
fi

# Every node to act on, as "<name> <zone>". GCP_ZONE=<zone> picks one when a name exists in two zones.
rows="$(nodes_rows)"
if [[ "$target" == all ]]; then
  if [[ "$action" == stop ]]; then targets="$(awk '$4 == "RUNNING" {print $1, $2}' <<<"$rows")"; else targets="$(awk 'NF {print $1, $2}' <<<"$rows")"; fi
else
  if [[ "$target" != "$GCP_NODE_PREFIX"-* ]]; then target="$GCP_NODE_PREFIX-$target"; fi
  targets="$(awk -v n="$target" -v z="${GCP_ZONE:-}" '$1 == n && (z == "" || $2 == z) {print $1, $2}' <<<"$rows")"
fi
if [[ -z "$targets" ]]; then echo "nothing to $action"; exit 0; fi
if [[ "$(awk '{print $1}' <<<"$targets" | sort | uniq -d)" != "" && "$target" != all ]]; then
  echo "$target exists in more than one zone; pick one with GCP_ZONE=<zone>:" >&2
  sed 's/^/  /' <<<"$targets" >&2
  exit 1
fi

if [[ "$action" == delete ]]; then
  echo "This deletes these VMs and their boot disks in $GCP_PROJECT, with everything on them:"
  sed 's/^/  /' <<<"$targets"
  want="${target}"
  read -r -p "Type $want to confirm: " ans || { echo "aborted: run this in a terminal, it has to ask"; exit 1; }
  [[ "$ans" == "$want" ]] || { echo "aborted"; exit 1; }
fi

# One gcloud at a time would leave the others billing for minutes.
pids=(); names=()
while read -r name zone; do
  if [[ "$action" == stop ]]; then
    gcloud compute instances stop "$name" --project "$GCP_PROJECT" --zone "$zone" --discard-local-ssd=true &
  else
    gcloud compute instances delete "$name" --project "$GCP_PROJECT" --zone "$zone" --quiet &
  fi
  pids+=($!); names+=("$name")
done <<<"$targets"
rc=0; i=0
for name in "${names[@]}"; do
  if ! wait "${pids[$i]}"; then echo "$action failed for $name" >&2; rc=1; fi
  i=$((i + 1))
done
exit $rc
