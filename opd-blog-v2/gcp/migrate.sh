#!/usr/bin/env bash
# Run on your Mac. Moves a stopped node to any zone with capacity, keeping everything on its boot disk.
#
# A preempted spot VM is stopped, and can only start again in its own zone. When that zone has no capacity, this
# snapshots the node's boot disk, deletes the VM and the disk, and creates the node again, under the same name, in
# whichever zone has room, with its boot disk made from the snapshot. Everything a job keeps is on the boot disk
# (checkpoints, exports, job records, data, models, logs, keys), so the jobs resume as after an in-place restart.
# The NVMe array is only a cache and comes back empty, as after any preemption.
#
#   bash skyrl-test/opd-blog-v2/gcp/migrate.sh h100-3            # the plan; changes nothing
#   nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/migrate.sh h100-3 --yes > ~/opd_v2_migrate.log 2>&1 < /dev/null &
#   then:  bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh --only h100-3 --yes
#   and:   bash skyrl-test/opd-blog-v2/orchestrate.sh run <stage> --only h100-3 --yes
#
# The node must be stopped (TERMINATED): a running one is refused (gcp/down.sh stop <node> first, if you mean it).
# The order makes it safe to interrupt: the VM and its disk are deleted only once the snapshot is READY, and the
# snapshot is deleted only once the new node is up with the old files on it. Interrupted? Run it again with --yes: it
# picks up from $GCP_STATE_DIR/migrate.<node>. It asks for capacity like 01_create_nodes.sh (every zone of the kind, a
# minute between passes, the same cooldowns), for up to GCP_TIMEOUT_SECS (4 hours); touch $GCP_STATE_DIR/stop ends
# it early, keeping the snapshot. --keep-snapshot keeps the snapshot after a successful move as well.
#
# Cost: the snapshot is billed for the minutes to hours it exists (the disk's used space, not its 2,000 GB size), and a
# disk made from it in another region is billed as network egress, roughly $0.01 to $0.08 per GB.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
NODE="" YES=0 KEEP=0
for arg in "$@"; do
  case "$arg" in
    --yes) YES=1 ;;
    --keep-snapshot) KEEP=1 ;;
    -*) echo "usage: $0 <node> [--yes] [--keep-snapshot]" >&2; exit 2 ;;
    *) NODE="$arg" ;;
  esac
done
[[ -n "$NODE" ]] || { echo "usage: $0 <node> [--yes] [--keep-snapshot]" >&2; exit 2; }
[[ "$NODE" == "$GCP_NODE_PREFIX"-* ]] || NODE="$GCP_NODE_PREFIX-$NODE"
case "$NODE" in
  "$GCP_NODE_PREFIX"-h100-*) KIND=h100 ;;
  "$GCP_NODE_PREFIX"-b200-*) KIND=b200 ;;
  *) echo "$NODE is not a node of this kit ($GCP_NODE_PREFIX-h100-N or -b200-N)" >&2; exit 2 ;;
esac
STATE="$GCP_STATE_DIR"
mkdir -p "$STATE/cool"
SAVED="$STATE/migrate.$NODE"     # snapshot, disk type and size, from-zone: what a rerun needs once the VM is gone
log() { echo "$(date +%H:%M:%S) $*"; }
if acquire_running; then
  echo "01_create_nodes.sh is running and would start or recreate the node from scratch. Stop it first:" >&2
  echo "  touch $STATE/stop     (then wait for it to end, and remove $STATE/stop)" >&2
  exit 1
fi
rm -f "$STATE/stop"

# ---- where it stands ----
ROWS="$(nodes_rows)"
ROW="$(awk -v n="$NODE" '$1 == n' <<<"$ROWS")"
if [[ "$(grep -c . <<<"$ROW" || true)" -gt 1 ]]; then
  echo "$NODE exists in more than one zone; delete the extra one first:" >&2; sed 's/^/  /' <<<"$ROW" >&2; exit 1
fi
SNAP="" DTYPE="" DSIZE="" FROM=""
if [[ -f "$SAVED" ]]; then source "$SAVED"; fi

if [[ -n "$ROW" ]]; then
  read -r _ ZONE _ STATUS <<<"$ROW"
  made_from=""
  if [[ -n "$SNAP" ]]; then
    made_from="$(gcloud_read compute disks describe \
      "$(gcloud_read compute instances describe "$NODE" --project "$GCP_PROJECT" --zone "$ZONE" --format='value(disks[0].source.basename())')" \
      --project "$GCP_PROJECT" --zone "$ZONE" --format='value(sourceSnapshot.basename())')"
  fi
  if [[ -n "$SNAP" && "$made_from" == "$SNAP" ]]; then
    PHASE=done    # created from the snapshot on an earlier run, which stopped before cleaning up
  else
    [[ "$STATUS" == TERMINATED || "$STATUS" == SUSPENDED ]] || {
      echo "$NODE is $STATUS in $ZONE. Only a stopped node is moved; stop it first if you mean it:" >&2
      echo "  bash $GCP_DIR/down.sh stop ${NODE#"$GCP_NODE_PREFIX"-}" >&2
      exit 1
    }
    PHASE=snapshot
    DISK="$(gcloud_read compute instances describe "$NODE" --project "$GCP_PROJECT" --zone "$ZONE" \
      --format='value(disks[0].source.basename())')"
    read -r DTYPE DSIZE < <(gcloud_read compute disks describe "$DISK" --project "$GCP_PROJECT" --zone "$ZONE" \
      --format='value(type.basename(),sizeGb)')
    FROM="$ZONE"
    [[ -n "$SNAP" ]] || SNAP="$NODE-mig-$(date -u +%Y%m%d-%H%M%S)"
  fi
else
  [[ -n "$SNAP" ]] || { echo "there is no node called $NODE, and no migration of it to resume ($SAVED)" >&2; exit 1; }
  PHASE=create
fi

echo "migrate $NODE ($KIND):"
case "$PHASE" in
  snapshot) echo "  1. snapshot its boot disk $DISK ($DTYPE, ${DSIZE}GB) in $FROM as $SNAP"
            echo "  2. delete the VM and that disk, once the snapshot is READY" ;;
  create)   echo "  (resuming: the VM is gone; snapshot $SNAP of its $DTYPE ${DSIZE}GB disk from $FROM)" ;;
  done)     echo "  (resuming: $NODE was already created from $SNAP in $ZONE)" ;;
esac
[[ $PHASE == done ]] || echo "  3. create $NODE again from the snapshot, in any zone that offers $(kind_machine "$KIND"), $FROM last"
echo "  4. check its files are there, then $([[ $KEEP == 1 ]] && echo keep || echo delete) the snapshot"
if [[ $YES == 0 ]]; then echo "Plan only, nothing was changed. Add --yes to go."; exit 0; fi

# ---- 1, 2: snapshot, then delete ----
if [[ $PHASE == snapshot ]]; then
  printf 'SNAP=%q\nDTYPE=%q\nDSIZE=%q\nFROM=%q\n' "$SNAP" "$DTYPE" "$DSIZE" "$FROM" > "$SAVED"
  if ! gcloud compute snapshots describe "$SNAP" --project "$GCP_PROJECT" >/dev/null 2>&1; then
    log "snapshotting $DISK as $SNAP (minutes, in proportion to the data on it)"
    gcloud compute snapshots create "$SNAP" --project "$GCP_PROJECT" --source-disk "$DISK" --source-disk-zone "$FROM"
  fi
  until [[ "$(gcloud_read compute snapshots describe "$SNAP" --project "$GCP_PROJECT" --format='value(status)')" == READY ]]; do
    log "waiting for $SNAP to be READY"; sleep 20
  done
  log "$SNAP is READY ($(gcloud_read compute snapshots describe "$SNAP" --project "$GCP_PROJECT" --format='value(storageBytes)') bytes stored); deleting the VM and its disk"
  gcloud compute instances delete "$NODE" --project "$GCP_PROJECT" --zone "$FROM" --quiet
  if gcloud compute disks describe "$DISK" --project "$GCP_PROJECT" --zone "$FROM" >/dev/null 2>&1; then
    gcloud compute disks delete "$DISK" --project "$GCP_PROJECT" --zone "$FROM" --quiet
  fi
  PHASE=create
fi

# ---- 3: create it again, anywhere, from the snapshot ----
if [[ $PHASE == create ]]; then
  zones="$(kind_zones "$KIND")"
  zones="$(tr ' ' '\n' <<<"$zones" | grep . | grep -vx "$FROM" | tr '\n' ' ') $FROM"
  deadline=$(($(date +%s) + GCP_TIMEOUT_SECS))
  cooling() { [[ -f "$STATE/cool/$1" && $(date +%s) -lt $(cat "$STATE/cool/$1") ]]; }
  cool()    { echo $(($(date +%s) + $2)) > "$STATE/cool/$1"; }
  log "asking for $NODE from $SNAP in $(wc -w <<<"$zones" | tr -d ' ') zones"
  got=""
  while [[ -z "$got" ]]; do
    for zone in $zones; do
      [[ -e "$STATE/stop" || $(date +%s) -ge $deadline ]] && break 2
      region="${zone%-*}"
      if cooling "$KIND-$region" || cooling "$KIND-$zone"; then continue; fi
      if out="$(create_vm "$KIND" "$zone" "$NODE" "$SNAP" "$DTYPE" "$DSIZE" 2>&1)"; then got="$zone"; break; fi
      case "$(classify "$out")" in
        capacity|transient) ;;
        quota) log "quota refused in $region; leaving it for $((GCP_QUOTA_COOLDOWN / 60))m"; cool "$KIND-$region" "$GCP_QUOTA_COOLDOWN" ;;
        auth)  log "gcloud's login has expired: run  gcloud auth login  (this keeps trying)" ;;
        *)     log "$zone refused, leaving it for $((GCP_OTHER_COOLDOWN / 60))m:"; sed -n '/ERROR/,$p' <<<"$out" | head -3 | sed 's/^/    /'
               cool "$KIND-$zone" "$GCP_OTHER_COOLDOWN" ;;
      esac
    done
    [[ -n "$got" ]] || { [[ -e "$STATE/stop" || $(date +%s) -ge $deadline ]] && break; sleep "$GCP_RETRY_SLEEP"; }
  done
  if [[ -z "$got" ]]; then
    echo "no zone gave $NODE before the deadline or the stop file. The snapshot $SNAP is kept; run this again to keep asking." >&2
    exit 1
  fi
  log "GOT $NODE in $got"
  ZONE="$got"
fi

# ---- 4: check the files came along, then clean up ----
GCP_VM="$NODE"; GCP_ZONE="$ZONE"
for i in $(seq 1 30); do
  if files="$(gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command 'ls ~/opd-store/jobs 2>/dev/null | wc -l; du -sh ~/opd-store 2>/dev/null | cut -f1' 2>/dev/null)"; then break; fi
  files=""; sleep 20
done
[[ -n "$files" ]] || { echo "$NODE is up in $ZONE but did not answer ssh; the snapshot $SNAP is kept. Check it, then delete the snapshot by hand." >&2; exit 1; }
log "$NODE in $ZONE has its files: $(head -1 <<<"$files") job records, $(tail -1 <<<"$files") under ~/opd-store"
if [[ $KEEP == 0 ]]; then
  gcloud compute snapshots delete "$SNAP" --project "$GCP_PROJECT" --quiet
  rm -f "$SAVED"
  log "deleted the snapshot $SNAP"
else
  log "kept the snapshot $SNAP: gcloud compute snapshots delete $SNAP --quiet   when you are done with it"
fi
echo "Next: bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh --only ${NODE#"$GCP_NODE_PREFIX"-} --yes"
echo "Then: bash skyrl-test/opd-blog-v2/orchestrate.sh run <stage> --only ${NODE#"$GCP_NODE_PREFIX"-} --yes"
