#!/usr/bin/env bash
# Run on your Mac. Is the VM alive? Asks gcloud only (no ssh), so it answers in a second or two, even while the VM
# is still booting. For what is running on it (GPUs, servers, records), use gcp/status.sh.
#   bash skyrl-test/skycap-responses/gcp/alive.sh                # one line
#   bash skyrl-test/skycap-responses/gcp/alive.sh --watch        # a line at every change (created, RUNNING, preempted,
#                                                                # ...), with a Mac notification; Ctrl-C to stop
#   bash skyrl-test/skycap-responses/gcp/alive.sh --watch 10     # check every 10 s (default 30)
# States: PROVISIONING / STAGING (booting), RUNNING, STOPPING, TERMINATED (stopped or preempted), or none.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
WATCH=0; EVERY=30
if [[ "${1:-}" == "--watch" ]]; then WATCH=1; EVERY="${2:-30}"; elif [[ -n "${1:-}" ]]; then echo "usage: $0 [--watch [seconds]]" >&2; exit 2; fi

# 2026-10-16T22:05:12.345-07:00 -> 2026-10-16 22:05
short_time() { sed -E 's/^([0-9-]+)T([0-9]+:[0-9]+).*/\1 \2/' <<<"$1"; }

# One line describing the VM, e.g.
#   kyuds-skycap-h100x1  RUNNING  a3-highgpu-1g SPOT  us-central1-c  up since 2026-10-16 22:05
describe() {
  [[ -n "$GCP_VM_FIXED" ]] || vm_find 2>/dev/null || { echo "gcloud could not list the VMs (expired login? gcloud auth login)"; return; }
  if [[ -z "$GCP_VM" ]]; then echo "no VM ($GCP_VM_BASE-{h100x1,h100,b200} in $GCP_PROJECT)"; return; fi
  local row st started stopped model type when=""
  if ! row="$(gcloud compute instances describe "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
      --format='csv[no-heading](status,lastStartTimestamp,lastStopTimestamp,scheduling.provisioningModel,machineType.basename())' 2>/dev/null)"; then
    st="$(vm_status 2>/dev/null)"
    if [[ -z "$st" ]]; then echo "$GCP_VM no longer exists"; else echo "$GCP_VM  state unknown (expired login? gcloud auth login)"; fi
    return
  fi
  # Comma-separated: tab is whitespace to `read`, so an empty field (no stop yet) would shift the rest.
  IFS=, read -r st started stopped model type <<<"$row"
  case "$st" in
    RUNNING) [[ -z "$started" ]] || when="up since $(short_time "$started")" ;;
    TERMINATED)
      [[ -z "$stopped" ]] || when="stopped $(short_time "$stopped")"
      # A preemption leaves an operation behind; a stop by down.sh does not.
      local preempted
      preempted="$(gcloud compute operations list --project "$GCP_PROJECT" \
        --filter="operationType=compute.instances.preempted AND targetLink~'/$GCP_VM\$'" \
        --sort-by=~insertTime --limit=1 --format='value(insertTime)' 2>/dev/null || true)"
      # It explains this stop when it came after the last start.
      if [[ -n "$preempted" && ( -z "$started" || "$preempted" > "$started" ) ]]; then when="PREEMPTED $(short_time "$preempted")"; fi
      when="$when; bring it back: gcp/01_create_vm.sh" ;;
    PROVISIONING|STAGING) when="booting" ;;
  esac
  echo "$GCP_VM  $st  ${type:-?} ${model:-}  $GCP_ZONE${when:+  $when}"
}

if [[ $WATCH == 0 ]]; then describe; exit 0; fi
echo "watching every ${EVERY}s; Ctrl-C to stop"
last=""
while :; do
  line="$(describe)"
  # Compare without the timestamps, so only a change of state prints.
  key="$(sed -E 's/  (up since|stopped|PREEMPTED) .*//' <<<"$line")"
  if [[ "$key" != "$last" ]]; then
    echo "$(date +%H:%M:%S) $line"
    if [[ -n "$last" ]] && command -v osascript >/dev/null 2>&1; then
      osascript -e "display notification \"$(sed 's/["\\]//g' <<<"$line")\" with title \"skycap VM\"" >/dev/null 2>&1 || true
    fi
    last="$key"
  fi
  sleep "$EVERY"
done
