#!/usr/bin/env bash
# Run on your Mac. Gets the spot 8-GPU VM: creates it, or starts it again after a preemption (a preempted spot
# VM is stopped, not deleted: its boot disk, with the checkpoints and exports, is still there).
#
# Creating. Either kind of node in gcp/config.sh will do: 8 x B200 (a4-highgpu-8g) or 8 x H100
# (a3-highgpu-8g). Spot capacity for both comes and goes, so each round asks for the B200 node in each of its
# zones, then for the H100 node in each of its zones, and stops at the first VM it is given. The VM is named
# after what it got: kyuds-opd-b200 or kyuds-opd-h100. When every request is refused for lack of capacity the
# script prints one line, sleeps GCP_RETRY_SLEEP seconds (60) and goes round again, up to GCP_RETRIES rounds
# (240). A region that refuses for quota is not asked again for that kind. A request refused for any other
# reason is printed, and that zone is not asked again for that kind. GCP_VERBOSE=1 prints gcloud's message for
# every refusal.
#
# Starting again. A stopped VM restarts only where it is and as what it is, so the script waits for capacity in
# that one zone. To take whatever is available instead, delete the VM (gcp/down.sh delete: its disk goes with
# it) and run this again.
#
#   caffeinate -i bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh -y                  # skip the confirmation
#   GCP_H100_ZONES= bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh     # B200 only
#   GCP_B200_ZONES= bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh     # H100 only
# This starts billing for 8 GPUs. gcp/down.sh stops or deletes the VM.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
YES=0; [[ "${1:-}" == "-y" ]] && YES=1

# A VM between states (being created, stopping) settles within a minute or two. So does the record a refused
# request leaves behind for a few seconds, e.g. from a copy of this script that was just interrupted.
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  st="$(vm_status)"
  case "$st" in ""|RUNNING|TERMINATED|SUSPENDED) break ;; esac
  echo "$GCP_VM is $st in $GCP_ZONE; waiting for it to settle"
  sleep 10
  [[ -n "$GCP_VM_FIXED" ]] || vm_find
done
if [[ "$st" == "RUNNING" ]]; then echo "$GCP_VM (${GCP_VM_TYPE:-?}) is already running in $GCP_ZONE, project $GCP_PROJECT"; exit 0; fi

TARGETS=()   # "<kind>:<zone>", in the order they are asked for
if [[ -z "$st" ]]; then
  for kind in $GCP_KINDS; do for zone in $(kind_zones "$kind"); do TARGETS+=("$kind:$zone"); done; done
  [[ ${#TARGETS[@]} -gt 0 ]] || { echo "nothing to ask for: GCP_B200_ZONES and GCP_H100_ZONES are both empty" >&2; exit 1; }
  IMAGE="$(gcloud compute images describe-from-family "$GCP_IMAGE_FAMILY" --project "$GCP_IMAGE_PROJECT" --format='value(name)' 2>/dev/null || true)"
  if [[ -z "$IMAGE" ]]; then
    echo "image family $GCP_IMAGE_FAMILY is not available in $GCP_IMAGE_PROJECT (families are retired over time)." >&2
    echo "Families available now:" >&2
    gcloud compute images list --project "$GCP_IMAGE_PROJECT" --no-standard-images --format='value(family)' 2>/dev/null | sort -u | sed 's/^/  /' >&2
    echo "Pick the PyTorch one for Ubuntu 22.04 and re-run with GCP_IMAGE_FAMILY=<family> (or change gcp/config.sh)." >&2
    exit 1
  fi
  gcloud compute networks describe "$GCP_NETWORK" --project "$GCP_PROJECT" >/dev/null 2>&1 || {
    echo "network $GCP_NETWORK does not exist in $GCP_PROJECT (the guide assumes it does); set GCP_NETWORK/GCP_SUBNET" >&2; exit 1; }
  echo "will CREATE one SPOT VM in project $GCP_PROJECT, the first of these that has capacity:"
  for kind in $GCP_KINDS; do
    [[ -n "$(kind_zones "$kind")" ]] || continue
    echo "  $(kind_vm "$kind")  $(kind_machine "$kind") ($(kind_gpus "$kind"))  in: $(kind_zones "$kind")"
  done
  echo "  image $IMAGE, ${GCP_BOOT_DISK_GB}GB boot disk, network $GCP_NETWORK/$GCP_SUBNET, tags $GCP_TAGS"
else
  echo "will START $GCP_VM (${GCP_VM_TYPE:-?}, currently $st) in $GCP_ZONE, project $GCP_PROJECT"
  echo "  A stopped VM restarts only in that zone and as that machine type. To take whatever has capacity"
  echo "  instead, delete it first (gcp/down.sh delete: its disk goes with it) and run this again."
fi
if [[ $YES == 0 ]]; then
  read -r -p "This bills for 8 GPUs until the VM is stopped. Type yes to continue: " ans
  [[ "$ans" == "yes" ]] || { echo "aborted"; exit 1; }
fi

# Runs one gcloud command. Returns 0 when it worked, 1 when it was refused for lack of capacity (worth
# retrying), 2 when it was refused for quota, 3 for any other error (retrying will not help; its first lines
# are printed).
CAPACITY_ERRORS='RESOURCE_POOL_EXHAUSTED|not have enough resources|currently unavailable|stockout|try again later'
QUOTA_ERRORS='QUOTA_EXCEEDED|quota .* exceeded'
attempt_gcloud() {
  local out
  if out="$("$@" 2>&1)"; then echo "$out"; return 0; fi
  if grep -qiE "$QUOTA_ERRORS" <<<"$out"; then
    if [[ "${GCP_VERBOSE:-0}" == "1" ]]; then echo "$out" >&2; else echo "$out" | grep -iE "$QUOTA_ERRORS" | head -2 >&2; fi
    return 2
  fi
  if grep -qiE "$CAPACITY_ERRORS" <<<"$out"; then
    if [[ "${GCP_VERBOSE:-0}" == "1" ]]; then echo "$out" >&2; fi
    return 1
  fi
  if [[ "${GCP_VERBOSE:-0}" == "1" ]]; then echo "$out" >&2
  else out="$(sed -n '/^ERROR/,$p' <<<"$out" | grep . || echo "$out")"; head -8 <<<"$out" >&2; fi
  return 3
}
create_in() {   # $1 = kind, $2 = zone
  # --instance-termination-action=STOP keeps the boot disk when the VM is preempted. The subnet is given by
  # name: gcloud looks it up in the zone's region, and b200-vpc has a subnet called "$GCP_SUBNET" in every
  # region. Local SSDs are not requested: both machine types come with theirs attached.
  local disk=()
  if [[ -n "$(kind_disk "$1")" ]]; then disk=(--boot-disk-type "$(kind_disk "$1")"); fi
  attempt_gcloud gcloud compute instances create "$(kind_vm "$1")" \
    --project "$GCP_PROJECT" --zone "$2" \
    --machine-type "$(kind_machine "$1")" \
    --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --image-project "$GCP_IMAGE_PROJECT" --image-family "$GCP_IMAGE_FAMILY" \
    --boot-disk-size "${GCP_BOOT_DISK_GB}GB" ${disk[@]+"${disk[@]}"} \
    --tags "$GCP_TAGS" \
    --network-interface "nic-type=GVNIC,network=$GCP_NETWORK,subnet=$GCP_SUBNET" \
    --metadata-from-file "startup-script=$GCP_DIR/vm_startup.sh"
}
# "b200 in 4 zones, h100 in 10 zones" for the targets given as arguments
tally() {
  local k t n out=""
  for k in $GCP_KINDS; do
    n=0; for t in "$@"; do if [[ "${t%%:*}" == "$k" ]]; then n=$((n + 1)); fi; done
    if [[ $n -gt 0 ]]; then out="$out${out:+, }$k in $n zone$([[ $n == 1 ]] || echo s)"; fi
  done
  echo "$out"
}

GOT=0
DROPPED=" "   # "<kind>:<region>" refused for quota and "<kind>:<zone>" refused for another reason
for round in $(seq 1 "${GCP_RETRIES:-240}"); do
  if [[ ${#TARGETS[@]} -gt 0 ]]; then
    LEFT=()
    for t in "${TARGETS[@]}"; do
      kind="${t%%:*}"; zone="${t#*:}"; region="${zone%-*}"
      case "$DROPPED" in *" $kind:$region "*|*" $t "*) continue ;; esac
      rc=0; create_in "$kind" "$zone" || rc=$?
      case $rc in
        0) GCP_VM="$(kind_vm "$kind")"; GCP_ZONE="$zone"; GCP_VM_TYPE="$(kind_machine "$kind")"; GOT=1; break 2 ;;
        1) LEFT+=("$t") ;;
        2) DROPPED="$DROPPED$kind:$region "
           echo "$(date +%H:%M:%S) round $round  $kind in $zone: refused for quota; not asking for $kind in $region again" ;;
        *) DROPPED="$DROPPED$t "
           echo "$(date +%H:%M:%S) round $round  $kind in $zone: refused for a reason that waiting will not fix (above); not asking there again" ;;
      esac
    done
    [[ ${#LEFT[@]} -gt 0 ]] || { echo "nothing left to ask for: every zone was refused for quota or for an error" >&2; exit 1; }
    TARGETS=("${LEFT[@]}")
    echo "$(date +%H:%M:%S) round $round  no spot capacity: $(tally "${TARGETS[@]}")"
  else
    st="$(vm_status)"
    case "$st" in
      RUNNING) GOT=1; break ;;
      TERMINATED|SUSPENDED)
        rc=0; attempt_gcloud gcloud compute instances start "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" || rc=$?
        case $rc in
          0) GOT=1; break ;;
          1) echo "$(date +%H:%M:%S) round $round  $GCP_ZONE: no spot capacity (a stopped VM can only restart in its own zone)" ;;
          2) echo "$(date +%H:%M:%S) round $round  $GCP_ZONE: refused for quota (a stopped VM can only restart in its own zone)" ;;
          *) echo "that is neither a capacity nor a quota refusal, so retrying will not help; stopping." >&2; exit 1 ;;
        esac ;;
      "") echo "$GCP_VM no longer exists; run this script again to create one" >&2; exit 1 ;;
      *) echo "$GCP_VM is $st; waiting" ;;
    esac
  fi
  sleep "${GCP_RETRY_SLEEP:-60}"
done
[[ $GOT == 1 && "$(vm_status)" == "RUNNING" ]] || { echo "no VM after ${GCP_RETRIES:-240} rounds; run this again to keep trying" >&2; exit 1; }
echo "$GCP_VM (${GCP_VM_TYPE:-?}) is running in $GCP_ZONE. Next: bash skyrl-test/opd-gcp-spot-full-repro/gcp/02_setup_node.sh"
