#!/usr/bin/env bash
# Run on your Mac. Creates the spot 8xB200 VM, or starts it again after a preemption (a preempted spot VM is
# stopped, not deleted: its boot disk, with the checkpoints and exports, is still there). Spot capacity for
# B200s comes and goes, so a refused request is retried every GCP_RETRY_SLEEP seconds (60) up to GCP_RETRIES
# times (240, i.e. four hours); leave it running under caffeinate.
#   caffeinate -i bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh -y        # skip the confirmation
# This starts billing for 8 B200s. gcp/down.sh stops or deletes the VM.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
YES=0; [[ "${1:-}" == "-y" ]] && YES=1

st="$(vm_status)"
if [[ "$st" == "RUNNING" ]]; then echo "$GCP_VM is already running ($GCP_PROJECT / $GCP_ZONE)"; exit 0; fi
if [[ -z "$st" ]]; then
  echo "will CREATE $GCP_VM: $GCP_MACHINE_TYPE (8 x B200), SPOT, $GCP_ZONE, project $GCP_PROJECT,"
  echo "  image $GCP_IMAGE_FAMILY, ${GCP_BOOT_DISK_GB}GB boot disk, network $GCP_NETWORK/$GCP_SUBNET, tags $GCP_TAGS"
  gcloud compute networks describe "$GCP_NETWORK" --project "$GCP_PROJECT" >/dev/null 2>&1 || {
    echo "network $GCP_NETWORK does not exist in $GCP_PROJECT (the guide assumes it does); set GCP_NETWORK/GCP_SUBNET" >&2; exit 1; }
else
  echo "will START $GCP_VM (currently $st) in $GCP_ZONE, project $GCP_PROJECT"
fi
if [[ $YES == 0 ]]; then
  read -r -p "This bills for 8 B200s until the VM is stopped. Type yes to continue: " ans
  [[ "$ans" == "yes" ]] || { echo "aborted"; exit 1; }
fi

for attempt in $(seq 1 "${GCP_RETRIES:-240}"); do
  st="$(vm_status)"
  if [[ "$st" == "RUNNING" ]]; then break; fi
  if [[ -z "$st" ]]; then
    # --instance-termination-action=STOP keeps the boot disk when the VM is preempted. No --boot-disk-type:
    # gcloud picks the type the machine family supports, as in the guide.
    gcloud compute instances create "$GCP_VM" \
      --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
      --machine-type "$GCP_MACHINE_TYPE" \
      --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
      --scopes=cloud-platform \
      --image-project "$GCP_IMAGE_PROJECT" --image-family "$GCP_IMAGE_FAMILY" \
      --boot-disk-size "${GCP_BOOT_DISK_GB}GB" \
      --tags "$GCP_TAGS" \
      --network-interface "nic-type=GVNIC,network=$GCP_NETWORK,subnet=$GCP_SUBNET" \
      --metadata-from-file "startup-script=$GCP_DIR/vm_startup.sh" && break
  elif [[ "$st" == "TERMINATED" || "$st" == "SUSPENDED" ]]; then
    gcloud compute instances start "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" && break
  else
    echo "$GCP_VM is $st; waiting"
  fi
  echo "attempt $attempt did not get the VM (no spot capacity is the usual reason); retrying in ${GCP_RETRY_SLEEP:-60}s"
  sleep "${GCP_RETRY_SLEEP:-60}"
done
[[ "$(vm_status)" == "RUNNING" ]] || { echo "could not get $GCP_VM running" >&2; exit 1; }
echo "$GCP_VM is running. Next: bash skyrl-test/opd-gcp-spot-full-repro/gcp/02_setup_node.sh"
