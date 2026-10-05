#!/usr/bin/env bash
# Run on your Mac. Stop paying for the GPUs.
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/down.sh stop      # keeps the boot disk (checkpoints, exports, data, the
#                                                                 # checkouts); you pay for the disk only. 01_create_vm.sh restarts it.
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/down.sh delete    # deletes the VM AND its boot disk; asks for the VM name
# Either way the NVMe array (HF and uv caches) is discarded. Anything not uploaded to the Hub is gone after `delete`.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
case "${1:-}" in stop|delete) need_vm ;; esac
case "${1:-}" in
  stop)
    gcloud compute instances stop "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --discard-local-ssd=true ;;
  delete)
    echo "This deletes $GCP_VM and its boot disk in $GCP_PROJECT / $GCP_ZONE: every checkpoint and every export on it."
    read -r -p "Type the VM name to confirm: " ans
    [[ "$ans" == "$GCP_VM" ]] || { echo "aborted"; exit 1; }
    gcloud compute instances delete "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --quiet ;;
  *) echo "usage: $0 stop|delete" >&2; exit 2 ;;
esac
