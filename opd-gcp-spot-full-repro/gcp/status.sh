#!/usr/bin/env bash
# Run on your Mac. Is the VM still there (spot VMs get preempted), and what is it doing: GPUs, disks, Ray,
# finished and uploaded runs, the tail of the newest log.
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/status.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
st="$(vm_status)"
echo "vm     : $GCP_VM  ${st:-DOES NOT EXIST}  ($GCP_PROJECT / $GCP_ZONE)"
case "$st" in
  RUNNING) ;;
  TERMINATED) echo "stopped or preempted. Bring it back: gcp/01_create_vm.sh, gcp/02_setup_node.sh, gcp/03_start_ray.sh, then re-run the same run command on the VM."; exit 0 ;;
  *) exit 0 ;;
esac
gscp "$GCP_DIR/node_setup.sh" "$GCP_VM:~/opd_node_setup.sh" >/dev/null
gssh --command 'bash ~/opd_node_setup.sh status'
