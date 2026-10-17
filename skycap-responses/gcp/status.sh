#!/usr/bin/env bash
# Run on your Mac. Is the VM still there (spot VMs get preempted), and what is it doing: GPUs, disks, vLLM, the
# skycap servers, the records, the newest drive.
#   bash skyrl-test/skycap-responses/gcp/status.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
if [[ -z "$GCP_VM" ]]; then echo "vm      : none (looked for $GCP_VM_BASE-{h100x1,b200,h100} in $GCP_PROJECT)"; exit 0; fi
st="$(vm_status)"
echo "vm      : $GCP_VM  ${st:-DOES NOT EXIST}  (${GCP_VM_TYPE:-?}, $GCP_PROJECT / $GCP_ZONE)"
case "$st" in
  RUNNING) ;;
  TERMINATED) echo "stopped or preempted. Bring it back: gcp/01_create_vm.sh, gcp/02_setup_node.sh, gcp/03_serve.sh."; exit 0 ;;
  *) exit 0 ;;
esac
push_setup
gssh --command 'bash ~/skycap_node_setup.sh status'
