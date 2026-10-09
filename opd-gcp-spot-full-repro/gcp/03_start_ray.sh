#!/usr/bin/env bash
# Run on your Mac, after 02_setup_node.sh and again after every preemption or reboot. Starts the single-node
# Ray cluster on the VM from SkyRL's base environment, with the node's NCCL settings in the raylet's
# environment (Ray workers inherit them), and prints the resources Ray sees: expect 8 GPUs.
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/03_start_ray.sh
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/03_start_ray.sh --restart    # stops Ray first; kills a run in progress
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
case "${1:-}" in ""|--restart) ;; *) echo "usage: $0 [--restart]" >&2; exit 2 ;; esac
need_vm
gscp "$GCP_DIR/node_setup.sh" "$GCP_VM:~/opd_node_setup.sh" >/dev/null
gssh --command "bash ~/opd_node_setup.sh ray ${1:-}"
echo "Next: bash skyrl-test/opd-gcp-spot-full-repro/gcp/04_push_secrets.sh   (once per VM), then gcp/ssh.sh"
