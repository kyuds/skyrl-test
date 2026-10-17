#!/usr/bin/env bash
# Run on your Mac, after 01_create_vm.sh and again after every preemption. Prepares the VM in three phases
# (gcp/node_setup.sh has the details): NVIDIA driver 580 with one reboot, the NVMe array and caches, then uv,
# SkyRL on $SKYRL_BRANCH with skyrl-test on $KIT_BRANCH, warm environments, Node, Codex and skycap-viewer. Safe
# to run again at any point: finished phases are skipped, and the long software phase runs detached on the VM, so
# a dropped connection or a sleeping laptop only detaches this script from it.
#   bash skyrl-test/skycap-responses/gcp/02_setup_node.sh
#   CODEX_VERSION=0.154.0 bash skyrl-test/skycap-responses/gcp/02_setup_node.sh      # pin Codex
# Push both branches first: the VM clones them from GitHub.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
[[ "$(vm_status)" == "RUNNING" ]] || { echo "$GCP_VM is not running: bash gcp/01_create_vm.sh" >&2; exit 1; }
echo "node $GCP_VM (${GCP_VM_TYPE:-?}, $GCP_ZONE)   SkyRL $SKYRL_BRANCH   kit $KIT_BRANCH"
for pair in "$SKYRL_REPO $SKYRL_BRANCH" "$KIT_REPO $KIT_BRANCH"; do
  read -r repo branch <<<"$pair"
  if ! git ls-remote --exit-code --heads "$repo" "$branch" >/dev/null 2>&1; then
    echo "branch $branch is not on $repo yet: push it first (the VM clones it from there)," >&2
    echo "or point SKYRL_REPO / KIT_REPO at the fork that has it" >&2
    exit 1
  fi
done
wait_ssh
push_setup

echo "--- driver"
out="$(gssh --command 'bash ~/skycap_node_setup.sh driver' | tee /dev/stderr)"
if grep -q REBOOT_NEEDED <<<"$out"; then
  echo "rebooting $GCP_VM (about 5 minutes)"
  gssh --command 'sudo reboot' >/dev/null 2>&1 || true
  sleep 45
  wait_ssh
  out="$(gssh --command 'bash ~/skycap_node_setup.sh driver' | tee /dev/stderr)"
  if grep -q REBOOT_NEEDED <<<"$out"; then
    echo "the driver or the open-file limit is still not right after a reboot; look at the output above" >&2; exit 1
  fi
fi

echo "--- storage"
gssh --command "SKYCAP_RAID_EXCLUDE=$(printf %q "${SKYCAP_RAID_EXCLUDE:-}") bash ~/skycap_node_setup.sh storage"

echo "--- software (10-20 minutes on a fresh node; detached on the VM, polled from here)"
gssh --command "$(printf 'SKYRL_REPO=%q SKYRL_BRANCH=%q KIT_REPO=%q KIT_BRANCH=%q CODEX_VERSION=%q' "$SKYRL_REPO" "$SKYRL_BRANCH" "$KIT_REPO" "$KIT_BRANCH" "$CODEX_VERSION") bash ~/skycap_node_setup.sh software-bg"
while :; do
  sleep 20
  out="$(gssh --command 'bash ~/skycap_node_setup.sh software-status' 2>/dev/null || echo SSH_FAILED)"
  case "$(head -1 <<<"$out")" in
    OK) echo "$out" | tail -n +2; break ;;
    FAILED) echo "$out" | tail -n +2; echo "the software phase failed; fix the cause and run this script again" >&2; exit 1 ;;
    RUNNING) echo "  $(date +%H:%M:%S) $(echo "$out" | sed -n 2p)" ;;
    *) echo "  $(date +%H:%M:%S) no answer from the VM; still trying" ;;
  esac
done
echo "node is ready. Next: bash skyrl-test/skycap-responses/gcp/03_serve.sh"
