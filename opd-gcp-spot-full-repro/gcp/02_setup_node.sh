#!/usr/bin/env bash
# Run on your Mac, after 01_create_vm.sh and again after every preemption. Prepares the VM in three phases
# (gcp/node_setup.sh has the details): NVIDIA driver 580 with one reboot, the NVMe array and caches, then uv,
# SkyRL on $SKYRL_BRANCH with this kit on $KIT_BRANCH, and a warm environment. Safe to run again at any point:
# finished phases are skipped, and the long software phase runs detached on the VM, so a dropped connection
# or a sleeping laptop only detaches this script from it.
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/02_setup_node.sh
# Push the kit branch first: the VM clones it from GitHub.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
[[ "$(vm_status)" == "RUNNING" ]] || { echo "$GCP_VM is not running: bash gcp/01_create_vm.sh" >&2; exit 1; }
echo "node $GCP_VM (${GCP_VM_TYPE:-?}, $GCP_ZONE)   SkyRL $SKYRL_BRANCH   kit $KIT_BRANCH"
if ! git -C "$GCP_DIR" ls-remote --exit-code --heads "$KIT_REPO" "$KIT_BRANCH" >/dev/null 2>&1; then
  echo "branch $KIT_BRANCH is not on $KIT_REPO yet: push it first (the VM clones the kit from there)" >&2; exit 1
fi
wait_ssh
gscp "$GCP_DIR/node_setup.sh" "$GCP_VM:~/opd_node_setup.sh" >/dev/null

echo "--- driver"
out="$(gssh --command 'bash ~/opd_node_setup.sh driver' | tee /dev/stderr)"
if grep -q REBOOT_NEEDED <<<"$out"; then
  echo "rebooting $GCP_VM (about 5 minutes)"
  gssh --command 'sudo reboot' >/dev/null 2>&1 || true
  sleep 45
  wait_ssh
  out="$(gssh --command 'bash ~/opd_node_setup.sh driver' | tee /dev/stderr)"
  if grep -q REBOOT_NEEDED <<<"$out"; then
    echo "the driver or the open-file limit is still not right after a reboot; look at the output above" >&2; exit 1
  fi
fi

echo "--- storage"
gssh --command "OPD_RAID_EXCLUDE=$(printf %q "${OPD_RAID_EXCLUDE:-}") bash ~/opd_node_setup.sh storage"

echo "--- software (10-20 minutes on a fresh node; detached on the VM, polled from here)"
gssh --command "$(printf 'SKYRL_REPO=%q SKYRL_BRANCH=%q KIT_REPO=%q KIT_BRANCH=%q' "$SKYRL_REPO" "$SKYRL_BRANCH" "$KIT_REPO" "$KIT_BRANCH") bash ~/opd_node_setup.sh software-bg"
while :; do
  sleep 20
  out="$(gssh --command 'bash ~/opd_node_setup.sh software-status' 2>/dev/null || echo SSH_FAILED)"
  case "$(head -1 <<<"$out")" in
    OK) echo "$out" | tail -n +2; break ;;
    FAILED) echo "$out" | tail -n +2; echo "the software phase failed; fix the cause and run this script again" >&2; exit 1 ;;
    RUNNING) echo "  $(date +%H:%M:%S) $(echo "$out" | sed -n 2p)" ;;
    *) echo "  $(date +%H:%M:%S) no answer from the VM; still trying" ;;
  esac
done
echo "node is ready. Next: bash skyrl-test/opd-gcp-spot-full-repro/gcp/03_start_ray.sh"
