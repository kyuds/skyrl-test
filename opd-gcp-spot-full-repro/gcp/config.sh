# Settings for the gcp/*.sh scripts, which run on your Mac and drive one spot VM through gcloud. Sourced by
# each of them; override anything from the environment, e.g. GCP_VM=kyuds-opd-2 bash gcp/01_create_vm.sh.
# The VM follows Charlie's guide "Running SkyRL on GCP Spot B200s"
# (https://gist.github.com/CharlieFRuan/6eeae93d70ede0e81f12f91a4eb74d57), cut down to a single node: one
# a4-highgpu-8g (8 x B200), one gVNIC on the existing b200-vpc, no RDMA networks, NCCL over plain sockets.
GCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v gcloud >/dev/null 2>&1 || {
  echo "gcloud is not installed. On a Mac: brew install --cask gcloud-cli   then: gcloud auth login" >&2
  return 1 2>/dev/null || exit 1
}
# The project is read from your gcloud config, not written into this public repo:
#   gcloud config set project <project id>      (the id is in Charlie's guide)
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "$GCP_PROJECT" || "$GCP_PROJECT" == "(unset)" ]]; then
  echo "no GCP project: gcloud config set project <project id>, or export GCP_PROJECT" >&2
  return 1 2>/dev/null || exit 1
fi
GCP_ZONE="${GCP_ZONE:-us-west3-b}"
GCP_VM="${GCP_VM:-kyuds-opd-b200}"
GCP_MACHINE_TYPE="${GCP_MACHINE_TYPE:-a4-highgpu-8g}"
GCP_IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
# The guide names pytorch-2-7-cu128-ubuntu-2204-nvidia-570, but every image of that family was deprecated by
# 2026-10 and it no longer resolves. This is its successor (Ubuntu 22.04, Python 3.12, driver 580 preinstalled).
# Families keep being retired; 01_create_vm.sh checks this one and lists the current ones if it is gone.
GCP_IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
# The guide uses 500GB. Doubled here because checkpoints and HF exports go to the boot disk (the only disk a
# preemption leaves intact): 4 runs x (2 checkpoints + an export every 10 steps) of 4B and 1.7B models.
GCP_BOOT_DISK_GB="${GCP_BOOT_DISK_GB:-1000}"
GCP_NETWORK="${GCP_NETWORK:-b200-vpc}"
GCP_SUBNET="${GCP_SUBNET:-b200-vpc}"
GCP_TAGS="${GCP_TAGS:-b200-train}"

# What 02_setup_node.sh checks out on the VM. The kit branch defaults to the branch this checkout is on,
# so push it before setting the node up.
SKYRL_REPO="${SKYRL_REPO:-https://github.com/NovaSky-AI/SkyRL.git}"
SKYRL_BRANCH="${SKYRL_BRANCH:-kyuds/opd-entrypoint}"
KIT_REPO="${KIT_REPO:-https://github.com/kyuds/skyrl-test.git}"
KIT_BRANCH="${KIT_BRANCH:-$(git -C "$GCP_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

gssh() { gcloud compute ssh "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
gscp() { gcloud compute scp --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
# RUNNING, TERMINATED (stopped or preempted), ..., or empty when the VM does not exist
vm_status() { gcloud compute instances describe "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --format='value(status)' 2>/dev/null || true; }
wait_ssh() {
  local i
  for i in $(seq 1 60); do
    if gssh --command true >/dev/null 2>&1; then return 0; fi
    echo "  waiting for ssh on $GCP_VM ($i/60)"; sleep 10
  done
  echo "ssh to $GCP_VM did not come up in 10 minutes" >&2; return 1
}
