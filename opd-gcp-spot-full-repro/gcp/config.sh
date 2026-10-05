# Settings for the gcp/*.sh scripts, which run on your Mac and drive one spot VM through gcloud. Sourced by
# each of them; override anything from the environment, e.g. GCP_H100_ZONES= bash gcp/01_create_vm.sh.
# The VM follows Charlie's guide "Running SkyRL on GCP Spot B200s"
# (https://gist.github.com/CharlieFRuan/6eeae93d70ede0e81f12f91a4eb74d57), cut down to a single node: one
# 8-GPU VM, one gVNIC on the existing b200-vpc, no RDMA networks, NCCL over plain sockets.
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

# ---- what 01_create_vm.sh asks for ----
# Either of two kinds of node will do. Each time round the script asks for the first kind in each of its
# zones, then for the second kind in each of its zones, and keeps the first VM it is given.
#   b200  a4-highgpu-8g: 8 x B200 (180GB each), 32 local SSDs. The guide's node.
#   h100  a3-highgpu-8g: 8 x H100 (80GB each), 16 local SSDs. The hardware the repo lists for its DAPO
#         reference runs of these two models (examples/train/algorithms/dapo/README.md). About half the speed.
# An empty zone list turns a kind off:   GCP_H100_ZONES= bash gcp/01_create_vm.sh    (B200 only)
#
# B200 zones. The guide pins us-west3-b because its RDMA network profile is zone-specific; a single node has
# no RDMA network, so any zone that offers the machine type will do. The four below all accepted the request
# on 2026-10-05 and refused it for capacity, not for quota. Also offered, never tried: us-east1-b us-east1-d
# us-east4-b us-west2-c. (GCP_ZONES is the old name of this list and is still read.)
# H100 zones: every US zone that offers the machine type. Whether the project has H100 quota in these regions
# is not known (its quota API is off, and the regional quota list has no H100 entry); a region that refuses
# for quota is not asked again for that kind.
# b200-vpc is an auto-mode network: it has a subnet called b200-vpc in every region, so the same network
# flags work in all of these zones.
GCP_B200_MACHINE_TYPE="${GCP_B200_MACHINE_TYPE:-a4-highgpu-8g}"
GCP_B200_ZONES="${GCP_B200_ZONES-${GCP_ZONES-us-west3-b us-west3-c us-south1-b us-central1-b}}"
GCP_H100_MACHINE_TYPE="${GCP_H100_MACHINE_TYPE:-a3-highgpu-8g}"
GCP_H100_ZONES="${GCP_H100_ZONES-us-west1-a us-west1-b us-west4-a us-central1-a us-central1-b us-central1-c us-east4-a us-east4-b us-east4-c us-east5-a}"
# Boot disk type. Empty leaves the choice to Compute Engine, as the guide does for the A4 (which only takes
# Hyperdisk). The A3 does not take pd-standard, so its type is named: pd-balanced, which it is documented to
# support.
GCP_B200_BOOT_DISK_TYPE="${GCP_B200_BOOT_DISK_TYPE-}"
GCP_H100_BOOT_DISK_TYPE="${GCP_H100_BOOT_DISK_TYPE-pd-balanced}"
GCP_KINDS="b200 h100"
kind_machine() { case "$1" in b200) echo "$GCP_B200_MACHINE_TYPE" ;; h100) echo "$GCP_H100_MACHINE_TYPE" ;; esac; }
kind_zones()   { case "$1" in b200) echo "$GCP_B200_ZONES" ;; h100) echo "$GCP_H100_ZONES" ;; esac; }
kind_disk()    { case "$1" in b200) echo "$GCP_B200_BOOT_DISK_TYPE" ;; h100) echo "$GCP_H100_BOOT_DISK_TYPE" ;; esac; }
kind_gpus()    { case "$1" in b200) echo "8 x B200" ;; h100) echo "8 x H100" ;; esac; }

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
# Network tags, comma-separated. b200-train is the guide's: it selects the rule that lets the VMs on b200-vpc
# reach each other (b200-allow-internal). kyuds-opd marks the VM as this kit's; no firewall rule names it.
# ssh is allowed by rules that apply to every VM on the network, whatever its tags.
GCP_TAGS="${GCP_TAGS:-b200-train,kyuds-opd}"

# What 02_setup_node.sh checks out on the VM. The kit branch defaults to the branch this checkout is on,
# so push it before setting the node up.
SKYRL_REPO="${SKYRL_REPO:-https://github.com/NovaSky-AI/SkyRL.git}"
SKYRL_BRANCH="${SKYRL_BRANCH:-kyuds/opd-entrypoint}"
KIT_REPO="${KIT_REPO:-https://github.com/kyuds/skyrl-test.git}"
KIT_BRANCH="${KIT_BRANCH:-$(git -C "$GCP_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

# ---- the VM ----
# It is named after the GPUs it got: <base>-b200 or <base>-h100, and only one of the two exists at a time.
# Every script finds it, and its zone, by asking the project. GCP_VM=<name> uses that one name for either kind;
# with GCP_ZONE as well, nothing is looked up.
GCP_VM_BASE="${GCP_VM_BASE:-kyuds-opd}"
GCP_VM_FIXED="${GCP_VM:-}"
kind_vm() { echo "${GCP_VM_FIXED:-$GCP_VM_BASE-$1}"; }
# Sets GCP_VM, GCP_ZONE and GCP_VM_TYPE to the VM that exists; leaves GCP_VM empty when there is none.
vm_find() {
  local pat rows
  if [[ -n "$GCP_VM_FIXED" ]]; then pat="^$GCP_VM_FIXED\$"; else pat="^$GCP_VM_BASE-(b200|h100)\$"; fi
  rows="$(gcloud compute instances list --project "$GCP_PROJECT" --filter="name~'$pat'" --format='value(name,zone.basename(),machineType.basename())' 2>/dev/null || true)"
  GCP_VM=""; GCP_ZONE=""; GCP_VM_TYPE=""
  [[ -n "$rows" ]] || return 0
  if [[ "$(wc -l <<<"$rows" | tr -d ' ')" -gt 1 ]]; then
    echo "more than one VM matches; pick one with GCP_VM=<name> GCP_ZONE=<zone>:" >&2
    sed 's/^/  /' <<<"$rows" >&2
    return 1
  fi
  read -r GCP_VM GCP_ZONE GCP_VM_TYPE <<<"$rows"
}
if [[ -n "${GCP_VM:-}" && -n "${GCP_ZONE:-}" ]]; then
  GCP_VM_TYPE="${GCP_VM_TYPE:-}"
else
  vm_find || { return 1 2>/dev/null || exit 1; }
fi
need_vm() {
  [[ -n "$GCP_VM" ]] && return 0
  echo "there is no VM yet (looked for $(kind_vm b200) and $(kind_vm h100) in $GCP_PROJECT): bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh" >&2
  exit 1
}

gssh() { gcloud compute ssh "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
gscp() { gcloud compute scp --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
# RUNNING, TERMINATED (stopped or preempted), ..., or empty when the VM does not exist
vm_status() {
  [[ -n "$GCP_VM" ]] || return 0
  gcloud compute instances describe "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --format='value(status)' 2>/dev/null || true
}
wait_ssh() {
  local i
  for i in $(seq 1 60); do
    if gssh --command true >/dev/null 2>&1; then return 0; fi
    echo "  waiting for ssh on $GCP_VM ($i/60)"; sleep 10
  done
  echo "ssh to $GCP_VM did not come up in 10 minutes" >&2; return 1
}
