# Settings for the gcp/*.sh scripts, which run on your Mac and drive one spot GPU VM through gcloud. Sourced by
# each of them; override anything from the environment, e.g. GCP_KINDS=h100x1 bash gcp/01_create_vm.sh.
# The VM follows Charlie's guide "Running SkyRL on GCP Spot B200s"
# (https://gist.github.com/CharlieFRuan/6eeae93d70ede0e81f12f91a4eb74d57) cut down to one node, as
# skyrl-test/opd-gcp-spot-full-repro/gcp/ does (these scripts are adapted from that kit's): one gVNIC on the
# existing b200-vpc, no RDMA networks, no shared filesystem. This kit needs one GPU (vLLM serving one model) and
# no Ray cluster.
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
# Any of these kinds of node will do. Each round asks for the first kind in each of its zones, then the next
# kind, and keeps the first VM it is given. GCP_KINDS sets which kinds and in what order.
#   h100x1  a3-highgpu-1g: 1 x H100 (80GB), 2 local SSDs. Enough for this kit, and the cheapest; offered as
#           spot (or flex-start) only, which is what this asks for.
#   b200    a4-highgpu-8g: 8 x B200, 32 local SSDs. The guide's node; 7 GPUs sit idle unless 01_engine.sh
#           gets --tensor-parallel-size.
#   h100    a3-highgpu-8g: 8 x H100, 16 local SSDs. Asked for when no single H100 is free; 7 GPUs sit idle
#           unless 01_engine.sh gets --tensor-parallel-size.
# By default one H100, else eight. B200s are off by default:  GCP_KINDS="h100x1 h100 b200"
GCP_KINDS="${GCP_KINDS:-h100x1 h100}"
# Zones. The H100 lists are every US zone that offered a3-highgpu-8g on 2026-10-05; a3-highgpu-1g is assumed to
# be offered in the same zones [not checked: gcloud's login had expired]. A zone that refuses for a reason other
# than capacity (e.g. the type isn't offered there) is dropped after one try, so a wrong guess costs one request.
# Spot H100 quota was seen in us-central1 (2026-10-05); a region that refuses for quota is not asked again.
# b200-vpc is an auto-mode network with a subnet called b200-vpc in every region, so the network flags work in
# all of these zones.
GCP_H100_ZONES="${GCP_H100_ZONES-us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west4-a us-east4-a us-east4-b us-east4-c us-east5-a}"
GCP_B200_ZONES="${GCP_B200_ZONES-us-west3-b us-west3-c us-south1-b us-central1-b}"
kind_machine() { case "$1" in h100x1) echo a3-highgpu-1g ;; h100) echo a3-highgpu-8g ;; b200) echo a4-highgpu-8g ;; esac; }
kind_zones()   { case "$1" in h100x1|h100) echo "$GCP_H100_ZONES" ;; b200) echo "$GCP_B200_ZONES" ;; esac; }
kind_gpus()    { case "$1" in h100x1) echo "1 x H100" ;; h100) echo "8 x H100" ;; b200) echo "8 x B200" ;; esac; }
# Boot disk type: the A3 does not take pd-standard, so pd-balanced; the A4 takes only Hyperdisk, so the choice is
# left to Compute Engine (the guide's way).
kind_disk()    { case "$1" in h100x1|h100) echo pd-balanced ;; b200) echo "" ;; esac; }

GCP_IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
# The guide's pytorch-2-7-cu128-ubuntu-2204-nvidia-570 no longer resolves (every image deprecated by 2026-10).
# Its successor: Ubuntu 22.04, Python 3.12, driver 580. 01_create_vm.sh lists the current families if it is gone.
GCP_IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
# Model weights and the uv/HF caches go to the local SSDs; the boot disk holds the checkouts and the records.
GCP_BOOT_DISK_GB="${GCP_BOOT_DISK_GB:-300}"
GCP_NETWORK="${GCP_NETWORK:-b200-vpc}"
GCP_SUBNET="${GCP_SUBNET:-b200-vpc}"
# b200-train is the guide's tag (the rule that lets VMs on b200-vpc reach each other); kyuds-skycap marks the VM
# as this kit's. ssh is allowed for every VM on the network.
GCP_TAGS="${GCP_TAGS:-b200-train,kyuds-skycap}"

# What 02_setup_node.sh checks out on the VM. Both branches must be pushed: the VM clones them from GitHub.
SKYRL_REPO="${SKYRL_REPO:-https://github.com/NovaSky-AI/SkyRL.git}"
SKYRL_BRANCH="${SKYRL_BRANCH:-kyuds/skycap-responses}"
KIT_REPO="${KIT_REPO:-https://github.com/kyuds/skyrl-test.git}"
KIT_BRANCH="${KIT_BRANCH:-$(git -C "$GCP_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
# The Codex CLI version _tools.sh installs on the VM ("latest" when empty).
CODEX_VERSION="${CODEX_VERSION:-}"

# ---- the VM ----
# Named after its kind: <base>-h100x1, <base>-b200 or <base>-h100; only one exists at a time. Every script finds
# it, and its zone, by asking the project. GCP_VM=<name> uses that name for any kind; with GCP_ZONE as well,
# nothing is looked up.
GCP_VM_BASE="${GCP_VM_BASE:-kyuds-skycap}"
GCP_VM_FIXED="${GCP_VM:-}"
kind_vm() { echo "${GCP_VM_FIXED:-$GCP_VM_BASE-$1}"; }
# Sets GCP_VM, GCP_ZONE and GCP_VM_TYPE to the VM that exists; leaves GCP_VM empty when there is none.
vm_find() {
  local pat rows
  if [[ -n "$GCP_VM_FIXED" ]]; then pat="^$GCP_VM_FIXED\$"; else pat="^$GCP_VM_BASE-(h100x1|b200|h100)\$"; fi
  GCP_VM=""; GCP_ZONE=""; GCP_VM_TYPE=""
  # A lookup that fails must not read as "there is no VM": gcloud's login expires after about a day on this
  # account, and from then on every call fails while the VM carries on.
  if ! rows="$(gcloud compute instances list --project "$GCP_PROJECT" --filter="name~'$pat'" --format='value(name,zone.basename(),machineType.basename())' 2>/dev/null)"; then
    rows="$(gcloud compute instances list --project "$GCP_PROJECT" --filter="name~'$pat'" --format='value(name)' 2>&1 >/dev/null || true)"
    echo "gcloud could not list the project's VMs, so whether the VM is there is not known:" >&2
    head -4 <<<"$rows" | sed 's/^/  /' >&2
    if grep -qiE 'reauth|auth login|credentials' <<<"$rows"; then
      echo "Your gcloud login has expired. Run:  gcloud auth login   The VM is not affected." >&2
    fi
    return 1
  fi
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
  echo "there is no VM yet (looked for $GCP_VM_BASE-{h100x1,b200,h100} in $GCP_PROJECT): bash skyrl-test/skycap-responses/gcp/01_create_vm.sh" >&2
  exit 1
}

gssh() { gcloud compute ssh "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
gscp() { gcloud compute scp --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
# Copies node_setup.sh to the VM, so every Mac-side script runs the current one.
push_setup() { gscp "$GCP_DIR/node_setup.sh" "$GCP_VM:~/skycap_node_setup.sh" >/dev/null; }
# RUNNING, TERMINATED (stopped or preempted), ..., empty when the VM does not exist, or UNKNOWN when gcloud
# could not say (its error goes to stderr); no script creates or starts anything on UNKNOWN.
vm_status() {
  [[ -n "$GCP_VM" ]] || return 0
  local out
  if out="$(gcloud compute instances describe "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --format='value(status)' 2>/dev/null)"; then
    echo "$out"; return 0
  fi
  out="$(gcloud compute instances describe "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --format='value(status)' 2>&1 >/dev/null || true)"
  if grep -q 'was not found' <<<"$out"; then return 0; fi
  echo "gcloud could not read the state of $GCP_VM:" >&2
  head -4 <<<"$out" | sed 's/^/  /' >&2
  echo UNKNOWN
}
wait_ssh() {
  local i
  for i in $(seq 1 60); do
    if gssh --command true >/dev/null 2>&1; then return 0; fi
    echo "  waiting for ssh on $GCP_VM ($i/60)"; sleep 10
  done
  echo "ssh to $GCP_VM did not come up in 10 minutes" >&2; return 1
}
