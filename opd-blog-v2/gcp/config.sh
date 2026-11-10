# Settings and helpers for the gcp/*.sh scripts, which run on your Mac and drive this kit's spot VMs through gcloud.
# Sourced by each of them; override anything from the environment, e.g. GCP_B200_NODES=0 bash gcp/01_create_nodes.sh.
# Every VM is an independent 8-GPU node, set up like opd-gcp-spot-full-repro's: one gVNIC on b200-vpc, no RDMA
# networks, NCCL over plain sockets. The nodes do not form a cluster.
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

# ---- what 01_create_nodes.sh asks for ----
# The most nodes of each kind. Each node has a fixed name, so these are hard caps. 0 turns a kind off.
GCP_H100_NODES="${GCP_H100_NODES:-4}"
GCP_B200_NODES="${GCP_B200_NODES:-2}"
GCP_H100_MACHINE_TYPE="${GCP_H100_MACHINE_TYPE:-a3-highgpu-8g}"
GCP_B200_MACHINE_TYPE="${GCP_B200_MACHINE_TYPE:-a4-highgpu-8g}"
# Zones. Empty asks gcloud for every zone in every region that offers the machine type. A space-separated list
# narrows it, e.g. GCP_B200_ZONES="us-west3-b us-west3-c".
GCP_H100_ZONES="${GCP_H100_ZONES:-}"
GCP_B200_ZONES="${GCP_B200_ZONES:-}"
# Boot disk type. Empty leaves the choice to Compute Engine, as the guide does for the A4 (which only takes
# Hyperdisk). The A3 does not take pd-standard, so its type is named.
GCP_H100_BOOT_DISK_TYPE="${GCP_H100_BOOT_DISK_TYPE-pd-balanced}"
GCP_B200_BOOT_DISK_TYPE="${GCP_B200_BOOT_DISK_TYPE-}"
GCP_KINDS="h100 b200"
kind_machine() { case "$1" in h100) echo "$GCP_H100_MACHINE_TYPE" ;; b200) echo "$GCP_B200_MACHINE_TYPE" ;; esac; }
kind_disk()    { case "$1" in h100) echo "$GCP_H100_BOOT_DISK_TYPE" ;; b200) echo "$GCP_B200_BOOT_DISK_TYPE" ;; esac; }
kind_count()   { case "$1" in h100) echo "$GCP_H100_NODES" ;; b200) echo "$GCP_B200_NODES" ;; esac; }

# How long 01_create_nodes.sh keeps asking, and how it paces itself.
GCP_TIMEOUT_SECS="${GCP_TIMEOUT_SECS:-14400}"       # 4 hours
GCP_GRACE_SECS="${GCP_GRACE_SECS:-300}"             # after the timeout, how long a request in flight may finish
GCP_RETRY_SLEEP="${GCP_RETRY_SLEEP:-60}"            # between two passes over the zones
GCP_POLL_SECS="${GCP_POLL_SECS:-60}"                # how often a node that was obtained is checked for preemption
GCP_MONITOR_SECS="${GCP_MONITOR_SECS:-300}"         # how often the progress line is printed
GCP_QUOTA_COOLDOWN="${GCP_QUOTA_COOLDOWN:-1800}"    # how long a region that refused for quota is left alone
GCP_OTHER_COOLDOWN="${GCP_OTHER_COOLDOWN:-600}"     # same, for a zone that refused for any other reason
# On this Mac, outside the public repo: gcloud's messages can carry the project id.
GCP_STATE_DIR="${GCP_STATE_DIR:-$HOME/opd-blog-v2-gcp}"

GCP_IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
# Ubuntu 22.04, Python 3.12, driver 580 preinstalled. Families get retired; 01_create_nodes.sh checks this one and
# lists the current ones if it is gone.
GCP_IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
# Checkpoints, exports, models and data go to the boot disk, the only disk a preemption leaves intact. b200-1 holds
# 1a's SFT (a ~100GB checkpoint, three 16GB exports, 20GB of pretokenized data) and then 1b's OPD checkpoints and
# exports. 2000GB costs little next to the GPUs.
GCP_BOOT_DISK_GB="${GCP_BOOT_DISK_GB:-2000}"
GCP_NETWORK="${GCP_NETWORK:-b200-vpc}"      # auto-mode: it has a subnet called b200-vpc in every region
GCP_SUBNET="${GCP_SUBNET:-b200-vpc}"
GCP_TAGS="${GCP_TAGS:-b200-train,kyuds-opd}"

# ---- the nodes ----
# Named <prefix>-<kind>-<id>: kyuds-opd-v2-h100-1 to -6 and kyuds-opd-v2-b200-1, -2. VM names take only lowercase
# letters, digits and hyphens. The scripts below accept the short form too: h100-1.
GCP_NODE_PREFIX="${GCP_NODE_PREFIX:-kyuds-opd-v2}"
node_name()  { echo "$GCP_NODE_PREFIX-$1-$2"; }
node_regex() { echo "^$GCP_NODE_PREFIX-(h100|b200)-[0-9]+\$"; }

# Runs a read-only gcloud command and prints its stdout. On failure explains why and returns 1: a lookup that fails
# must not read as "there is no VM", since gcloud's login expires after about a day while the VMs carry on.
gcloud_read() {
  local err rc=0
  err="$(mktemp)"
  gcloud "$@" 2>"$err" || rc=$?
  if [[ $rc != 0 ]]; then
    echo "gcloud failed: gcloud $1 $2 $3" >&2
    head -4 "$err" | sed 's/^/  /' >&2
    if grep -qiE 'reauth|auth login|credentials' "$err"; then
      echo "Your gcloud login has expired. Run:  gcloud auth login   The VMs and any run on them are not affected." >&2
    fi
  fi
  rm -f "$err"
  return $rc
}
# One row per node of this kit, in any state: name zone machine-type status.
nodes_rows() {
  gcloud_read compute instances list --project "$GCP_PROJECT" --filter="name~'$(node_regex)'" \
    --format='value(name,zone.basename(),machineType.basename(),status)'
}
# Every zone that offers the kind's machine type, or the list set above.
kind_zones() {
  local listed=""
  case "$1" in h100) listed="$GCP_H100_ZONES" ;; b200) listed="$GCP_B200_ZONES" ;; esac
  if [[ -n "$listed" ]]; then echo "$listed"; return 0; fi
  gcloud_read compute machine-types list --project "$GCP_PROJECT" --filter="name=$(kind_machine "$1")" \
    --format='value(zone.basename())' | sort -u | tr '\n' ' '
}
# Sets GCP_VM, GCP_ZONE, GCP_VM_TYPE and GCP_VM_STATUS for one node, given its full or short name.
resolve_node() {
  local want="$1" rows row
  if [[ "$want" != "$GCP_NODE_PREFIX"-* ]]; then want="$GCP_NODE_PREFIX-$want"; fi
  rows="$(nodes_rows)" || return 1
  row="$(awk -v n="$want" '$1 == n' <<<"$rows")"
  if [[ -z "$row" ]]; then
    echo "there is no node called $want in $GCP_PROJECT. Nodes:" >&2
    if [[ -n "$rows" ]]; then sed 's/^/  /' <<<"$rows" >&2; else echo "  none" >&2; fi
    return 1
  fi
  if [[ "$(wc -l <<<"$row" | tr -d ' ')" -gt 1 ]]; then
    echo "$want exists in more than one zone; delete the extra one:" >&2
    sed 's/^/  /' <<<"$row" >&2
    return 1
  fi
  read -r GCP_VM GCP_ZONE GCP_VM_TYPE GCP_VM_STATUS <<<"$row"
}
# ---- what a refusal means (01_create_nodes.sh, migrate.sh) ----
# Order matters: an API rate limit says "Quota exceeded" too, and must not read as a quota refusal for GPUs.
AUTH_ERRORS='reauth|auth login|invalid_grant'
RATE_ERRORS='rate.?limit|quota metric|quota group|per minute|per 100 seconds'
CAPACITY_ERRORS='RESOURCE_POOL_EXHAUSTED|not have enough resources|currently unavailable|stockout|try again later'
QUOTA_ERRORS='QUOTA_EXCEEDED|quota .*exceeded'
TRANSIENT_ERRORS='database is locked|internal error|backendError|HTTPError 50[0-9]|Error 50[0-9]|connection (reset|aborted|refused)|timed out|temporary failure|server not found|ServerNotFound|remote end closed'
classify() {
  if grep -qiE "$AUTH_ERRORS" <<<"$1"; then echo auth
  elif grep -qiE "$RATE_ERRORS" <<<"$1"; then echo transient
  elif grep -qiE "$CAPACITY_ERRORS" <<<"$1"; then echo capacity
  elif grep -qiE "$QUOTA_ERRORS" <<<"$1"; then echo quota
  elif grep -qiE "$TRANSIENT_ERRORS" <<<"$1"; then echo transient
  else echo other
  fi
}

# Creates one spot node: SPOT, and --instance-termination-action=STOP so a preemption keeps the boot disk. Its boot
# disk comes from the image, or, for migrate.sh, from a snapshot of the node's old disk, with that disk's type and size.
create_vm() {   # $1 = kind, $2 = zone, $3 = node, [$4 = snapshot, $5 = boot disk type, $6 = boot disk GB]
  local dtype="${5:-$(kind_disk "$1")}" size="${6:-$GCP_BOOT_DISK_GB}" disk=() source=()
  if [[ -n "$dtype" ]]; then disk=(--boot-disk-type "$dtype"); fi
  if [[ -n "${4:-}" ]]; then source=(--source-snapshot "$4")
  else source=(--image-project "$GCP_IMAGE_PROJECT" --image-family "$GCP_IMAGE_FAMILY"); fi
  gcloud compute instances create "$3" \
    --project "$GCP_PROJECT" --zone "$2" \
    --machine-type "$(kind_machine "$1")" \
    --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    "${source[@]}" \
    --boot-disk-size "${size}GB" ${disk[@]+"${disk[@]}"} \
    --tags "$GCP_TAGS" \
    --network-interface "nic-type=GVNIC,network=$GCP_NETWORK,subnet=$GCP_SUBNET" \
    --metadata-from-file "startup-script=$GCP_DIR/vm_startup.sh"
}
# True while 01_create_nodes.sh is running.
acquire_running() {
  local pid
  pid="$(cat "$GCP_STATE_DIR/acquire.pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}
# Runs `<script> <node>` for every RUNNING node at the same time. Each line of output is prefixed with the
# node, and a full log goes to $GCP_STATE_DIR/logs/<script>.<node>.log. Returns 1 if any of them failed.
on_all_nodes() {
  local script="$1" rows names n log i=0 rc=0 pids=()
  shift
  rows="$(nodes_rows)" || return 1
  names="$(awk '$4 == "RUNNING" {print $1}' <<<"$rows")"
  if [[ -z "$names" ]]; then echo "no node is running" >&2; return 1; fi
  mkdir -p "$GCP_STATE_DIR/logs"
  for n in $names; do
    log="$GCP_STATE_DIR/logs/$(basename "$script" .sh).$n.log"
    (
      set -o pipefail
      bash "$script" "$n" "$@" 2>&1 < /dev/null | tee "$log" | while IFS= read -r line; do echo "[${n#"$GCP_NODE_PREFIX"-}] $line"; done
    ) &
    pids+=($!)
  done
  for n in $names; do
    if wait "${pids[$i]}"; then echo "[${n#"$GCP_NODE_PREFIX"-}] done"; else echo "[${n#"$GCP_NODE_PREFIX"-}] FAILED"; rc=1; fi
    i=$((i + 1))
  done
  return $rc
}

gssh() { gcloud compute ssh "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" "$@"; }
