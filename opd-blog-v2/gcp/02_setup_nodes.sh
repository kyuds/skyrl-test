#!/usr/bin/env bash
# Run on your Mac. Sets up every node that has a job in ../config.sh, all at the same time: starts the VM if it is
# stopped, checks the driver, builds the NVMe array, installs SkyRL at the pinned commit with this kit inside it,
# copies your keys and starts Ray. It starts no job: orchestrate.sh does (README.md, "Running it"). Safe to run again at any time,
# and it is the first step after a preemption (the NVMe array and Ray do not survive one).
#   bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh                     # the plan; changes nothing
#   nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh --yes > ~/opd_v2_setup.log 2>&1 < /dev/null &
#   ... --only b200-1,h100-3                                              # just these nodes, with a job or not
set -euo pipefail
GCP_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_DIR="$(cd "$GCP_DIR/.." && pwd)"
source "$KIT_DIR/config.sh"
source "$GCP_DIR/config.sh"

STATE="$GCP_STATE_DIR"                       # on this Mac, outside the public repo
SSH_WAIT="${V2_SSH_WAIT:-15}"                # between two tries while a VM boots
POLL="${V2_POLL_SECS:-20}"                   # between two looks at the software phase
REBOOT_WAIT="${V2_REBOOT_WAIT:-45}"          # before the first ssh after a reboot
SOFTWARE_TRIES="${V2_SOFTWARE_TRIES:-135}"   # 45 minutes at POLL

GO=0; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) GO=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1 (--help lists them)" >&2; exit 2 ;;
  esac
done

# The nodes with a job, once each, in config.sh order (V2_NODE_MAP applied); or exactly the ones --only names, which
# may have no job yet (a spare node).
if [[ -n "$ONLY" ]]; then
  NODES="$(tr ',' '\n' <<<"$ONLY" | awk 'NF && !seen[$0]++')"
  bad="$(grep -v -E '^(h100|b200)-[0-9]+$' <<<"$NODES" || true)"
  [[ -z "$bad" ]] || { echo "not a node name: $(tr '\n' ' ' <<<"$bad")(names look like h100-3 or b200-1)" >&2; exit 2; }
else
  NODES="$(v2_jobs | awk '{print $2}' | awk '!seen[$0]++')"
fi
rows="$(nodes_rows)" || exit 1

echo "kit $KIT_BRANCH, SkyRL ${SKYRL_COMMIT:0:8}"
for node in $NODES; do
  vm="$(awk -v n="$GCP_NODE_PREFIX-$node" '$1 == n {print $4}' <<<"$rows")"
  jobs="$(v2_jobs | awk -v n="$node" '$2 == n' | wc -l | tr -d ' ')"
  printf '  %-7s %-11s %s\n' "$node" "${vm:-NO-VM}" "$([[ $jobs == 0 ]] && echo '(no job in config.sh)' || echo "$jobs jobs")"
done
if [[ $GO == 0 ]]; then
  echo "Nothing was changed. Add --yes to go."
  exit 0
fi

# ---- checks that are cheaper to fail now than on four nodes ----
: "${WANDB_API_KEY:?export WANDB_API_KEY in this shell first}"
: "${HF_TOKEN:?export HF_TOKEN (a Hugging Face write token) in this shell first}"
# The token goes in through a config on stdin, so it is never on a command line.
if ! printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | curl -sf -K - -o /dev/null https://huggingface.co/api/whoami-v2; then
  echo "the Hub does not accept HF_TOKEN" >&2; exit 1
fi
# The VMs clone the kit from GitHub, so what runs there is what is pushed, not what is on this Mac.
kit_head="$(git -C "$KIT_DIR" rev-parse HEAD 2>/dev/null || true)"
pushed="$(git ls-remote "$KIT_REPO" "refs/heads/$KIT_BRANCH" 2>/dev/null | awk '{print $1}')"
if [[ -z "$pushed" || "$kit_head" != "$pushed" || -n "$(git -C "$KIT_DIR" status --porcelain -- . 2>/dev/null)" ]]; then
  echo "this kit is not what $KIT_BRANCH on GitHub holds, and the VMs read it from there. Commit and push first." >&2; exit 1
fi
# Setting up the nodes already up while 01_create_nodes.sh asks for the rest saves their billed idle time. Both may
# start a stopped VM; if they collide, this script reports the node FAILED and a second run fixes it.
if acquire_running; then echo "note: 01_create_nodes.sh is still asking for nodes; setting up the ones named here anyway"; fi
mkdir -p "$STATE/logs"
if [[ -e "$STATE/setup.pid" ]] && kill -0 "$(cat "$STATE/setup.pid")" 2>/dev/null; then
  echo "another 02_setup_nodes.sh is running (pid $(cat "$STATE/setup.pid"))" >&2; exit 1
fi
echo $$ > "$STATE/setup.pid"
pids=()
trap 'rm -f "$STATE/setup.pid"' EXIT
trap 'kill ${pids[@]+"${pids[@]}"} 2>/dev/null; exit 130' INT TERM

# ---- one worker per node. Each runs in its own subshell, so the GCP_VM it sets is its own ----
nssh() { gssh --quiet --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=30" --ssh-flag="-o ServerAliveCountMax=4" "$@"; }
node_phase() { nssh --command "bash ~/opd_node_setup.sh $*"; }
finish() { echo "$2" > "$STATE/setup_result.$1"; echo "$2"; }   # $1 = node, $2 = its one-line outcome

# Ends with the VM RUNNING and answering ssh.
vm_up() {
  local i
  resolve_node "$1" || return 1
  if [[ "$GCP_VM_STATUS" == TERMINATED || "$GCP_VM_STATUS" == STOPPED || "$GCP_VM_STATUS" == SUSPENDED ]]; then
    echo "$GCP_VM is $GCP_VM_STATUS: starting it in $GCP_ZONE"
    if ! gcloud compute instances start "$GCP_VM" --project "$GCP_PROJECT" --zone "$GCP_ZONE" 2>&1 | tail -n 4; then
      echo "could not start $GCP_VM. For spot this is usually no capacity right now."
      return 1
    fi
  fi
  for i in $(seq 1 40); do
    resolve_node "$1" || return 1
    if [[ "$GCP_VM_STATUS" == RUNNING ]] && nssh --command true >/dev/null 2>&1; then return 0; fi
    sleep "$SSH_WAIT"
  done
  echo "$GCP_VM did not answer ssh in 10 minutes (it is $GCP_VM_STATUS)"
  return 1
}

setup_node() {
  local out i
  vm_up "$1" || return 1
  gcloud compute scp "$GCP_DIR/node_setup.sh" "$GCP_VM:~/opd_node_setup.sh" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --quiet >/dev/null ||
    { echo "could not copy node_setup.sh"; return 1; }

  echo "--- driver"
  out="$(node_phase driver)" || { echo "$out"; return 1; }
  echo "$out"
  if grep -q REBOOT_NEEDED <<<"$out"; then
    echo "rebooting (about 5 minutes)"
    nssh --command 'sudo reboot' >/dev/null 2>&1 || true
    sleep "$REBOOT_WAIT"
    vm_up "$1" || return 1
    out="$(node_phase driver)" || { echo "$out"; return 1; }
    if grep -q REBOOT_NEEDED <<<"$out"; then
      echo "$out"; echo "the driver or the open-file limit is still not right after a reboot"; return 1
    fi
  fi

  echo "--- storage"
  node_phase storage || return 1

  # Detached on the VM, and polled from here, so a dropped connection only costs this script its view of it.
  echo "--- software (10-20 minutes on a fresh node)"
  nssh --command "$(printf 'SKYRL_REPO=%q SKYRL_COMMIT=%q KIT_REPO=%q KIT_BRANCH=%q' "$SKYRL_REPO" "$SKYRL_COMMIT" "$KIT_REPO" "$KIT_BRANCH") bash ~/opd_node_setup.sh software-bg" || return 1
  for i in $(seq 1 "$SOFTWARE_TRIES"); do
    sleep "$POLL"
    out="$(node_phase software-status 2>/dev/null || echo SSH_FAILED)"
    case "$(head -n 1 <<<"$out")" in
      OK) echo "$out" | tail -n +2; break ;;
      FAILED) echo "$out" | tail -n +2; echo "the software phase failed"; return 1 ;;
      RUNNING) if (( i % 6 == 0 )); then echo "  $(sed -n 2p <<<"$out")"; fi ;;
      *) resolve_node "$1" || return 1
         if [[ "$GCP_VM_STATUS" != RUNNING ]]; then echo "$GCP_VM went $GCP_VM_STATUS during setup"; return 1; fi ;;
    esac
    if [[ $i == "$SOFTWARE_TRIES" ]]; then echo "the software phase did not finish in time"; return 1; fi
  done

  # The values travel over ssh's stdin: never on a command line, never printed, and never in this repo.
  printf 'export WANDB_API_KEY=%q\nexport HF_TOKEN=%q\n' "$WANDB_API_KEY" "$HF_TOKEN" |
    nssh --command 'umask 077; cat > ~/.opd_secrets; chmod 600 ~/.opd_secrets' || return 1

  echo "--- ray"
  node_phase ray || return 1
}

for node in $NODES; do
  (
    rm -f "$STATE/setup_result.$node"
    if setup_node "$node"; then finish "$node" "ready"; else finish "$node" "FAILED. Log: $STATE/logs/setup.$node.log"; exit 1; fi
  ) 2>&1 < /dev/null | tee "$STATE/logs/setup.$node.log" | while IFS= read -r line; do echo "[$node] $line"; done &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done

echo
echo "=== setup"
rc=0
for node in $NODES; do
  result="$(cat "$STATE/setup_result.$node" 2>/dev/null || echo "FAILED: no result")"
  printf '  %-7s %s\n' "$node" "$result"
  [[ "$result" == ready ]] || rc=1
done
echo "Start the jobs with: bash skyrl-test/opd-blog-v2/orchestrate.sh run <stage>   (README.md, \"Running it\")"
exit $rc
