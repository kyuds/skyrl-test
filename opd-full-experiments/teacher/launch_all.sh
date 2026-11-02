#!/usr/bin/env bash
# Run on your Mac. The one command that brings every teacher run up on its node, all at the same time. For each run in
# config.sh it starts the VM if it is stopped, sets the node up, copies your keys, starts Ray and starts the run under
# nohup on the VM. After that the run does not need this Mac.
#   bash skyrl-test/opd-full-experiments/teacher/launch_all.sh                       # the plan; changes nothing
#   nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --smoke --yes > ~/opd_teachers.log 2>&1 < /dev/null &
#   nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --yes > ~/opd_teachers.log 2>&1 < /dev/null &
#   ... --only h100-1,b200-2                                                         # just these nodes
#   ... --restart                                                                    # replace runs from an older recipe
#
# Safe to run again at any time, and that is how a preemption is handled: a run that is active or finished is left
# alone, and anything else picks up from its last checkpoint. --smoke runs 2 tiny steps on each node and waits for them.
# --restart is for a new recipe with new run names in config.sh: it stops an active run that is not under its node's
# current name, and leaves that run's checkpoints and exports on disk. A current run is still left alone.
set -euo pipefail
TEACHER_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$TEACHER_DIR/config.sh"
source "$TEACHER_DIR/../gcp/config.sh"

STATE="${TEACHER_STATE_DIR:-$HOME/opd-full-experiments-teacher}"   # on this Mac, outside the public repo
SSH_WAIT="${TEACHER_SSH_WAIT:-15}"           # between two tries while a VM boots
POLL="${TEACHER_POLL_SECS:-20}"              # between two looks at the software phase or a smoke test
REBOOT_WAIT="${TEACHER_REBOOT_WAIT:-45}"     # before the first ssh after a reboot
SOFTWARE_TRIES="${TEACHER_SOFTWARE_TRIES:-135}"   # 45 minutes at POLL
SMOKE_TRIES="${TEACHER_SMOKE_TRIES:-135}"         # 45 minutes at POLL

GO=0; SMOKE=""; ONLY=""; RESTART=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) GO=1; shift ;;
    --smoke) SMOKE="--smoke"; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --restart) RESTART=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1 (--help lists them)" >&2; exit 2 ;;
  esac
done
[[ -z "$SMOKE" || -z "$RESTART" ]] || { echo "--restart starts real runs, so it does not go with --smoke" >&2; exit 2; }

# The runs to bring up, as "<node> <model>" lines.
selected() {
  local node model
  while read -r node model; do
    [[ -n "$node" ]] || continue
    if [[ -z "$ONLY" || ",$ONLY," == *",$node,"* ]]; then echo "$node $model"; fi
  done <<<"$TEACHER_RUNS"
}
RUNS="$(selected)"
[[ -n "$RUNS" ]] || { echo "no run matches --only $ONLY. Nodes with a run: $(awk 'NF {print $1}' <<<"$TEACHER_RUNS" | tr '\n' ' ')" >&2; exit 2; }
rows="$(nodes_rows)" || exit 1

echo "project $TEACHER_PROJECT on W&B, kit $KIT_BRANCH, SkyRL ${SKYRL_COMMIT:0:8}"
if [[ -n "$SMOKE" ]]; then echo "smoke test: 2 tiny steps on each node, nothing uploaded"; else echo "$TEACHER_STEPS steps each, then uploaded to $TEACHER_HF_REPO"; fi
if [[ -n "$RESTART" ]]; then echo "restart: an active run under any other name is stopped. Its checkpoints and exports stay on disk"; fi
while read -r node model; do
  printf '  %-7s %-11s %-24s %s\n' "$node" "$(awk -v n="$GCP_NODE_PREFIX-$node" '$1 == n {print $4}' <<<"$rows")" "$model" "$(run_name "$model")"
done <<<"$RUNS"
if [[ $GO == 0 ]]; then
  echo "Nothing was changed. Add --yes to go."
  exit 0
fi

# ---- checks that are cheaper to fail now than on six nodes ----
: "${WANDB_API_KEY:?export WANDB_API_KEY in this shell first}"
: "${HF_TOKEN:?export HF_TOKEN (a Hugging Face write token) in this shell first}"
# The token goes in through a config on stdin, so it is never on a command line.
if ! printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | curl -sf -K - -o /dev/null https://huggingface.co/api/whoami-v2; then
  echo "the Hub does not accept HF_TOKEN: check the token before training for days" >&2; exit 1
fi
# The VMs clone the kit from GitHub, so what runs there is what is pushed, not what is on this Mac.
kit_head="$(git -C "$TEACHER_DIR" rev-parse HEAD 2>/dev/null || true)"
pushed="$(git ls-remote "$KIT_REPO" "refs/heads/$KIT_BRANCH" 2>/dev/null | awk '{print $1}')"
if [[ -z "$pushed" || "$kit_head" != "$pushed" || -n "$(git -C "$TEACHER_DIR/.." status --porcelain -- . 2>/dev/null)" ]]; then
  echo "this kit is not what $KIT_BRANCH on GitHub holds, and the VMs read it from there. Commit and push first." >&2; exit 1
fi
if acquire_running; then echo "01_create_nodes.sh is still running: wait for it, or touch $GCP_STATE_DIR/stop" >&2; exit 1; fi
mkdir -p "$STATE/logs"
if [[ -e "$STATE/launch.pid" ]] && kill -0 "$(cat "$STATE/launch.pid")" 2>/dev/null; then
  echo "another launch_all.sh is running (pid $(cat "$STATE/launch.pid"))" >&2; exit 1
fi
echo $$ > "$STATE/launch.pid"
pids=()
trap 'rm -f "$STATE/launch.pid"' EXIT
trap 'kill ${pids[@]+"${pids[@]}"} 2>/dev/null; exit 130' INT TERM

# ---- one worker per node. Each runs in its own subshell, so the GCP_VM it sets is its own ----
nssh() { gssh --quiet --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=30" --ssh-flag="-o ServerAliveCountMax=4" "$@"; }
node_phase() { nssh --command "bash ~/opd_node_setup.sh $*"; }
finish() { echo "$2" > "$STATE/result.$1"; echo "$2"; }   # $1 = node, $2 = its one-line outcome
field() { sed -n "s/^$1=//p" <<<"$2" | head -n 1; }

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

wait_smoke() {   # $1 = the smoke run's name
  local i rep
  for i in $(seq 1 "$SMOKE_TRIES"); do
    sleep "$POLL"
    rep="$(node_phase report "$1" 2>/dev/null || true)"
    if [[ "$(field active "$rep")" == 0 ]]; then
      [[ -n "$(field smoke_ok "$rep")" ]]
      return
    fi
  done
  return 1
}

bring_up() {
  local node="$1" model="$2" run rep
  run="$(run_name "$model")"
  [[ -z "$SMOKE" ]] || run="smoke_$run"
  rm -f "$STATE/result.$node"
  vm_up "$node" || { finish "$node" "FAILED: the VM did not come up"; return 1; }
  gcloud compute scp "$TEACHER_DIR/node_setup.sh" "$GCP_VM:~/opd_node_setup.sh" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --quiet >/dev/null ||
    { finish "$node" "FAILED: could not copy node_setup.sh"; return 1; }

  rep="$(node_phase report "$run" 2>/dev/null || true)"
  # Active, with no log under this name: the run is from an older recipe.
  if [[ -n "$RESTART" && "$(field active "$rep")" == 1 && "$(field ever_started "$rep")" == 0 ]]; then
    echo "--- stopping the run from the older recipe"
    node_phase stop-run || { finish "$node" "FAILED: could not stop the old run. Log: $STATE/logs/$node.log"; return 1; }
    rep="$(node_phase report "$run" 2>/dev/null || true)"
  fi
  if [[ "$(field active "$rep")" == 1 ]]; then
    if [[ -n "$SMOKE" ]]; then finish "$node" "FAILED: a run is active, so no smoke test here"; return 1; fi
    finish "$node" "already running, step $(field step "$rep")"; return 0
  fi
  if [[ -z "$SMOKE" && -n "$(field uploaded "$rep")" ]]; then finish "$node" "finished and uploaded"; return 0; fi

  setup_node "$node" || { finish "$node" "FAILED: setup. Log: $STATE/logs/$node.log"; return 1; }
  echo "--- train"
  node_phase train "$(printf %q "$model")" "$(printf %q "$run")" $SMOKE ||
    { finish "$node" "FAILED: the run did not start. Log: $STATE/logs/$node.log"; return 1; }
  if [[ -z "$SMOKE" ]]; then finish "$node" "running: $run"; return 0; fi
  if wait_smoke "$run"; then finish "$node" "smoke test passed"; return 0; fi
  nssh --command "tail -n 30 ~/opd-store/logs/$run.log" 2>/dev/null | cut -c1-200 || true
  finish "$node" "FAILED: smoke test. Log on the VM: ~/opd-store/logs/$run.log"
  return 1
}

nodes=()
while read -r node model; do
  rm -f "$STATE/result.$node"
  (
    set +e +o pipefail
    bring_up "$node" "$model" 2>&1 < /dev/null | tee "$STATE/logs/$node.log" |
      while IFS= read -r line; do echo "[$node] $line"; done
  ) &
  pids+=($!)
  nodes+=("$node")
done <<<"$RUNS"
wait

echo
echo "==== result ===="
rc=0
for node in "${nodes[@]}"; do
  outcome="$(cat "$STATE/result.$node" 2>/dev/null || echo 'FAILED: no outcome recorded')"
  printf '  %-7s %s\n' "$node" "$outcome"
  if [[ "$outcome" == FAILED* ]]; then rc=1; fi
done
echo
if [[ $rc != 0 ]]; then echo "Fix what failed and run this again: it picks up where it left off."; fi
if [[ -z "$SMOKE" ]]; then echo "Check on them with: bash skyrl-test/opd-full-experiments/teacher/status.sh"; fi
exit $rc
