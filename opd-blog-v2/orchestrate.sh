#!/usr/bin/env bash
# Run on your Mac. Starts the jobs of config.sh's table on their nodes, stage by stage, and shows where they are.
# It reads the nodes over ssh and starts things only with --yes. See README.md, "Running it".
#   bash skyrl-test/opd-blog-v2/orchestrate.sh                      # every job: state, progress, what can start next
#   bash skyrl-test/opd-blog-v2/orchestrate.sh run 1                # the plan for stage 1; changes nothing
#   bash skyrl-test/opd-blog-v2/orchestrate.sh run 1 --yes          # start it
#   bash skyrl-test/opd-blog-v2/orchestrate.sh run 2 --only h100-3 --yes
#   bash skyrl-test/opd-blog-v2/orchestrate.sh log <job> [lines]    # the end of a job's log
#   bash skyrl-test/opd-blog-v2/orchestrate.sh stop <node>          # stop the node's jobs; checkpoints and outputs stay
#
# `run <stage>` starts, on each node of the stage, its jobs that are not done yet, in table order, as one chain
# (common/chain.sh): a job starts when the one before it finishes, and a failure stops the chain. It skips a node
# that is down, has no Ray, runs another SkyRL commit, or is busy with a chain, and it cuts a node's list at the
# first job whose deps (on any node) are not done yet: run the same stage again later to start the rest. Before
# starting, it moves the node's kit checkout to what is pushed to GitHub (local edits there are discarded).
set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_DIR/config.sh"
source "$KIT_DIR/gcp/config.sh"
TAB="$(printf '\t')"
CACHE="$GCP_STATE_DIR/jobs_done"   # jobs seen done; used only while their node cannot be read
mkdir -p "$CACHE"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
v2_jobs > "$TMP/table"

cmd="${1:-status}"; [[ $# -gt 0 ]] && shift
GO=0; ONLY=""; STAGE=""
case "$cmd" in
  status|run)
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --yes) GO=1; shift ;;
        --only) ONLY=",${2:-},"; shift 2 ;;
        [0-9]*) STAGE="$1"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done ;;
  log|stop) ;;
  -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
  *) echo "unknown command: $cmd (status, run, log, stop; --help)" >&2; exit 2 ;;
esac

table_nodes() { awk '{print $2}' "$TMP/table" | awk '!seen[$0]++'; }
node_of() { awk -v j="$1" '$3 == j {print $2}' "$TMP/table"; }

# Reads every node of the table that is running, in parallel: $TMP/vm has "<node> <status> <zone>", $TMP/r.<node> the
# node's job_report.sh lines.
sweep() {
  local rows node vm status zone jobs
  rows="$(nodes_rows)" || exit 2
  : > "$TMP/vm"
  for node in $(table_nodes); do
    vm="$GCP_NODE_PREFIX-$node"
    status="$(awk -v n="$vm" '$1 == n {print $4}' <<<"$rows")"
    zone="$(awk -v n="$vm" '$1 == n {print $2}' <<<"$rows")"
    echo "$node ${status:-NO-VM} ${zone:--}" >> "$TMP/vm"
    if [[ "$status" == RUNNING ]]; then
      jobs="$(awk -v n="$node" '$2 == n {print $3}' "$TMP/table" | tr '\n' ' ')"
      (
        GCP_VM="$vm"; GCP_ZONE="$zone"
        gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command "bash -s -- $jobs" \
          < "$KIT_DIR/common/job_report.sh" > "$TMP/r.$node" 2>/dev/null || printf 'NODE\tnoanswer\n' > "$TMP/r.$node"
      ) &
    fi
  done
  wait
  for node in $(table_nodes); do
    [[ -f "$TMP/r.$node" ]] || continue
    awk -F"$TAB" '$1 != "NODE" {print $1, $2}' "$TMP/r.$node" | while read -r job state; do
      if [[ "$state" == done ]]; then touch "$CACHE/$job"; else rm -f "$CACHE/$job"; fi
    done
  done
}
vm_status() { awk -v n="$1" '$1 == n {print $2}' "$TMP/vm"; }
# The report of a node that was not read is empty, so these print nothing for it.
report() { cat "$TMP/r.$1" 2>/dev/null || true; }
node_field() { report "$1" | awk -F"$TAB" -v k="$2" '$1 == "NODE" {for (i = 2; i <= NF; i++) if (index($i, k "=") == 1) print substr($i, length(k) + 2)}'; }
# A job's state: from its node's report, or, when the node cannot be read, "done" if it was seen done, else the node's.
job_state() {
  local node s
  node="$(node_of "$1")"
  s="$(report "$node" | awk -F"$TAB" -v j="$1" '$1 == j {print $2}')"
  if [[ -z "$s" ]]; then
    if [[ -f "$CACHE/$1" ]]; then s=done
    elif grep -qs noanswer "$TMP/r.$node"; then s=no-answer
    else s="node-$(vm_status "$node" | tr '[:upper:]' '[:lower:]')"
    fi
  fi
  echo "$s"
}
job_field() { report "$(node_of "$1")" | awk -F"$TAB" -v j="$1" -v c="$2" '$1 == j {print $c}'; }

# For one node and stage: the jobs to start now, in order, and why the rest wait. Sets START and WAITS.
plan_node() {
  local node="$1" stage="$2" job dep st missing
  START=""; WAITS=""
  for job in $(awk -v n="$node" -v s="$stage" '$1 == s && $2 == n {print $3}' "$TMP/table"); do
    [[ "$(job_state "$job")" == done ]] && continue
    missing=""
    for dep in $(v2_job_deps "$job"); do
      st="$(job_state "$dep")"
      if [[ "$st" != done && " $START " != *" $dep "* ]]; then missing="$missing $dep($st)"; fi
    done
    if [[ -n "$missing" ]]; then WAITS="$job waits on:$missing"; return 0; fi
    START="$START $job"
  done
}

status() {
  local node job st age prog line
  sweep
  {
    echo "STAGE${TAB}NODE${TAB}JOB${TAB}STATE${TAB}LOG_AGE${TAB}PROGRESS"
    while read -r stage node job _; do
      [[ -z "$STAGE" || "$stage" == "$STAGE" ]] || continue
      st="$(job_state "$job")"; age="$(job_field "$job" 3)"; prog="$(job_field "$job" 4)"
      if [[ -n "$age" ]]; then age="$((age / 60))m"; else age="-"; fi
      echo "$stage$TAB$node$TAB$job$TAB$st$TAB$age$TAB${prog:--}"
    done < "$TMP/table"
  } | column -t -s "$TAB"
  echo
  for node in $(table_nodes); do
    line="$(vm_status "$node")"
    if [[ -f "$TMP/r.$node" ]]; then
      line="$line ray=$(node_field "$node" ray) gpu=$(node_field "$node" gpu)% disk=$(node_field "$node" disk) chain=$(node_field "$node" chain)"
    fi
    echo "  $node: $line"
  done
  echo
  echo "Can start now (orchestrate.sh run <stage> --only <node> --yes):"
  local any=0 stage
  for stage in $(awk '{print $1}' "$TMP/table" | sort -un); do
    for node in $(awk -v s="$stage" '$1 == s {print $2}' "$TMP/table" | awk '!seen[$0]++'); do
      [[ "$(vm_status "$node")" == RUNNING && "$(node_field "$node" chain)" == none ]] || continue
      plan_node "$node" "$stage"
      if [[ -n "$START" ]]; then echo "  stage $stage on $node:$START"; any=1; fi
    done
  done
  [[ $any == 1 ]] || echo "  nothing"
  echo "W&B: https://wandb.ai/sky-posttraining-uc-berkeley/$V2_PROJECT  (evals: $V2_PROJECT-evals)"
}

run_stage() {
  [[ -n "$STAGE" ]] || { echo "which stage? orchestrate.sh run <stage>" >&2; exit 2; }
  awk -v s="$STAGE" '$1 == s' "$TMP/table" | grep -q . || { echo "stage $STAGE has no jobs in config.sh" >&2; exit 2; }
  local kit_head pushed node status chain skyrl todo=""
  # The nodes run what is on GitHub, so that has to be what is here.
  kit_head="$(git -C "$KIT_DIR" rev-parse HEAD 2>/dev/null || true)"
  pushed="$(git ls-remote "$KIT_REPO" "refs/heads/$KIT_BRANCH" 2>/dev/null | awk '{print $1}')"
  if [[ -z "$pushed" || "$kit_head" != "$pushed" || -n "$(git -C "$KIT_DIR" status --porcelain -- . 2>/dev/null)" ]]; then
    if [[ "${V2_ALLOW_UNPUSHED:-0}" != 1 ]]; then   # 1 only to test this script itself
      echo "this kit is not what $KIT_BRANCH on GitHub holds, and the nodes run that. Commit and push first." >&2; exit 1
    fi
    echo "warning: this kit is not what is pushed; the nodes will run what is pushed (V2_ALLOW_UNPUSHED=1)" >&2
  fi
  sweep
  echo "stage $STAGE (kit ${kit_head:0:8}, SkyRL ${SKYRL_COMMIT:0:8}):"
  for node in $(awk -v s="$STAGE" '$1 == s {print $2}' "$TMP/table" | awk '!seen[$0]++'); do
    [[ -z "$ONLY" || "$ONLY" == *",$node,"* ]] || continue
    status="$(vm_status "$node")"
    if [[ "$status" != RUNNING ]]; then echo "  $node: skipped, the VM is $status"; continue; fi
    if grep -qs noanswer "$TMP/r.$node"; then echo "  $node: skipped, it does not answer ssh"; continue; fi
    chain="$(node_field "$node" chain)"; skyrl="$(node_field "$node" skyrl)"
    if [[ "$(node_field "$node" ray)" != 1 ]]; then echo "  $node: skipped, Ray is not up (gcp/02_setup_nodes.sh --only $node)"; continue; fi
    if [[ "$skyrl" != "$SKYRL_COMMIT" ]]; then echo "  $node: skipped, SkyRL is at ${skyrl:0:8}, not ${SKYRL_COMMIT:0:8} (gcp/02_setup_nodes.sh --only $node)"; continue; fi
    if [[ "$chain" != none ]]; then echo "  $node: skipped, busy with $chain"; continue; fi
    plan_node "$node" "$STAGE"
    if [[ -z "$START" && -z "$WAITS" ]]; then echo "  $node: every job of stage $STAGE is done"; continue; fi
    if [[ -n "$START" ]]; then echo "  $node: start$START"; todo="$todo $node:$(tr ' ' ',' <<<"${START# }")"; fi
    if [[ -n "$WAITS" ]]; then
      echo "  $node: $([[ -n "$START" ]] && echo "after those, ")$WAITS (run stage $STAGE again once it is done)"
    fi
  done
  [[ -n "$todo" ]] || { echo "Nothing to start."; return 0; }
  if [[ $GO == 0 ]]; then echo "Nothing was started. Add --yes to start it."; return 0; fi
  local item jobs zone
  for item in $todo; do
    node="${item%%:*}"; jobs="$(tr ',' ' ' <<<"${item#*:}")"
    zone="$(awk -v n="$node" '$1 == n {print $3}' "$TMP/vm")"
    (
      GCP_VM="$GCP_NODE_PREFIX-$node"; GCP_ZONE="$zone"
      gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command "set -e
        git -C ~/SkyRL/skyrl-test fetch -q origin $KIT_BRANCH
        git -C ~/SkyRL/skyrl-test checkout -q -f -B $KIT_BRANCH FETCH_HEAD
        mkdir -p ~/opd-store/logs
        cd ~/SkyRL
        setsid nohup bash skyrl-test/opd-blog-v2/common/chain.sh $jobs >> ~/opd-store/logs/chain.log 2>&1 < /dev/null &
        sleep 3
        echo \"started (kit \$(git -C ~/SkyRL/skyrl-test rev-parse --short HEAD)): \$(cat ~/opd-store/jobs/chain.jobs 2>/dev/null)\"" \
        < /dev/null 2>&1 | sed "s/^/  $node: /"
    ) || echo "  $node: could not start its jobs (see above); the other nodes go on"
  done
  echo "Follow them with: bash skyrl-test/opd-blog-v2/orchestrate.sh"
}

case "$cmd" in
  status) status ;;
  run) run_stage ;;
  log)
    job="${1:?orchestrate.sh log <job> [lines]}"; node="$(node_of "$job")"
    [[ -n "$node" ]] || { echo "no job called $job in config.sh" >&2; exit 2; }
    resolve_node "$node"
    gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command "tail -n ${2:-60} ~/opd-store/logs/$job.log" < /dev/null | cut -c1-240 ;;
  stop)
    node="${1:?orchestrate.sh stop <node>}"
    resolve_node "$node"
    # The chain runs in its own session (setsid), so its process group holds the job, the run script and the trainer;
    # Ray ends the trainer's actors and vLLM engines when the trainer goes. The jobs are left "interrupted".
    gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command '
      pid="$(cat ~/opd-store/jobs/chain.pid 2>/dev/null || true)"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "no chain is running"; exit 0; fi
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
      for i in $(seq 1 60); do
        if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then echo "stopped; the GPUs are free"; exit 0; fi
        sleep 3
      done
      echo "stopped the chain, but the GPUs are still busy:"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader' < /dev/null ;;
esac
