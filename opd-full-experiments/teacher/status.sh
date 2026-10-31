#!/usr/bin/env bash
# Run on your Mac. Is anything preempted, and how far along is each run?
#   bash skyrl-test/opd-full-experiments/teacher/status.sh           # one line per run
#   bash skyrl-test/opd-full-experiments/teacher/status.sh h100-1    # also the end of that run's log
# Exit code: 0 all fine, 1 something needs you, 2 could not tell. It only reads.
set -euo pipefail
TEACHER_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$TEACHER_DIR/config.sh"
source "$TEACHER_DIR/../gcp/config.sh"
STATE="${TEACHER_STATE_DIR:-$HOME/opd-full-experiments-teacher}"
STALL_SECS="${TEACHER_STALL_SECS:-1800}"   # a step and an eval take minutes, so half an hour of silence is not normal

rows="$(nodes_rows)" || exit 2
# Every preemption these VMs have had, even if the VM is back up.
ops="$(gcloud_read compute operations list --project "$GCP_PROJECT" \
  --filter="operationType=compute.instances.preempted AND targetLink~$GCP_NODE_PREFIX" \
  --format='value(targetLink.basename(),insertTime)')" || exit 2

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
while read -r node model; do
  [[ -n "$node" ]] || continue
  (
    GCP_VM="$GCP_NODE_PREFIX-$node"
    GCP_ZONE="$(awk -v n="$GCP_VM" '$1 == n {print $2}' <<<"$rows")"
    if [[ "$(awk -v n="$GCP_VM" '$1 == n {print $4}' <<<"$rows")" == RUNNING ]]; then
      gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command \
        "if [ -f ~/opd_node_setup.sh ]; then bash ~/opd_node_setup.sh report $(run_name "$model"); else echo not_set_up=1; fi" \
        > "$tmp/$node" 2>/dev/null < /dev/null || echo "no_answer=1" > "$tmp/$node"
    fi
  ) &
done <<<"$TEACHER_RUNS"
wait

field() { sed -n "s/^$1=//p" "$tmp/$2" 2>/dev/null | head -n 1; }
down=""; dead=""; check=""; finished=""; rc=0
# Written to a file and not piped into column: a pipeline would run this loop in a subshell and lose what it collects.
table="$tmp/table"
echo "NODE VM PREEMPTED STATE STEP ETA LOG_AGE GPU DISK_FREE RUN" > "$table"
while read -r node model; do
  [[ -n "$node" ]] || continue
  name="$GCP_NODE_PREFIX-$node"; run="$(run_name "$model")"
  vm="$(awk -v n="$name" '$1 == n {print $4}' <<<"$rows")"
  times="$(awk -v n="$name" '$1 == n {print $2}' <<<"$ops")"
  count=0
  if [[ -n "$times" ]]; then count="$(wc -l <<<"$times" | tr -d ' ')"; fi
  preempted="$count"
  if [[ $count != 0 ]]; then preempted="${count}x,last:$(sort <<<"$times" | tail -n 1 | cut -c6-16)"; fi
  step="-"; eta="-"; age="-"; gpu="-"; disk="-"
  if [[ -z "$vm" ]]; then
    state=NO-VM; vm="-"; down="$down $node"
  elif [[ "$vm" != RUNNING ]]; then
    if [[ $count != 0 ]]; then state=PREEMPTED; else state=STOPPED; fi
    down="$down $node"
  elif [[ -n "$(field no_answer "$node")" ]]; then
    state=NO-ANSWER; check="$check $node"
  elif [[ -n "$(field not_set_up "$node")" ]]; then
    state=NOT-STARTED; dead="$dead $node"
  else
    active="$(field active "$node")"; log_age="$(field log_age "$node")"
    step="$(field step "$node")"; step="${step:-$(field ckpt_step "$node")}"
    if [[ -n "$step" ]]; then step="$step/$TEACHER_STEPS"; else step="-"; fi
    eta="$(field eta "$node")"; eta="${eta:--}"
    age="-"; if [[ -n "$log_age" ]]; then age="$((log_age / 60))m"; fi
    gpu="$(field gpu "$node")"; gpu="${gpu:+$gpu%}"; gpu="${gpu:--}"
    disk="$(field disk "$node")"; disk="${disk:--}"
    if [[ -n "$(field uploaded "$node")" ]]; then state=DONE; eta="-"; finished="$finished $node"
    elif [[ -n "$(field done "$node")" && "$active" == 1 ]]; then state=UPLOADING
    elif [[ -n "$(field done "$node")" ]]; then state=UPLOAD-FAILED; dead="$dead $node"
    elif [[ "$active" == 1 && -n "$log_age" && "$log_age" -gt "$STALL_SECS" ]]; then state=STALLED; check="$check $node"
    elif [[ "$active" == 1 ]]; then state=TRAINING
    elif [[ "$(field ever_started "$node")" == 1 ]]; then state=RUN-DIED; dead="$dead $node"
    else state=NOT-STARTED; dead="$dead $node"
    fi
  fi
  echo "$node $vm $preempted $state $step $eta $age $gpu $disk $run" >> "$table"
done <<<"$TEACHER_RUNS"
column -t "$table"
echo
if [[ -e "$STATE/launch.pid" ]] && kill -0 "$(cat "$STATE/launch.pid")" 2>/dev/null; then
  echo "launch_all.sh is still running, so a node that is not going yet may just be mid-setup."
fi

if [[ -n "$down" || -n "$dead" ]]; then
  rc=1
  [[ -z "$down" ]] || echo "Down:$down"
  [[ -z "$dead" ]] || echo "Up, but the run is not going:$dead"
  echo "To bring them back, from the last checkpoint:"
  echo "  nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --yes > ~/opd_teachers.log 2>&1 < /dev/null &"
fi
if [[ -n "$check" ]]; then
  rc=1
  echo "Look at the log, the run has gone quiet or the VM is not answering:$check"
  echo "  bash skyrl-test/opd-full-experiments/teacher/status.sh <node>"
fi
if [[ -n "$finished" ]]; then
  echo "Finished, and still billing until stopped:$finished"
  for node in $finished; do echo "  bash skyrl-test/opd-full-experiments/gcp/down.sh stop $node"; done
fi
# Nodes with no run in config.sh, up and billing.
spare=""
for name in $(awk '$4 == "RUNNING" {print $1}' <<<"$rows"); do
  node="${name#"$GCP_NODE_PREFIX"-}"
  if ! awk -v n="$node" '$1 == n {found = 1} END {exit !found}' <<<"$TEACHER_RUNS"; then spare="$spare $node"; fi
done
[[ -z "$spare" ]] || echo "No run assigned, still billing:$spare"
echo "W&B: https://wandb.ai/sky-posttraining-uc-berkeley/$TEACHER_PROJECT"
echo "Hub: https://huggingface.co/$TEACHER_HF_REPO/tree/main"

if [[ -n "${1:-}" ]]; then
  model="$(awk -v n="$1" '$1 == n {print $2}' <<<"$TEACHER_RUNS")"
  [[ -n "$model" ]] || { echo "$1 has no run in config.sh" >&2; exit 2; }
  resolve_node "$1" || exit 2
  echo
  echo "--- end of the log on $1"
  gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command "tail -n 40 ~/opd-store/logs/$(run_name "$model").log" < /dev/null | tr '\r' '\n' | tail -n 40 | cut -c1-200
fi
exit $rc
