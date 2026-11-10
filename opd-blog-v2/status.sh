#!/usr/bin/env bash
# Run on your Mac. One line per run in config.sh: is its node up or preempted, and how far along is the run?
#   bash skyrl-test/opd-blog-v2/status.sh                                 # the table
#   bash skyrl-test/opd-blog-v2/status.sh p1_opd_m2rl_sft_from_rl_math    # also the end of that run's log
# Exit code: 0 all fine, 1 something needs you, 2 could not tell. It only reads.
set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$KIT_DIR/config.sh"
source "$KIT_DIR/gcp/config.sh"
STALL_SECS="${V2_STALL_SECS:-2700}"   # a step plus an eval at 32k tokens takes long, but 45 minutes of silence is not normal

rows="$(nodes_rows)" || exit 2
# Every preemption these VMs have had, even if the VM is back up.
ops="$(gcloud_read compute operations list --project "$GCP_PROJECT" \
  --filter="operationType=compute.instances.preempted AND targetLink~$GCP_NODE_PREFIX" \
  --format='value(targetLink.basename(),insertTime)')" || exit 2

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
while read -r node run script; do
  [[ -n "$node" ]] || continue
  (
    GCP_VM="$GCP_NODE_PREFIX-$node"
    GCP_ZONE="$(awk -v n="$GCP_VM" '$1 == n {print $2}' <<<"$rows")"
    if [[ "$(awk -v n="$GCP_VM" '$1 == n {print $4}' <<<"$rows")" == RUNNING ]]; then
      gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command \
        "if [ -f ~/opd_node_setup.sh ]; then bash ~/opd_node_setup.sh report $run $script; else echo not_set_up=1; fi" \
        > "$tmp/$run" 2>/dev/null < /dev/null || echo "no_answer=1" > "$tmp/$run"
    fi
  ) &
done <<<"$V2_RUNS"
wait

field() { sed -n "s/^$1=//p" "$tmp/$2" 2>/dev/null | head -n 1; }
novm=""; down=""; dead=""; check=""; finished=""
table="$tmp/table"
echo "RUN NODE VM PREEMPTED STATE STEP ETA LOG_AGE GPU DISK_FREE" > "$table"
while read -r node run script; do
  [[ -n "$node" ]] || continue
  name="$GCP_NODE_PREFIX-$node"
  vm="$(awk -v n="$name" '$1 == n {print $4}' <<<"$rows")"
  times="$(awk -v n="$name" '$1 == n {print $2}' <<<"$ops")"
  count=0
  if [[ -n "$times" ]]; then count="$(wc -l <<<"$times" | tr -d ' ')"; fi
  preempted="$count"
  if [[ $count != 0 ]]; then preempted="${count}x,last:$(sort <<<"$times" | tail -n 1 | cut -c6-16)"; fi
  step="-"; eta="-"; age="-"; gpu="-"; disk="-"
  if [[ -z "$vm" ]]; then
    state=NO-VM; vm="-"; novm="$novm $node"
  elif [[ "$vm" != RUNNING ]]; then
    if [[ $count != 0 ]]; then state=PREEMPTED; else state=STOPPED; fi
    down="$down $node"
  elif [[ -n "$(field no_answer "$run")" ]]; then
    state=NO-ANSWER; check="$check $run"
  elif [[ -n "$(field not_set_up "$run")" ]]; then
    state=NOT-SET-UP; down="$down $node"
  else
    active="$(field active "$run")"; log_age="$(field log_age "$run")"
    step="$(field step "$run")"; step="${step:-$(field ckpt_step "$run")}"
    total="$(field total "$run")"
    if [[ -n "$step" ]]; then step="$step/${total:-?}"; else step="-"; fi
    eta="$(field eta "$run")"; eta="${eta:--}"
    if [[ -n "$log_age" ]]; then age="$((log_age / 60))m"; fi
    gpu="$(field gpu "$run")"; gpu="${gpu:+$gpu%}"; gpu="${gpu:--}"
    disk="$(field disk "$run")"; disk="${disk:--}"
    if [[ -n "$(field done "$run")" ]]; then state=DONE; eta="-"; finished="$finished $run"
    elif [[ "$active" == 1 && -n "$log_age" && "$log_age" -gt "$STALL_SECS" ]]; then state=STALLED; check="$check $run"
    elif [[ "$active" == 1 ]]; then state=TRAINING
    elif [[ "$(field ever_started "$run")" == 1 ]]; then state=RUN-DIED; dead="$dead $run"
    else state=NOT-STARTED
    fi
  fi
  echo "$run $node $vm $preempted $state $step $eta $age $gpu $disk" >> "$table"
done <<<"$V2_RUNS"
column -t "$table"
echo

rc=0
if [[ -n "$novm" ]]; then
  rc=1
  echo "No VM yet:$(tr ' ' '\n' <<<"$novm" | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
  echo "  create them as gcp/README.md says (gcp/01_create_nodes.sh), then set them up (gcp/02_setup_nodes.sh)."
fi
if [[ -n "$down" ]]; then
  rc=1
  echo "Node down or not set up:$(tr ' ' '\n' <<<"$down" | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
  echo "  bring it back:  nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh --yes > ~/opd_v2_setup.log 2>&1 < /dev/null &"
  echo "  then start the same command again from the run's README; it resumes from the last checkpoint."
fi
if [[ -n "$dead" ]]; then
  rc=1
  echo "Run stopped without finishing:$dead"
  echo "  see why:  bash skyrl-test/opd-blog-v2/status.sh <run>   then start it again from its README; it resumes."
fi
if [[ -n "$check" ]]; then
  rc=1
  echo "Look at the log, the run has gone quiet or the VM is not answering:$check"
fi
[[ -z "$finished" ]] || echo "Finished:$finished"
echo "NOT-STARTED is expected for p1_opd_qwen3_8b_from_grpo until the GRPO run on its node is DONE."
echo "W&B: https://wandb.ai/sky-posttraining-uc-berkeley/$V2_PROJECT"

if [[ -n "${1:-}" ]]; then
  line="$(awk -v r="$1" '$2 == r' <<<"$V2_RUNS")"
  [[ -n "$line" ]] || { echo "no run called $1 in config.sh" >&2; exit 2; }
  read -r node run script <<<"$line"
  resolve_node "$node"
  echo
  echo "--- end of the log of $run on $node"
  gssh --quiet --ssh-flag="-o ConnectTimeout=20" --command "tail -n 40 ~/opd-store/logs/$run.log" < /dev/null 2>/dev/null | cut -c1-220 || true
fi
exit $rc
