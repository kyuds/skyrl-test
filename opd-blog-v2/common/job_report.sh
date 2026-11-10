#!/usr/bin/env bash
# Runs on the VM, piped there by orchestrate.sh (`bash -s -- <job> ...` with this file on stdin), so it needs nothing
# from the node's checkout and works on a node whose kit is out of date. It only reads. Prints tab-separated lines:
#   <job>  <state>  <seconds since its log last changed>  <progress>
#     state: none (never started), running, done, failed, interrupted (started, then stopped without an exit code:
#     a preemption, a reboot or orchestrate.sh stop)
#   NODE  ray=<0|1>  chain=<its jobs, or none>  skyrl=<commit>  kit=<commit>  gpu=<mean util %>  disk=<free on ~>
STORE="$HOME/opd-store"; J="$STORE/jobs"; L="$STORE/logs"
alive() {   # $1 pid file, $2 what its command line must contain
  local pid
  pid="$(cat "$1" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$2"
}
for job in "$@"; do
  state=none
  if [[ -f "$J/$job.done" ]]; then state=done
  elif alive "$J/$job.pid" "job.sh $job"; then state=running
  elif [[ -f "$J/$job.failed" ]]; then state=failed
  elif [[ -f "$J/$job.started" ]]; then state=interrupted
  fi
  log="$L/$job.log"; age=""; progress=""
  if [[ -f "$log" ]]; then age=$(( $(date +%s) - $(stat -c %Y "$log") )); fi
  # OPD: SkyRL's progress bar, "Training Batches Processed:  41%|####  | 62/150 [5:36:00<7:58:00, 326.0s/it]"
  bar="$(tail -c 300000 "$log" 2>/dev/null | tr '\r' '\n' | grep -a 'Training Batches Processed' | tail -n 1 || true)"
  if [[ -n "$bar" ]]; then
    progress="$(sed -n 's/.*| *\([0-9][0-9]*\/[0-9][0-9]*\) *\[[^<]*<\([^,]*\),.*/step \1, eta \2/p' <<<"$bar")"
  elif grep -aq 'Starting SFT training for' "$log" 2>/dev/null; then
    total="$(grep -a -o 'Starting SFT training for [0-9]* steps' "$log" | tail -n 1 | grep -o '[0-9][0-9]*')"
    # SkyRL logs "Step N: loss=..., grad_norm=..." every few steps; checkpoints come every SFT_CKPT_EVERY steps.
    step="$(grep -a -o 'Step [0-9]*: loss=[0-9.]*' "$log" | tail -n 1)"
    progress="step ${step:-0/}"
    if [[ -n "$step" ]]; then progress="step ${step%%:*}/$total, ${step#*: }"; progress="${progress/step Step /step }"; fi
    progress="$progress, checkpoint $(cat "$STORE/ckpts/$job/latest_ckpt_global_step.txt" 2>/dev/null || echo 0)"
  elif [[ -d "$STORE/evals/$job" ]]; then
    d="$STORE/evals/$job"
    if [[ -f "$d/scores.json" ]]; then
      progress="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(" ".join("%s=%s" % (b, v["score"]) for b, v in s.items()))' "$d/scores.json" 2>/dev/null)"
    elif [[ -f "$d/responses.jsonl" ]]; then progress="grading"
    elif [[ -f "$d/requests.jsonl" ]]; then progress="sampling, $(ls "$d" | grep -c '^responses.jsonl.shard[0-9]*$') GPUs done"
    fi
  fi
  printf '%s\t%s\t%s\t%s\n' "$job" "$state" "$age" "$progress"
done
ray=0; if pgrep -x raylet >/dev/null; then ray=1; fi
chain=none; if alive "$J/chain.pid" chain.sh; then chain="$(tr ' ' ',' < "$J/chain.jobs")"; fi
gpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s += $1} END {if (NR) printf "%d", s / NR}')"
printf 'NODE\tray=%s\tchain=%s\tskyrl=%s\tkit=%s\tgpu=%s\tdisk=%s\n' "$ray" "$chain" \
  "$(git -C "$HOME/SkyRL" rev-parse HEAD 2>/dev/null)" "$(git -C "$HOME/SkyRL/skyrl-test" rev-parse --short HEAD 2>/dev/null)" \
  "$gpu" "$(df -BG --output=avail "$HOME" 2>/dev/null | tail -n 1 | tr -d ' ')"
