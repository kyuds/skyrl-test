#!/usr/bin/env bash
# Run on your Mac. Where an eval run on the VM stands: running, finished, or stopped early, with how many rollouts
# are done and accepted and the log's last line. --wait checks every 60 s (or N) and returns when the run ends,
# with a Mac notification.
#   bash skyrl-test/skycap-codex-eval/status.sh cc16x4
#   bash skyrl-test/skycap-codex-eval/status.sh cc16x4 --wait          # until it ends
#   bash skyrl-test/skycap-codex-eval/status.sh                        # the newest run
# Exit code: 0 finished, 1 stopped before every rollout was done, 2 still running (without --wait).
# The check itself runs on the VM, piped from here over ssh, so it needs nothing pulled there.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../skycap-responses/gcp/config.sh"
need_vm
run="latest"; WAIT=0; EVERY=60
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) WAIT=1; if [[ "${2:-}" =~ ^[0-9]+$ ]]; then EVERY="$2"; shift; fi ;;
    -*) echo "usage: $0 [run] [--wait [seconds]]" >&2; exit 2 ;;
    *) run="$1" ;;
  esac
  shift
done

# Runs on the VM: prints one line, exits 0 (finished), 1 (stopped early), 2 (running) or 4 (no such run).
read -r -d '' REMOTE <<'EOF' || true
set -uo pipefail
runs=~/SkyRL/skyrl-test/skycap-codex-eval/runs
run="$1"
if [[ "$run" == latest ]]; then
  run="$(ls -t "$runs"/*/config.json 2>/dev/null | head -1 | xargs -r dirname | xargs -r basename)"
  [[ -n "$run" ]] || { echo "no eval runs in $runs"; exit 4; }
fi
dir="$runs/$run"
[[ -f "$dir/config.json" ]] || { echo "no run $run in $runs"; exit 4; }
planned="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(len(c["tasks"]) * c["repeats"])' "$dir/config.json")"
done_="$( [[ -f "$dir/results.jsonl" ]] && wc -l < "$dir/results.jsonl" || echo 0)"
done_="${done_// /}"
accepted="$( [[ -f "$dir/results.jsonl" ]] && grep -c '"reward": 1.0' "$dir/results.jsonl" || true)"
errors="$( [[ -f "$dir/results.jsonl" ]] && grep -c '"error":' "$dir/results.jsonl" || true)"
minutes=$(( ($(date +%s) - $(stat -c %Y "$dir/config.json")) / 60 ))
# The run is alive while its runner or its skycap server (which records to records/<run>) is.
alive=0
pgrep -f "01_eval.py.*--run $run( |\$)" >/dev/null && alive=1
pgrep -f "skycap serve.*records/$run( |\$)" >/dev/null && alive=1
# The log the README's nohup command writes.
tail_line=""
[[ -f "$runs/$run.log" ]] && tail_line="$(grep -v '^[[:space:]]*$' "$runs/$run.log" | tail -1 | cut -c1-160)"
counts="$done_/$planned rollouts done, ${accepted:-0} accepted${errors:+, $errors errored}"
counts="${counts/, 0 errored/}"
if [[ $alive == 1 ]]; then
  state="running"; code=2
elif [[ "$done_" -ge "$planned" ]]; then
  state="finished"; code=0
else
  state="STOPPED EARLY"; code=1
fi
echo "$run: $state, $counts, ${minutes} min since it started${tail_line:+ | log: $tail_line}"
exit $code
EOF

check() { gssh --command "bash -s -- $(printf %q "$run")" <<<"$REMOTE"; }

if [[ $WAIT == 0 ]]; then
  rc=0; check || rc=$?
  exit $rc
fi
echo "checking every ${EVERY}s until the run ends; Ctrl-C to stop"
last=""
while :; do
  rc=0; line="$(check 2>/dev/null)" || rc=$?
  case $rc in
    0|1|4)
      echo "$(date +%H:%M:%S) $line"
      if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"$(sed 's/["\\]//g' <<<"$line" | cut -c1-200)\" with title \"skycap eval\"" >/dev/null 2>&1 || true
      fi
      exit $rc ;;
    2)
      # Print only when the count moves, not every minute.
      key="$(sed -E 's/, [0-9]+ min since.*//' <<<"$line")"
      if [[ "$key" != "$last" ]]; then echo "$(date +%H:%M:%S) $line"; last="$key"; fi ;;
    *) echo "$(date +%H:%M:%S) no answer from the VM (rc $rc); still trying (gcp/alive.sh says whether it is up)" ;;
  esac
  sleep "$EVERY"
done
