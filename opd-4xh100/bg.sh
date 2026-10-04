#!/usr/bin/env bash
# Run one of this kit's scripts detached from the terminal, so a dropped SSH session or a sleeping laptop
# does not kill it: its own session (setsid; its own process group on macOS) with hangups ignored (nohup),
# stdin from /dev/null, stdout and stderr to a log under $OPD_LOGS. The log's last line says how it ended.
#
#   bash skyrl-test/opd-4xh100/bg.sh 04_run_grpo.sh --smoke     # start; prints the log path
#   tail -f ~/logs/latest.log                                    # follow the newest detached run
#   bash skyrl-test/opd-4xh100/bg.sh --status                    # running or not, and the log's last lines
#   bash skyrl-test/opd-4xh100/bg.sh --stop                      # stop it: its whole process group
#
# Start from the shell where you sourced 00_env.sh: the detached run inherits its exports (and
# WANDB_API_KEY). One detached run at a time, since each experiment needs the whole node. --status and
# --stop need no env.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
LOGS="${OPD_LOGS:-$HOME/logs}"; mkdir -p "$LOGS"
PIDFILE="$LOGS/bg_latest.pid"; LATEST="$LOGS/latest.log"
alive() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
latest_log() { readlink "$LATEST" 2>/dev/null || echo "none"; }

case "${1:-}" in
  ""|-h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
  --status)
    if alive; then echo "running: pid $(cat "$PIDFILE"), log $(latest_log)"; else echo "not running; last log: $(latest_log)"; fi
    if [[ -e "$LATEST" ]]; then echo "--- last lines"; tail -5 "$LATEST"; fi
    exit 0 ;;
  --stop)
    alive || { echo "no detached run is going (last log: $(latest_log))"; exit 0; }
    pid="$(cat "$PIDFILE")"
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    if kill -0 "$pid" 2>/dev/null; then echo "still alive after 60 s; sending SIGKILL"; kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid"; fi
    echo "stopped pid $pid; its Ray job ends with the driver, which releases the job's actors and GPUs"
    exit 0 ;;
esac

: "${OPD_STUDENT_MODEL:?source 00_env.sh first, in this shell: the detached run inherits its exports}"
script="$1"; shift
[[ -f "$KIT/$script" ]] || { echo "no such script in the kit: $script (one of: $(cd "$KIT" && ls 0[2-9]_*.sh | tr '\n' ' '))"; exit 2; }
if alive; then echo "a detached run is still going (pid $(cat "$PIDFILE"), log $(latest_log)); bg.sh --status or --stop first"; exit 1; fi
suffix=""; [[ $# -gt 0 ]] && suffix="_$(printf '%s_' "$@" | sed 's/--*//g; s/[^A-Za-z0-9_.=]/_/g; s/_*$//' | cut -c1-40)"
log="$LOGS/$(basename "$script" .sh)${suffix}_$(date +%Y%m%d_%H%M%S).log"
# The child runs the script, then appends the end marker with its exit status.
RUNNER='bash "$0" "$@"; status=$?; echo "=== $(basename "$0")${*:+ $*} exited with status $status at $(date "+%F %T")"; exit $status'
if command -v setsid >/dev/null 2>&1; then
  setsid nohup bash -c "$RUNNER" "$KIT/$script" "$@" > "$log" 2>&1 < /dev/null &
else   # macOS has no setsid: job control gives the run its own process group instead
  set -m; nohup bash -c "$RUNNER" "$KIT/$script" "$@" > "$log" 2>&1 < /dev/null & set +m
fi
echo $! > "$PIDFILE"
ln -sfn "$log" "$LATEST"
echo "started: $script $*  (pid $(cat "$PIDFILE"), detached)"
echo "log    : $log"
echo "follow : tail -f $LATEST"
echo "status : bash $KIT/bg.sh --status      stop: bash $KIT/bg.sh --stop"
