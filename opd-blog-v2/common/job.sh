#!/usr/bin/env bash
# Runs on the VM. Runs one job of config.sh's table by name and records it in ~/opd-store/jobs/: <job>.started,
# <job>.pid while it runs, then <job>.done or <job>.failed. Its output goes to ~/opd-store/logs/<job>.log. A job that is
# done is not run again (delete its .done to rerun it); one that failed or was cut off resumes where it can, since
# every training script resumes from its last checkpoint. orchestrate.sh runs jobs through common/chain.sh; by hand,
# from ~/SkyRL:
#   nohup bash skyrl-test/opd-blog-v2/common/job.sh <job> > /dev/null 2>&1 < /dev/null &
#   tail -f ~/opd-store/logs/<job>.log
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
source "$KIT/config.sh"
job="${1:?usage: job.sh <job> (a name from config.sh)}"
cmd="$(v2_job_command "$job")"
[[ -n "$cmd" ]] || { echo "no job called $job in config.sh" >&2; exit 2; }
read -r -a parts <<<"$cmd"
STORE="${OPD_STORE:-$HOME/opd-store}"
J="$STORE/jobs"; L="$STORE/logs"
mkdir -p "$J" "$L"

if [[ -f "$J/$job.done" ]]; then
  echo "$job: done at $(cat "$J/$job.done"); not running it again (delete $J/$job.done to rerun)"
  exit 0
fi
pid="$(cat "$J/$job.pid" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "job.sh $job"; then
  echo "$job is already running (pid $pid)" >&2
  exit 1
fi
echo $$ > "$J/$job.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$J/$job.started"
rm -f "$J/$job.failed"
echo "$job: started $(cat "$J/$job.started"), log $L/$job.log"
echo "=== job $job: $cmd  ($(date -u +%Y-%m-%dT%H:%M:%SZ), kit $(git -C "$KIT" rev-parse --short HEAD 2>/dev/null || echo '?'))" >> "$L/$job.log"
rc=0
bash "$KIT/${parts[0]}" "${parts[@]:1}" >> "$L/$job.log" 2>&1 < /dev/null || rc=$?
rm -f "$J/$job.pid"
if [[ $rc == 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$J/$job.done"
  echo "$job: done at $(cat "$J/$job.done")"
else
  echo "exit $rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$J/$job.failed"
  echo "$job: FAILED with exit $rc; the end of its log:"
  tail -n 15 "$L/$job.log" | cut -c1-200 | sed 's/^/    /'
fi
exit $rc
