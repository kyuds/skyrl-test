#!/usr/bin/env bash
# Runs on the VM, started detached by orchestrate.sh. Runs the given jobs one after another through common/job.sh and
# stops at the first that fails, so a job never starts on the remains of a failed one. One chain per node at a time;
# ~/opd-store/jobs/chain.pid and chain.jobs say what it is doing, and its own log is ~/opd-store/logs/chain.log.
#   setsid nohup bash skyrl-test/opd-blog-v2/common/chain.sh <job> [<job> ...] >> ~/opd-store/logs/chain.log 2>&1 &
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
STORE="${OPD_STORE:-$HOME/opd-store}"
J="$STORE/jobs"
mkdir -p "$J"
[[ $# -gt 0 ]] || { echo "usage: chain.sh <job> [<job> ...]" >&2; exit 2; }
pid="$(cat "$J/chain.pid" 2>/dev/null || true)"
if [[ -n "$pid" && "$pid" != "$$" ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q chain.sh; then
  echo "a chain is already running on this node (pid $pid: $(cat "$J/chain.jobs" 2>/dev/null))" >&2
  exit 1
fi
echo $$ > "$J/chain.pid"
echo "$*" > "$J/chain.jobs"
trap 'rm -f "$J/chain.pid"' EXIT

# A finished job's Ray actors and vLLM engines can take a moment to let go of their GPUs.
gpus_free() {
  local i
  for i in $(seq 1 60); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && return 0
    sleep 5
  done
  echo "warning: the GPUs are still busy after 5 minutes:"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader || true
}

echo "=== chain $(date -u +%Y-%m-%dT%H:%M:%SZ): $*"
for job in "$@"; do
  gpus_free
  if ! bash "$KIT/common/job.sh" "$job"; then
    echo "=== chain stopped at $job ($(date -u +%Y-%m-%dT%H:%M:%SZ)); the jobs after it did not start"
    exit 1
  fi
done
echo "=== chain finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
