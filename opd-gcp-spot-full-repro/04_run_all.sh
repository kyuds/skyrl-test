#!/usr/bin/env bash
# The whole reproduction back to back on the one node, in the order that gets to the result soonest:
#   1. DAPO  Qwen3-4B-Base   (90 steps)   the teacher, and the 4B RL curve
#   2. OPD   Qwen3-4B-Base   <- 1         (30 steps)
#   3. OPD   Qwen3-1.7B-Base <- 1         (60 steps)
#   4. DAPO  Qwen3-1.7B-Base (200 steps)  the 1.7B RL curve that 3 is compared against; the longest stage, so last
# Each stage uploads its final export when it succeeds. A stage that already finished is skipped (its
# $OPD_LOGS/<run>.done marker), so after a preemption the same command resumes the stage that was running.
# A failed upload does not stop the sequence (the OPD stages read the teacher from disk); a failed run does.
#   nohup bash skyrl-test/opd-gcp-spot-full-repro/04_run_all.sh > ~/opd-store/logs/run_all.log 2>&1 < /dev/null &
# OPD_STAGES picks a subset in order, e.g. OPD_STAGES="dapo:4b opd:4b".
set -uo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
: "${OPD_STORE:?source 00_env.sh first}"
UPLOADS_FAILED=()
for stage in ${OPD_STAGES:-dapo:4b opd:4b opd:1.7b dapo:1.7b}; do
  kind="${stage%%:*}"; size="${stage##*:}"
  case "$kind" in
    dapo) script="$KIT/02_run_dapo.sh" ;;
    opd)  script="$KIT/03_run_opd.sh" ;;
    *) echo "unknown stage '$stage' (want dapo:<size> or opd:<size>)" >&2; exit 2 ;;
  esac
  echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  stage $stage ############"
  bash "$script" "$size"; rc=$?
  if [[ $rc == 3 ]]; then UPLOADS_FAILED+=("$stage")
  elif [[ $rc != 0 ]]; then echo "stage $stage failed with status $rc; stopping. Fix it and run this script again." >&2; exit "$rc"
  fi
done
echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  all stages finished ############"
if [[ ${#UPLOADS_FAILED[@]} -gt 0 ]]; then
  echo "uploads still to do: ${UPLOADS_FAILED[*]} (run this script again, or upload_export.sh --run <run>)" >&2
  exit 3
fi
