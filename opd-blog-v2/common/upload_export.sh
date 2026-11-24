#!/usr/bin/env bash
# Runs on the VM. Uploads one of a run's HF exports (safetensors + config + tokenizer) to its own folder in the Hub
# repo. Every OPD run uploads its last export when it finishes; this is for any other step, picked from the run's
# eval curve on W&B. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/upload_export.sh --run opd_2b_m2rl_sft_from_qwen3_32b --step 250
#   bash skyrl-test/opd-blog-v2/common/upload_export.sh --run <run> --step 250 --dry-run   # show what would go where
#
# Uploads $EXPORT_ROOT/<run>/global_step_<N>/policy to the folder <run>_step<N> of $V2_HF_REPO (config.sh). --step
# defaults to the run's highest complete export. The Hub creates the repo on the first upload, public.
set -euo pipefail
source "$(dirname "$0")/00_env.sh" >/dev/null
RUN=""; STEP=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1 (--help lists them)" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN" ]] || { echo "usage: $0 --run <run> [--step N] [--dry-run]" >&2; exit 2; }
RUN_DIR="$EXPORT_ROOT/$RUN"
[[ -d "$RUN_DIR" ]] || { echo "no exports for run '$RUN' ($RUN_DIR); runs with exports: $(ls "$EXPORT_ROOT" 2>/dev/null | tr '\n' ' ')" >&2; exit 1; }

complete() { [[ -f "$1/config.json" ]] && compgen -G "$1/*.safetensors" >/dev/null; }   # $1 = a policy dir
if [[ -z "$STEP" ]]; then   # the highest step whose export is complete
  for n in $(ls "$RUN_DIR" | sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p' | sort -n); do
    if complete "$RUN_DIR/global_step_$n/policy"; then STEP="$n"; fi
  done
  [[ -n "$STEP" ]] || { echo "run '$RUN' has no complete export under $RUN_DIR (found: $(ls "$RUN_DIR" | tr '\n' ' '))" >&2; exit 1; }
fi
EXPORT="$RUN_DIR/global_step_$STEP/policy"
complete "$EXPORT" || { echo "$EXPORT is not a complete HF export (needs config.json and *.safetensors)" >&2; exit 1; }

FOLDER="${RUN}_step$STEP"
URL="https://huggingface.co/$V2_HF_REPO/tree/main/$FOLDER"
echo "run    : $RUN (step $STEP)"
echo "export : $EXPORT  ($(du -sh "$EXPORT" 2>/dev/null | cut -f1), $(ls "$EXPORT" | wc -l | tr -d ' ') files)"
echo "goes to: $URL"
[[ $DRY == 0 ]] || { echo "dry run: nothing uploaded"; exit 0; }

cd "$SKYRL_DIR"   # uv resolves the project from the cwd; its environment carries the `hf` CLI
# Several nodes may commit to the same repo at once, which can conflict, so a failure is tried again.
for attempt in 1 2 3 4 5; do
  if uv run --isolated --extra fsdp hf upload "$V2_HF_REPO" "$EXPORT" "$FOLDER" \
       --commit-message "Upload $RUN global_step_$STEP (skyrl-test/opd-blog-v2)"; then
    echo "uploaded: $URL"
    exit 0
  fi
  echo "upload attempt $attempt failed"
  [[ $attempt == 5 ]] || sleep "${V2_UPLOAD_RETRY_SECS:-120}"
done
exit 1
