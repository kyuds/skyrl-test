#!/usr/bin/env bash
# Upload a run's HF export (safetensors + config + tokenizer) to the Hugging Face Hub as a public model repo.
# The run scripts call this after a successful run; by hand it uploads any other step or retries a failed upload.
#
#   bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh                       # the last run started ($OPD_LOGS/last_run)
#   bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --run dapo_qwen3_4b_base
#   bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --run <run> --step 50
#   bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --repo my-model-name  # instead of the derived name
#   bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --private             # default is public
#   bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --dry-run             # show what would go where
#
# Uploads $OPD_STORE/exports/$OPD_PROJECT/<run>/global_step_<N>/policy to <HF_USER>/<repo>. --step defaults to
# the run's highest complete export, the repo name to opd-gcp-spot-full-repro-<run name>-step<N>. The Hub
# creates the repo on first upload (public unless --private; the flag does not change an existing repo); a second
# upload to the same repo adds a commit. HF_USER defaults to kyuds.
#
# Needs a write token in HF_TOKEN (never in this repo, which is public). Needs 00_env.sh sourced, or at least
# OPD_STORE set.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
export OPD_STORE="${OPD_STORE:-$HOME/opd-store}"
source "$KIT/_common.sh"
HF_USER="${HF_USER:-kyuds}"
RUN=""; STEP=""; REPO=""; VISIBILITY=""; DRY=0; FINAL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --step) STEP="$2"; FINAL=0; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --private) VISIBILITY="--private"; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1 (--help lists them)" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN" ]] || RUN="$(cat "$LOGS/last_run" 2>/dev/null || true)"
[[ -n "$RUN" ]] || { echo "no --run given and no $LOGS/last_run to default to" >&2; exit 1; }
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
complete "$EXPORT" || { echo "$EXPORT is not a complete HF export (needs config.json and *.safetensors); steps exported: $(ls "$RUN_DIR" | grep '^global_step_' | tr '\n' ' ')" >&2; exit 1; }

[[ -n "$REPO" ]] || REPO="opd-gcp-spot-full-repro-$(echo "$RUN" | tr '_' '-')-step$STEP"
REPO="${REPO#"$HF_USER"/}"   # accept either "name" or "user/name"
[[ "$REPO" =~ ^[A-Za-z0-9._-]{1,96}$ ]] || { echo "repo name '$REPO' is not valid on the Hub (letters, digits, . _ -, at most 96 characters); pass --repo" >&2; exit 1; }
REPO_ID="$HF_USER/$REPO"

echo "run    : $RUN (step $STEP)"
echo "export : $EXPORT  ($(du -sh "$EXPORT" 2>/dev/null | cut -f1), $(ls "$EXPORT" | wc -l | tr -d ' ') files)"
echo "repo   : https://huggingface.co/$REPO_ID  ($([[ -n "$VISIBILITY" ]] && echo private || echo public))"
[[ $DRY == 0 ]] || { echo "dry run: nothing uploaded"; exit 0; }

cd "$SKYRL_DIR"   # uv resolves the project from the cwd; its environment carries the `hf` CLI
if [[ -z "${HF_TOKEN:-}" ]] && ! uv run --isolated --extra fsdp hf auth whoami >/dev/null 2>&1; then
  echo "not logged in to the Hub: export HF_TOKEN=<a write token from huggingface.co/settings/tokens> and re-run" >&2; exit 1
fi
uv run --isolated --extra fsdp hf upload "$REPO_ID" "$EXPORT" $VISIBILITY \
  --commit-message "Upload $RUN global_step_$STEP (skyrl-test/opd-gcp-spot-full-repro)"
# The marker the run scripts check is only for the run's final export, not for a step picked by hand.
[[ $FINAL == 0 ]] || echo "https://huggingface.co/$REPO_ID (step $STEP, $(date -u +%Y-%m-%dT%H:%M:%SZ))" > "$LOGS/$RUN.uploaded"
echo "uploaded: https://huggingface.co/$REPO_ID"
