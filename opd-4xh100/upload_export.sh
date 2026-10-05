#!/usr/bin/env bash
# Upload a run's HF export (safetensors + config + tokenizer) to the Hugging Face Hub, as a private model repo.
#
#   bash skyrl-test/opd-4xh100/upload_export.sh                      # newest OPD run (~/logs/last_opd_run), its last export
#   bash skyrl-test/opd-4xh100/upload_export.sh --grpo               # newest GRPO run (~/logs/last_grpo_run)
#   bash skyrl-test/opd-4xh100/upload_export.sh --run <run name> --step 20
#   bash skyrl-test/opd-4xh100/upload_export.sh --repo my-model-name # repo name instead of the derived one
#   bash skyrl-test/opd-4xh100/upload_export.sh --public             # default is private
#   bash skyrl-test/opd-4xh100/upload_export.sh --dry-run            # show what would be uploaded where
#
# Uploads $HOME/exports/$OPD_PROJECT/<run>/global_step_<N>/policy to <HF_USER>/<repo>; --step defaults to the
# run's highest complete export, and the repo name to opd-4xh100-<run name>-step<N> (the run name carries the
# method, the pair and the start time, so the repo traces back to its W&B run). The Hub creates the repo on
# first upload; a second upload to the same repo adds a commit. HF_USER defaults to kyuds.
#
# Needs a write token: export HF_TOKEN=... in the shell (never in this repo, which is public), or a prior
# `hf auth login`. Needs no sourced env.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT/_locate.sh"; opd_locate || exit 1
LOGS="${OPD_LOGS:-$HOME/logs}"; PROJECT="${OPD_PROJECT:-opd_4xh100}"; HF_USER="${HF_USER:-kyuds}"
RUN=""; STEP=""; REPO=""; VISIBILITY="--private"; DRY=0; LAST="$LOGS/last_opd_run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --grpo) LAST="$LOGS/last_grpo_run"; shift ;;
    --step) STEP="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --public) VISIBILITY=""; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown argument: $1 (--help lists them)" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN" ]] || RUN="$(cat "$LAST" 2>/dev/null || true)"
[[ -n "$RUN" ]] || { echo "no --run given and no $LAST to default to" >&2; exit 1; }
RUN_DIR="$HOME/exports/$PROJECT/$RUN"
[[ -d "$RUN_DIR" ]] || { echo "no exports for run '$RUN' ($RUN_DIR); runs with exports: $(ls "$HOME/exports/$PROJECT" 2>/dev/null | tr '\n' ' ')" >&2; exit 1; }

complete() { [[ -f "$1/config.json" ]] && compgen -G "$1/*.safetensors" >/dev/null; }   # $1 = a policy dir
if [[ -z "$STEP" ]]; then   # the highest step whose export is complete
  for n in $(ls "$RUN_DIR" | sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p' | sort -n); do
    complete "$RUN_DIR/global_step_$n/policy" && STEP="$n"
  done
  [[ -n "$STEP" ]] || { echo "run '$RUN' has no complete export under $RUN_DIR (found: $(ls "$RUN_DIR" | tr '\n' ' '))" >&2; exit 1; }
fi
EXPORT="$RUN_DIR/global_step_$STEP/policy"
complete "$EXPORT" || { echo "$EXPORT is not a complete HF export (needs config.json and *.safetensors); steps exported: $(ls "$RUN_DIR" | grep '^global_step_' | tr '\n' ' ')" >&2; exit 1; }

[[ -n "$REPO" ]] || REPO="opd-4xh100-$(echo "$RUN" | tr '_' '-')-step$STEP"
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
  --commit-message "Upload $RUN global_step_$STEP (skyrl-test/opd-4xh100)"
echo "uploaded: https://huggingface.co/$REPO_ID"
