#!/usr/bin/env bash
# OPD with a teacher this kit trained: an HF export of an earlier run (by default the last full OPD run's
# global_step_40, i.e. the 0.8B student distilled from the 9B) teaches a fresh student, OPD_STUDENT_MODEL from
# 00_env.sh. Same task, batch shape, GPUs and schedule as 03_run_opd.sh, which this script runs after pointing
# OPD_TEACHER_MODEL at the export, so the two runs differ only in the teacher. The run launches the teacher
# from the local export directory (no Hub download); the export carries the student's tokenizer.
#
#   bash skyrl-test/opd-4xh100/05_run_opd_from_export.sh                                  # last OPD run, step 40
#   TEACHER_RUN=<opd run name> TEACHER_STEP=20 bash skyrl-test/opd-4xh100/05_run_opd_from_export.sh
#   OPD_TEACHER_EXPORT=/path/to/global_step_N/policy bash skyrl-test/opd-4xh100/05_run_opd_from_export.sh
#   OPD_RUN_NAME=<this run's printed name> bash skyrl-test/opd-4xh100/05_run_opd_from_export.sh    # resume
#
# Which export: OPD_TEACHER_EXPORT if set; else $HOME/exports/$OPD_PROJECT/<TEACHER_RUN>/global_step_<TEACHER_STEP>/policy,
# TEACHER_RUN defaulting to ~/logs/last_opd_run as it is at launch (the run below overwrites that file; the
# default refuses when that names a run of this script) and TEACHER_STEP to 40. The resolved path is recorded in $OPD_LOGS/<run name>.teacher, and a resume with
# OPD_RUN_NAME reads it back, so a resumed run keeps its teacher whatever last_opd_run says by then.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
: "${OPD_STUDENT_MODEL:?source 00_env.sh first (it selects the pair)}"
LOGS="${OPD_LOGS:-$HOME/logs}"; PROJECT="${OPD_PROJECT:-opd_4xh100}"

if [[ -n "${OPD_RUN_NAME:-}" && -f "$LOGS/$OPD_RUN_NAME.teacher" ]]; then
  EXPORT="$(cat "$LOGS/$OPD_RUN_NAME.teacher")"
  echo "resuming $OPD_RUN_NAME with its recorded teacher: $EXPORT"
else
  if [[ -n "${OPD_TEACHER_EXPORT:-}" ]]; then
    EXPORT="$OPD_TEACHER_EXPORT"
  else
    SRC_RUN="${TEACHER_RUN:-$(cat "$LOGS/last_opd_run" 2>/dev/null || true)}"
    [[ -n "$SRC_RUN" ]] || { echo "no TEACHER_RUN given and no $LOGS/last_opd_run to default to" >&2; exit 1; }
    # last_opd_run names the newest OPD run of any kind; if that is one of these runs (it has a .teacher
    # record), defaulting to it would distill from a student of this script. Make the choice explicit.
    if [[ -z "${TEACHER_RUN:-}" && -f "$LOGS/$SRC_RUN.teacher" ]]; then
      echo "$LOGS/last_opd_run names $SRC_RUN, itself a run of this script; set TEACHER_RUN=<the OPD run whose export teaches>" >&2
      echo "(OPD runs with exports: $(ls "$HOME/exports/$PROJECT" 2>/dev/null | grep '^opd_' | tr '\n' ' '))" >&2; exit 1
    fi
    EXPORT="$HOME/exports/$PROJECT/$SRC_RUN/global_step_${TEACHER_STEP:-40}/policy"
  fi
  EXPORT="$(cd "$EXPORT" 2>/dev/null && pwd)" || { echo "teacher export not found: ${OPD_TEACHER_EXPORT:-$HOME/exports/$PROJECT/${SRC_RUN:-?}/global_step_${TEACHER_STEP:-40}/policy}" >&2
    echo "exports of that run: $(ls "$(dirname "$(dirname "${OPD_TEACHER_EXPORT:-$HOME/exports/$PROJECT/${SRC_RUN:-?}/x/policy}")")" 2>/dev/null | tr '\n' ' ')" >&2; exit 1; }
  STEP_TAG="$(echo "$EXPORT" | grep -oE 'global_step_[0-9]+' | tail -1 | sed 's/global_step_/s/')"
  export OPD_RUN_NAME="${OPD_RUN_NAME:-opd_${OPD_PAIR}_0p8b_from_export_${STEP_TAG:-x}_$(date +%Y%m%d%H%M%S)}"
  [[ "$EXPORT" != *"/$OPD_RUN_NAME/"* ]] || { echo "the teacher export belongs to the run being started ($OPD_RUN_NAME); pick another TEACHER_RUN" >&2; exit 1; }
fi
[[ -f "$EXPORT/config.json" ]] && compgen -G "$EXPORT/*.safetensors" >/dev/null || {
  echo "$EXPORT is not a complete HF export (needs config.json and *.safetensors)" >&2; exit 1; }
mkdir -p "$LOGS"; echo "$EXPORT" > "$LOGS/$OPD_RUN_NAME.teacher"

# A student trained with language_model_only=true is saved text-only (Qwen3_5ForCausalLM, no vision_config),
# so the teacher's text-only switch only applies when the export is still multimodal.
read -r ARCH IS_MM < <(python3 - "$EXPORT/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(",".join(c.get("architectures") or ["?"]), "yes" if c.get("vision_config") else "no")
PY
)
EXTRA=()
[[ "$IS_MM" == "yes" ]] || EXTRA+=(trainer.teacher.inference_engine.language_model_only=false)
echo "teacher export: $EXPORT  ($ARCH, multimodal: $IS_MM)"

# Cheap preflight: can this vLLM load that architecture? A miss would otherwise surface as a server-actor
# error minutes into the run. A failure of the check itself (no vllm import) only warns.
if [[ "${OPD_SKIP_ARCH_CHECK:-}" != 1 ]]; then
  source "$KIT/_locate.sh"; opd_locate
  verdict="$(cd "$SKYRL_DIR" && uv run --isolated --extra fsdp ${OPD_UV_WITH:-} python -c "
import sys
from vllm import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
print('ok' if all(a in archs for a in sys.argv[1].split(',')) else 'missing')
" "$ARCH" 2>/dev/null | tail -1 || true)"
  case "$verdict" in
    ok) echo "vLLM supports $ARCH" ;;
    missing) echo "this vLLM does not register $ARCH, so it cannot serve the export as the teacher" >&2; exit 1 ;;
    *) echo "WARNING: could not ask vLLM about $ARCH; continuing (OPD_SKIP_ARCH_CHECK=1 skips this check)" ;;
  esac
fi

export OPD_TEACHER_MODEL="$EXPORT"
exec bash "$KIT/03_run_opd.sh" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
