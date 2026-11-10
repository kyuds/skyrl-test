#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. Phase 1.2: OPD of Qwen3-8B from the GRPO teacher in grpo-qwen3-8b/, thinking off, on
# M2RL's math set. It runs on the GRPO run's node, after that run, because its teacher is one of that run's exports.
# See README.md in this folder. From ~/SkyRL, with the export step you picked from the GRPO run's eval:
#   TEACHER_STEP=250 nohup bash skyrl-test/opd-blog-v2/phase1-opd/run_qwen3_8b.sh >> ~/opd-store/logs/p1_opd_qwen3_8b_from_grpo.log 2>&1 < /dev/null &
#   TEACHER_STEP=250 bash skyrl-test/opd-blog-v2/phase1-opd/run_qwen3_8b.sh --smoke
# TEACHER_MODEL=<path or HF id> replaces the export (e.g. after downloading it from the Hub to another node).
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run the same command again,
# with the same TEACHER_STEP.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init p1_opd_qwen3_8b_from_grpo "$@"
TRAIN_FILE="$MATH_TRAIN"

if [[ -n "${TEACHER_MODEL:-}" ]]; then
  TEACHER="$TEACHER_MODEL"
elif [[ $V2_DRY_RUN == 1 ]]; then
  TEACHER="$EXPORT_ROOT/grpo_qwen3_8b_nothink/global_step_${TEACHER_STEP:-400}/policy"
else
  : "${TEACHER_STEP:?set TEACHER_STEP to the GRPO export to distill from (a multiple of 50; see README.md)}"
  TEACHER="$EXPORT_ROOT/grpo_qwen3_8b_nothink/global_step_$TEACHER_STEP/policy"
  [[ -f "$TEACHER/config.json" ]] || {
    echo "no GRPO export at $TEACHER. Exports on this node: $(ls "$EXPORT_ROOT/grpo_qwen3_8b_nothink" 2>/dev/null | tr '\n' ' ')" >&2; exit 1
  }
fi
echo "teacher: $TEACHER" >&2

STUDENT_GPUS=4
TEACHER_ENGINES=4
MAX_RESP=8192        # thinking off: answers ran 2,000 to 3,000 tokens in the 2026-11 teacher runs
v2_collect < <(
  v2_opd_recipe "$QWEN3_8B" "$STUDENT_GPUS" "$MAX_RESP" "$MAX_RESP" 2 "generator.chat_template_kwargs={enable_thinking: false}"
  printf '%s\n' \
    trainer.teacher.backend=skyrl \
    "trainer.teacher.model=$TEACHER" \
    "trainer.teacher.inference_engine.num_engines=$TEACHER_ENGINES" \
    trainer.teacher.inference_engine.tensor_parallel_size=1 \
    "trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=$((2048 + MAX_RESP + 1))" \
    trainer.teacher.max_concurrency=64
  v2_bookkeeping "['opd','phase1','nemotron3-nano-rl-math','Qwen3-8B','non-thinking','teacher:grpo_qwen3_8b_nothink']"
)
v2_launch opd
