#!/usr/bin/env bash
# The blog's OPD runs: a base model distilled from the DAPO-trained Qwen3-4B-Base, pure reverse-KL OPD.
#   03_run_opd.sh 4b      Qwen3-4B-Base   <- DAPO-trained 4B   ("distillation back into the base model")
#   03_run_opd.sh 1.7b    Qwen3-1.7B-Base <- DAPO-trained 4B   ("distilling into a smaller model")
# The flags are examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_4b.sh (and its 1.7b
# twin) from PR #2256: batch 512 x 16 with no mini-batching (one optimizer step per batch), LR 1e-5, no
# warmup, kl_coef 1.0, no task reward, micro-batches of 2, enforce_eager off, and the post's loss aggregation
# (seq_mean_token_sum_norm with max_seq_len = prompt + response; the entrypoint's default is token_mean).
# Run 02_run_dapo.sh 4b first.
#
# Teacher: the HF export of the DAPO 4B run at step TEACHER_STEP (90), read from this node's disk and served
# by the run itself (trainer.teacher.backend=skyrl) as OPD_TEACHER_NUM_GPUS TP-1 vLLM servers. The student
# takes OPD_NUM_STUDENT_GPUS more. In the blog the teacher was the reference-model slot, an FSDP forward on
# the student's own 8 GPUs; the entrypoint under test serves it from engines instead, hence the 4 + 4 split.
# TEACHER_RUN / TEACHER_STEP pick another export; OPD_TEACHER_MODEL=<path or Hub id> overrides both.
#
# Stops after OPD_MAX_STEPS steps: 30 for 4b, 60 for 1.7b (where the blog's W&B curves end, read off the
# charts). A successful run uploads its final export to the Hub (OPD_UPLOAD=0 to skip).
#
#   nohup bash skyrl-test/opd-gcp-spot-full-repro/03_run_opd.sh 4b > ~/opd-store/logs/opd_4b.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-gcp-spot-full-repro/03_run_opd.sh 4b --smoke     # after 02_run_dapo.sh 4b --smoke
# Preempted or interrupted? Run the same command again (fixed run name, resume_mode=latest). Anything after
# the size (and --smoke) is passed to SkyRL as extra overrides.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
opd_model "${1:-}" || { echo "usage: $0 <4b|1.7b> [--smoke] [skyrl overrides...]" >&2; exit 2; }
shift
SMOKE=0; if [[ "${1:-}" == "--smoke" ]]; then SMOKE=1; shift; fi

# ---- which teacher ----
if [[ $SMOKE == 1 ]]; then TEACHER_RUN="${TEACHER_RUN:-smoke_dapo_4b}"; TEACHER_STEP="${TEACHER_STEP:-2}"
else TEACHER_RUN="${TEACHER_RUN:-dapo_qwen3_4b_base}"; TEACHER_STEP="${TEACHER_STEP:-90}"; fi
if [[ -n "${OPD_TEACHER_MODEL:-}" ]]; then
  TEACHER="$OPD_TEACHER_MODEL"
  TEACHER_TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')"
else
  TEACHER="$EXPORT_ROOT/$TEACHER_RUN/global_step_$TEACHER_STEP/policy"
  TEACHER_TAG="dapo4b_s$TEACHER_STEP"
fi
case "$TEACHER" in
  /*) if [[ ! -f "$TEACHER/config.json" ]] || ! compgen -G "$TEACHER/*.safetensors" >/dev/null; then
        echo "no complete HF export at $TEACHER (needs config.json and *.safetensors)." >&2
        echo "exports of $TEACHER_RUN: $(ls "$EXPORT_ROOT/$TEACHER_RUN" 2>/dev/null | grep '^global_step_' | tr '\n' ' ')" >&2
        echo "Run 02_run_dapo.sh 4b first, or set TEACHER_STEP / TEACHER_RUN / OPD_TEACHER_MODEL." >&2
        exit 1
      fi ;;
esac

EXTRA=()
if [[ $SMOKE == 1 ]]; then
  RUN_NAME="smoke_opd_${MODEL_TAG}"
  # 16 prompts x 4 samples in one batch, 1k-token responses, one eval sample. Starts from scratch every time.
  EXTRA=(trainer.train_batch_size=16 trainer.policy_mini_batch_size=16 generator.n_samples_per_prompt=4
         generator.sampling_params.max_generate_length=1024 generator.eval_sampling_params.max_generate_length=1024
         generator.eval_n_samples_per_prompt=1 trainer.eval_interval=2
         trainer.max_training_steps=2 trainer.epochs=1 trainer.ckpt_interval=2 trainer.hf_save_interval=2
         trainer.resume_mode=none)
  MAX_SEQ_LEN=$((2048 + 1024))
else
  case "$MODEL_TAG" in 4b) STEPS=30 ;; *) STEPS=60 ;; esac
  STEPS="${OPD_MAX_STEPS:-$STEPS}"
  RUN_NAME="opd_qwen3_${MODEL_TAG}_base_from_${TEACHER_TAG}${OPD_TAG:+_$OPD_TAG}"
  EXTRA=(trainer.max_training_steps="$STEPS")
  MAX_SEQ_LEN=$((2048 + 8192))   # max prompt length + max response length (TASK_OPTS in _common.sh)
  opd_skip_if_done "$RUN_NAME"
fi
opd_check_data
NUM_GPUS="${OPD_NUM_STUDENT_GPUS:-4}"
TEACHER_GPUS="${OPD_TEACHER_NUM_GPUS:-4}"
opd_run_opts "$RUN_NAME" "$NUM_GPUS"
echo "$RUN_NAME" > "$LOGS/last_run"
echo "$TEACHER" > "$LOGS/$RUN_NAME.teacher"
echo "run_name=$RUN_NAME  method=OPD  student=$MODEL_ID on $NUM_GPUS GPUs  teacher=$TEACHER on $TEACHER_GPUS GPUs (launched by the run)"

# The teacher block's own defaults are not repeated: gpu_memory_utilization 0.9, prefix caching off, and
# max_model_len = longest input + longest response + 1 (2048 + 8192 + 1 = 10241). If the previous run's GPUs
# are not released yet, the teacher's placement group waits and fails after SKYRL_RAY_PG_TIMEOUT_IN_S (180 s).
opd_uv_run -m skyrl.train.entrypoints.main_opd \
  "${TASK_OPTS[@]}" \
  "${RUN_OPTS[@]}" \
  trainer.policy.model.path="$MODEL_ID" \
  trainer.teacher.backend=skyrl \
  trainer.teacher.model="$TEACHER" \
  trainer.teacher.inference_engine.num_engines="$TEACHER_GPUS" \
  trainer.teacher.inference_engine.tensor_parallel_size=1 \
  trainer.teacher.max_concurrency="${OPD_TEACHER_MAX_CONCURRENCY:-64}" \
  trainer.algorithm.opd.kl_coef=1.0 \
  trainer.algorithm.opd.use_task_reward=false \
  trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
  trainer.algorithm.max_seq_len="$MAX_SEQ_LEN" \
  generator.inference_engine.enforce_eager=false \
  trainer.policy_mini_batch_size=512 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  "${EXTRA[@]}" \
  "$@"

if [[ $SMOKE == 1 ]]; then echo "=== $RUN_NAME finished (smoke: no marker, no upload)"; else opd_finish "$RUN_NAME"; fi
