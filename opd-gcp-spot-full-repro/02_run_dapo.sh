#!/usr/bin/env bash
# The blog's RL runs: a base model trained with the DAPO recipe on DAPO-Math-17k, evaluated on AIME 2024.
#   02_run_dapo.sh 4b      Qwen3-4B-Base   -> the TEACHER for both OPD runs, and the 4B RL baseline curve
#   02_run_dapo.sh 1.7b    Qwen3-1.7B-Base -> the RL baseline the 1.7B OPD run is compared against
# The flags are examples/train/algorithms/dapo/run_dapo_aime_qwen3_4b_aime.sh (and its 1.7b twin) verbatim:
# batch 512 x 16, mini-batch 32 (16 optimizer steps per batch), LR 1e-6 with 160 warmup optimizer steps,
# dual-clip loss with clip-higher (0.2 / 0.28), overlong filtering plus the soft overlong punishment of
# main_dapo.py, no KL loss, no dynamic sampling, enforce_eager on the engines. One thing differs: the 4B run is
# written for 2 nodes x 8 GPUs and here takes this node's 8 (OPD_DAPO_NUM_GPUS). Mini-batch, optimizer steps
# and micro-batch size are unchanged, so the same gradients are computed over half as many ranks.
#
# Stops after DAPO_MAX_STEPS steps: 90 for 4b (the step the blog's OPD scripts name as the teacher, and where
# the blog's 4B curve ends), 200 for 1.7b (the blog compares against "~200 steps of RL"). A successful run
# uploads its final export to the Hub (OPD_UPLOAD=0 to skip).
#
#   nohup bash skyrl-test/opd-gcp-spot-full-repro/02_run_dapo.sh 4b > ~/opd-store/logs/dapo_4b.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-gcp-spot-full-repro/02_run_dapo.sh 4b --smoke     # 2 tiny steps, proves the plumbing
# Preempted or interrupted? Run the same command again: the run name is fixed and resume_mode=latest picks up
# the last checkpoint. Anything after the size (and --smoke) is passed to SkyRL as extra overrides.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
opd_model "${1:-}" || { echo "usage: $0 <4b|1.7b> [--smoke] [skyrl overrides...]" >&2; exit 2; }
shift
EXTRA=()
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  RUN_NAME="smoke_dapo_${MODEL_TAG}"
  # 16 prompts x 4 samples, two mini-batches of 8, 1k-token responses, one eval sample; exported at step 2 so
  # 03_run_opd.sh --smoke has a teacher to load. Starts from scratch every time.
  EXTRA=(trainer.train_batch_size=16 trainer.policy_mini_batch_size=8 generator.n_samples_per_prompt=4
         generator.sampling_params.max_generate_length=1024 generator.eval_sampling_params.max_generate_length=1024
         trainer.algorithm.overlong_buffer_len=256 generator.eval_n_samples_per_prompt=1 trainer.eval_interval=2
         trainer.max_training_steps=2 trainer.epochs=1 trainer.ckpt_interval=2 trainer.hf_save_interval=2
         trainer.resume_mode=none)
else
  case "$MODEL_TAG" in 4b) STEPS=90 ;; *) STEPS=200 ;; esac
  STEPS="${DAPO_MAX_STEPS:-$STEPS}"
  RUN_NAME="dapo_qwen3_${MODEL_TAG}_base${OPD_TAG:+_$OPD_TAG}"
  EXTRA=(trainer.max_training_steps="$STEPS")
  opd_skip_if_done "$RUN_NAME"
fi
opd_check_data
NUM_GPUS="${OPD_DAPO_NUM_GPUS:-8}"
opd_run_opts "$RUN_NAME" "$NUM_GPUS"
echo "$RUN_NAME" > "$LOGS/last_run"
echo "run_name=$RUN_NAME  method=DAPO  model=$MODEL_ID  gpus=$NUM_GPUS  ckpt=$CKPT_ROOT/$RUN_NAME  export=$EXPORT_ROOT/$RUN_NAME"

opd_uv_run -m examples.train.algorithms.dapo.main_dapo \
  "${TASK_OPTS[@]}" \
  "${RUN_OPTS[@]}" \
  trainer.policy.model.path="$MODEL_ID" \
  trainer.policy.fsdp_config.fsdp_size="$NUM_GPUS" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=dual_clip \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.overlong_buffer_len=4096 \
  trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
  generator.apply_overlong_filtering=true \
  generator.inference_engine.enforce_eager=true \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=160 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  "${EXTRA[@]}" \
  "$@"

if [[ "$RUN_NAME" == smoke_* ]]; then echo "=== $RUN_NAME finished (smoke: no marker, no upload)"; else opd_finish "$RUN_NAME"; fi
