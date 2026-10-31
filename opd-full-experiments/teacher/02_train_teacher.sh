#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. Trains one base model on Nemotron-Cascade-RL-Math, evaluates on AIME 2024 and 2025,
# and uploads the final export to the Hub. launch_all.sh starts it on every node. By hand:
#   nohup bash skyrl-test/opd-full-experiments/teacher/02_train_teacher.sh Qwen/Qwen3-8B >> ~/opd-store/logs/<run>.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-full-experiments/teacher/02_train_teacher.sh Qwen/Qwen3-8B --smoke    # 2 tiny steps, no upload
# Anything after the model (and --smoke) is passed to SkyRL as extra overrides.
#
# The recipe is opd-gcp-spot-full-repro/02_run_dapo.sh, which is examples/train/algorithms/dapo/run_dapo_*_aime.sh flag
# for flag: GRPO advantages, dual-clip loss with clip-higher (0.2 / 0.28), overlong filtering plus the soft overlong
# punishment, no KL loss, LR 1e-6. What differs, all set in config.sh or below:
#   - Data: Nemotron-Cascade-RL-Math instead of DAPO-Math-17k, and AIME 2025 added to the eval.
#   - Batch: 128 prompts x 8 rollouts instead of 512 x 16, 2 optimizer steps per training step instead of 16.
#   - Thinking off, on both Base and instruct models: the chat template opens every answer with an empty think block.
#     That needs generator.batched=false, because the batched path drops chat_template_kwargs.
#   - Eval every 25 steps, checkpoints every 10, HF exports every 100.
#
# Preempted or interrupted? launch_all.sh starts this again: the run name is fixed and resume_mode=latest picks up the
# last checkpoint. A finished run is not trained again; only a missing upload is retried.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
MODEL_ID="${1:-}"
[[ "$MODEL_ID" == */* ]] || { echo "usage: $0 <hf model id> [--smoke] [skyrl overrides...]" >&2; exit 2; }
shift
RUN="$(run_name "$MODEL_ID")"
PROJECT="$TEACHER_PROJECT"
STEPS="$TEACHER_STEPS"
SMOKE=0
EXTRA=()
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  SMOKE=1
  RUN="smoke_$RUN"
  PROJECT="${TEACHER_PROJECT}_smoke"   # keeps the real project to the real runs
  STEPS=2
  rm -f "$LOGS/$RUN.smoke_ok"
  # 16 prompts x 4 rollouts, 1k-token responses, one eval sample, exported at step 2. Starts from scratch every time.
  EXTRA=(trainer.train_batch_size=16 trainer.policy_mini_batch_size=8 generator.n_samples_per_prompt=4
         generator.sampling_params.max_generate_length=1024 generator.eval_sampling_params.max_generate_length=1024
         trainer.algorithm.overlong_buffer_len=256 generator.eval_n_samples_per_prompt=1 trainer.eval_interval=2
         trainer.epochs=1 trainer.ckpt_interval=2 trainer.hf_save_interval=2 trainer.resume_mode=none
         trainer.policy.optimizer_config.num_warmup_steps=0)
else
  rm -rf "$CKPT_ROOT"/smoke_* "$EXPORT_ROOT"/smoke_*   # a smoke test that failed leaves these behind
fi

upload() {
  if bash "$TEACHER_KIT/upload_export.sh" --run "$RUN"; then return 0; fi
  echo "TRAINING FINISHED BUT THE UPLOAD FAILED for $RUN. The export is intact under $EXPORT_ROOT/$RUN." >&2
  echo "launch_all.sh retries it." >&2
  return 3
}
if [[ $SMOKE == 0 && -f "$LOGS/$RUN.done" ]]; then
  echo "$RUN already finished ($(cat "$LOGS/$RUN.done")); not training again"
  rc=0; upload || rc=$?
  exit "$rc"
fi

bash "$TEACHER_KIT/01_prepare_data.sh" "$MODEL_ID"
NUM_GPUS=8
echo "=== run=$RUN  project=$PROJECT  model=$MODEL_ID  steps=$STEPS  gpus=$NUM_GPUS  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# HF_TOKEN is removed from the training process: SkyRL logs its value when it copies it into the Ray runtime env, and
# nothing in training needs it. Only the upload does.
cd "$SKYRL_DIR"   # uv resolves the project (and its extras) from the cwd
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$NEMOTRON_FILE']" \
  data.val_data="['$EVAL_2024','$EVAL_2025']" \
  environment.env_class=aime \
  trainer.policy.model.path="$MODEL_ID" \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=0.7 \
  generator.batched=false \
  generator.chat_template_kwargs="{enable_thinking: false}" \
  generator.n_samples_per_prompt="$TEACHER_SAMPLES" \
  generator.eval_n_samples_per_prompt=32 \
  trainer.train_batch_size="$TEACHER_PROMPTS" \
  trainer.policy_mini_batch_size="$TEACHER_MINI_BATCH" \
  trainer.max_training_steps="$STEPS" \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval="$TEACHER_EVAL_INTERVAL" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.policy.fsdp_config.fsdp_size="$NUM_GPUS" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.enforce_eager=true \
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
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.num_warmup_steps="$TEACHER_WARMUP" \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.ckpt_interval="$TEACHER_CKPT_INTERVAL" \
  trainer.max_ckpts_to_keep=2 \
  trainer.hf_save_interval="$TEACHER_HF_INTERVAL" \
  trainer.resume_mode=latest \
  trainer.logger=wandb \
  trainer.project_name="$PROJECT" \
  trainer.run_name="$RUN" \
  trainer.tags="['teacher','$TEACHER_METHOD','$TEACHER_DATA_LONG','${MODEL_ID##*/}']" \
  trainer.ckpt_path="$CKPT_ROOT/$RUN" \
  trainer.export_path="$EXPORT_ROOT/$RUN" \
  trainer.log_path="$LOGS/skyrl/$RUN" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"

if [[ $SMOKE == 1 ]]; then
  bash "$TEACHER_KIT/upload_export.sh" --run "$RUN" --dry-run
  rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/$RUN.smoke_ok"
  echo "=== $RUN passed"
  exit 0
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/$RUN.done"
echo "=== $RUN finished training at $(cat "$LOGS/$RUN.done")"
upload
