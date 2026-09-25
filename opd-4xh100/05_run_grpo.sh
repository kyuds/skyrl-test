#!/usr/bin/env bash
# Baseline for Charlie's comparison: naive GRPO on the same student, data, eval, batch shape and lengths as
# the OPD run, with no teacher. "Naive" = SkyRL's GRPO defaults, stated here so nothing is inherited silently:
# grpo advantages (group-normalized), `regular` PPO-clip loss with eps 0.2/0.2, token_mean loss reduction,
# KL loss to the reference model with coef 0.001 (k3), no clip-higher, no overlong filtering or punishment,
# no dynamic sampling, no zero-variance filter. LR 1e-6 (the blog's RL baseline; 1e-5 was unstable for RL).
# Mini-batch 32 prompts, i.e. 16 optimizer steps per batch as in the blog's RL baseline; the OPD run has none.
# Runs on all four GPUs by default (no teacher to feed), so stop the teacher first; it refuses to start on
# a GPU a running teacher holds. GRPO_GPUS=2,3 GRPO_NUM_GPUS=2 gives the like-for-like wall clock instead.
# 200 steps by default (the blog needed ~200 RL steps for what OPD did in ~20); GRPO_MAX_STEPS=N to change.
# Interrupted? Re-run with GRPO_RUN_NAME=<the printed run name> to resume from the last checkpoint.
#   OPD_PAIR=post bash skyrl-test/opd-4xh100/05_run_grpo.sh --smoke   # 3 tiny steps, proves the plumbing
#   OPD_PAIR=post bash skyrl-test/opd-4xh100/05_run_grpo.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
GPUS="${GRPO_GPUS:-0,1,2,3}"
NUM_GPUS="${GRPO_NUM_GPUS:-4}"
SMOKE=()
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  SMOKE=(trainer.train_batch_size=16 trainer.policy_mini_batch_size=16 generator.n_samples_per_prompt=8
         generator.sampling_params.max_generate_length=2048 generator.eval_sampling_params.max_generate_length=2048
         generator.eval_n_samples_per_prompt=4 trainer.eval_batch_size=128 trainer.eval_interval=3
         trainer.max_training_steps=3 trainer.epochs=1 trainer.ckpt_interval=0 trainer.hf_save_interval=-1 trainer.resume_mode=none)
  RUN_NAME="${GRPO_RUN_NAME:-grpo_smoke_${OPD_PAIR}_$(date +%Y%m%d%H%M%S)}"
else
  RUN_NAME="${GRPO_RUN_NAME:-grpo_${OPD_PAIR}_0p8b_$(date +%Y%m%d%H%M%S)}"
fi
refuse_gpu_collision "$GPUS"
echo "$RUN_NAME" > "${OPD_LOGS:-$HOME/logs}/last_grpo_run"
echo "run_name=$RUN_NAME  pair=${OPD_PAIR}  student=${OPD_STUDENT_MODEL}  gpus=${GPUS}  thinking=${OPD_THINKING:-false}"
CUDA_VISIBLE_DEVICES="$GPUS" \
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$AIME_FILE','$GSM8K_FILE']" \
  trainer.policy.model.path="${OPD_STUDENT_MODEL}" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.remove_microbatch_padding=false \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=regular \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.2 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.kl_estimator_type=k3 \
  trainer.algorithm.zero_variance_filter=false \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.batched=true \
  generator.chat_template_kwargs="{enable_thinking: ${OPD_THINKING:-false}}" \
  environment.env_class=aime \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=0.7 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=32 \
  generator.n_samples_per_prompt=16 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_n_samples_per_prompt=32 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.epochs=20 \
  trainer.max_training_steps="${GRPO_MAX_STEPS:-200}" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=2 \
  trainer.hf_save_interval=50 \
  trainer.resume_mode=latest \
  trainer.logger=wandb \
  trainer.project_name="${OPD_PROJECT:-opd_4xh100}" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$HOME/ckpts/${OPD_PROJECT:-opd_4xh100}/$RUN_NAME" \
  trainer.export_path="$HOME/exports/${OPD_PROJECT:-opd_4xh100}/$RUN_NAME" \
  ${SMOKE[@]+"${SMOKE[@]}"} \
  "$@"
