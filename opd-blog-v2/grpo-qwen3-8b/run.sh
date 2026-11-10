#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. GRPO on Qwen3-8B with thinking off, on M2RL's math set; the teacher for phase 1.2.
# See README.md in this folder for every setting and why. From ~/SkyRL:
#   nohup bash skyrl-test/opd-blog-v2/grpo-qwen3-8b/run.sh >> ~/opd-store/logs/grpo_qwen3_8b_nothink.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-blog-v2/grpo-qwen3-8b/run.sh --smoke     # 2 tiny steps, W&B project opd-blog-v2-smoke
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run the same command again:
# the run name is fixed and resume_mode=latest picks up the last checkpoint.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init grpo_qwen3_8b_nothink "$@"

NUM_GPUS=8
v2_collect < <(
  printf '%s\n' \
    "trainer.policy.model.path=$QWEN3_8B" \
    "data.train_data=['$MATH_TRAIN']" \
    "data.val_data=['$AIME24','$AIME25']" \
    environment.env_class=opd_v2_math \
    "generator.chat_template_kwargs={enable_thinking: false}" \
    generator.batched=false \
    trainer.strategy=fsdp \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_nodes=1 \
    "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS" \
    "trainer.policy.fsdp_config.fsdp_size=$NUM_GPUS" \
    "generator.inference_engine.num_engines=$NUM_GPUS" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    generator.inference_engine.enforce_eager=true \
    trainer.train_batch_size=128 \
    generator.n_samples_per_prompt=16 \
    trainer.policy_mini_batch_size=128 \
    trainer.update_epochs_per_batch=1 \
    trainer.max_training_steps=400 \
    trainer.max_prompt_length=2048 \
    generator.sampling_params.max_generate_length=8192 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.eval_sampling_params.max_generate_length=8192 \
    generator.eval_sampling_params.temperature=0.6 \
    generator.eval_sampling_params.top_p=0.95 \
    generator.eval_n_samples_per_prompt=16 \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.policy_loss_type=regular \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.28 \
    trainer.algorithm.loss_reduction=token_mean \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.off_policy_correction.tis_ratio_type=token \
    trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
    trainer.policy.optimizer_config.lr=2e-6 \
    trainer.policy.optimizer_config.num_warmup_steps=20 \
    trainer.policy.optimizer_config.weight_decay=0.1 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.micro_forward_batch_size_per_gpu=8 \
    trainer.micro_train_batch_size_per_gpu=4
  v2_bookkeeping "['teacher','grpo','nemotron3-nano-rl-math','Qwen3-8B','non-thinking']"
)
v2_launch grpo
