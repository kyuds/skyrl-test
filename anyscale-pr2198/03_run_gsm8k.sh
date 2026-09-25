#!/usr/bin/env bash
# Run B: non-step-wise smoke run. examples/train/gsm8k/run_gsm8k.sh is the config CI runs on l4_ci
# (Qwen2.5-1.5B-Instruct, 4 GPUs); the micro batch sizes are CI's L4 values.
#   bash skyrl-test/anyscale-pr2198/03_run_gsm8k.sh
set -euo pipefail
DATA="${PR2198_DATA:-$HOME/data}"
RUN_NAME="gsm8k_$(date +%Y%m%d%H%M%S)"
echo "$RUN_NAME" > /tmp/pr2198_last_gsm8k_run
echo "run_name=$RUN_NAME  data=$DATA"

NUM_GPUS=4 DATA_DIR="$DATA/gsm8k" bash examples/train/gsm8k/run_gsm8k.sh \
  data.val_data="['$DATA/gsm8k/validation_smoke.parquet']" \
  trainer.max_training_steps=3 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  trainer.eval_batch_size=128 \
  generator.eval_n_samples_per_prompt=2 \
  trainer.num_logger_eval_samples=4 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=0 \
  trainer.project_name="${PR2198_PROJECT:-pr2198_smoke}" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$HOME/ckpts/pr2198_gsm8k_smoke" \
  "$@"
