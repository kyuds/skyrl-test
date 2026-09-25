#!/usr/bin/env bash
# Run A: Charlie's step-wise SQL script, cut to a 3-step smoke run on 4xL4.
#   bash skyrl-test/anyscale-pr2198/02_run_sql_stepwise.sh
#   MODEL=Qwen/Qwen3-1.7B bash skyrl-test/anyscale-pr2198/02_run_sql_stepwise.sh   # if Qwen3-4B OOMs
# Every line below re-sets a key the script already passes. OmegaConf.from_cli applies overrides in
# order, last wins -- CI's tests/train/gpu_e2e_test/gsm8k_colocate.sh relies on exactly this.
set -euo pipefail
DATA="${PR2198_DATA:-$HOME/data}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
RUN_NAME="sql_stepwise_$(date +%Y%m%d%H%M%S)"
echo "$RUN_NAME" > /tmp/pr2198_last_sql_run
echo "run_name=$RUN_NAME  model=$MODEL  data=$DATA"

bash examples/train/step_wise/run_skyrl_sql_step_wise_qwen3.sh \
  trainer.policy.model.path="$MODEL" \
  data.train_data="['$DATA/sql/train.parquet']" \
  data.val_data="['$DATA/sql/validation_smoke.parquet']" \
  environment.skyrl_gym.text2sql.db_path="$DATA/sql/db_files/data" \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=2 \
  trainer.max_training_steps=3 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  trainer.eval_batch_size=16 \
  generator.eval_n_samples_per_prompt=2 \
  trainer.num_logger_eval_samples=4 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  generator.n_samples_per_prompt=2 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=-1 \
  trainer.dump_data_batch=false \
  trainer.project_name="${PR2198_PROJECT:-pr2198_smoke}" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$HOME/ckpts/pr2198_sql_smoke" \
  "$@"
