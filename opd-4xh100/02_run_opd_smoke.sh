#!/usr/bin/env bash
# Stage 0, plumbing: 3 steps of pure OPD on the selected pair, tiny batch, short responses, a 4-sample eval
# before and after. Proves the run brings up its teacher servers, the Qwen3.5 student trains on FSDP, syncs
# to its engines and gets scored; puts opd/* and timing/* on W&B. Expect well under an hour after the caches
# are warm (the teacher's model load is part of every run).
#   OPD_PAIR=post bash skyrl-test/opd-4xh100/02_run_opd_smoke.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
RUN_NAME="smoke_${OPD_PAIR}_$(date +%Y%m%d%H%M%S)"
opd_run \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  generator.n_samples_per_prompt=8 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=2048 \
  generator.eval_sampling_params.max_generate_length=2048 \
  generator.eval_n_samples_per_prompt=4 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=3 \
  trainer.max_training_steps=3 \
  trainer.epochs=1 \
  trainer.ckpt_interval=0 \
  trainer.resume_mode=none \
  "$@"
