#!/usr/bin/env bash
# The experiment for the selected pair, at the blog's shape: pure OPD, batch 512 prompts x 16 samples with
# no mini-batching, LR 1e-5, prompt 2048 / response 8192, AIME24 and GSM8K-256 at avg@32 every 5 steps.
# 40 steps by default (the blog's 1.7B curve flattened by ~20); OPD_MAX_STEPS=N to change. Interrupted?
# Re-run with OPD_RUN_NAME=<the printed run name>: resume_mode=latest picks up the last checkpoint.
#   OPD_PAIR=post bash skyrl-test/opd-4xh100/03_run_opd.sh
#   OPD_PAIR=base bash skyrl-test/opd-4xh100/03_run_opd.sh
# Each run launches its own teacher (OPD_TEACHER_MODEL, on OPD_TEACHER_NUM_GPUS GPUs) and takes it down at
# the end, so the two pairs simply run one after the other.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
TEACHER_TAG="$(basename "${OPD_TEACHER_MODEL}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')"
RUN_NAME="${OPD_RUN_NAME:-opd_${OPD_PAIR}_0p8b_from_${TEACHER_TAG}_$(date +%Y%m%d%H%M%S)}"
opd_run \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=512 \
  generator.n_samples_per_prompt=16 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_n_samples_per_prompt=32 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.epochs=20 \
  trainer.max_training_steps="${OPD_MAX_STEPS:-40}" \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=2 \
  trainer.hf_save_interval=20 \
  trainer.resume_mode=latest \
  "$@"
