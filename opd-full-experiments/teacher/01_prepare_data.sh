#!/usr/bin/env bash
# Runs on the VM. 02_train_teacher.sh calls it first, so launch_all.sh needs no separate data step. By hand:
#   bash skyrl-test/opd-full-experiments/teacher/01_prepare_data.sh Qwen/Qwen3-8B
# DAPO-Math-17k, AIME 2024 and AIME 2025, through the repo's own prepare_dapo_data.sh. Then the base model into the HF
# cache, so the run does not download at step 0. The data is on the boot disk and is kept across a preemption. The
# cache is on the NVMe array and is not, so the model comes down again after one.
set -euo pipefail
MODEL_ID="${1:?usage: $0 <hf model id>}"
source "$(dirname "$0")/00_env.sh" >/dev/null
cd "$SKYRL_DIR"   # the example script and uv's project live here

if [[ -f "$DAPO_TRAIN" && -f "$EVAL_2024" && -f "$EVAL_2025" ]]; then
  echo "data already prepared"
else
  DATA_DIR="$DAPO_DIR" bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
ls -la "$DAPO_TRAIN" "$EVAL_2024" "$EVAL_2025"

echo "caching $MODEL_ID"
uv run --isolated --extra fsdp hf download "$MODEL_ID" >/dev/null
