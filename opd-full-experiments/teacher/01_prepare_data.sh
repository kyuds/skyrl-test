#!/usr/bin/env bash
# Runs on the VM. 02_train_teacher.sh calls it first, so launch_all.sh needs no separate data step. By hand:
#   bash skyrl-test/opd-full-experiments/teacher/01_prepare_data.sh Qwen/Qwen3-8B
# Training data: Nemotron-Cascade-RL-Math in the DAPO format. Eval data: AIME 2024 and 2025, through the repo's own
# prepare_dapo_data.sh, which also fetches DAPO-Math-17k. Then the base model into the HF cache, so the run does not
# download at step 0. The data is on the boot disk and is kept across a preemption. The cache is on the NVMe array and
# is not, so the model comes down again after one.
set -euo pipefail
MODEL_ID="${1:?usage: $0 <hf model id>}"
source "$(dirname "$0")/00_env.sh" >/dev/null
cd "$SKYRL_DIR"   # the example script and uv's project live here

if [[ -f "$EVAL_2024" && -f "$EVAL_2025" ]]; then
  echo "eval data already prepared"
else
  DATA_DIR="$EVAL_DIR" bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
if [[ -f "$NEMOTRON_FILE" ]]; then
  echo "training data already prepared"
else
  uv run --isolated --extra fsdp python "$TEACHER_KIT/prepare_nemotron_math.py" --output "$NEMOTRON_FILE"
fi
ls -la "$NEMOTRON_FILE" "$EVAL_2024" "$EVAL_2025"

echo "caching $MODEL_ID"
uv run --isolated --extra fsdp hf download "$MODEL_ID" >/dev/null
