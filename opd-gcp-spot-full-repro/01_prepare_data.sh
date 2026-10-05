#!/usr/bin/env bash
# The blog's data and models, fetched once: DAPO-Math-17k (train) and AIME 2024 (eval) through the repo's own
# examples/train/algorithms/dapo/prepare_dapo_data.sh, which also drops the duplicated rows and writes the
# *-cleaned.parquet files the runs read; then the two base models into the HF cache, so no run downloads at
# step 0. Nothing here needs a token (public datasets and models).
#   nohup bash skyrl-test/opd-gcp-spot-full-repro/01_prepare_data.sh > ~/opd-store/logs/prepare.log 2>&1 < /dev/null &
# The data goes to the boot disk ($OPD_DATA) and survives a preemption; the model cache is on the NVMe array
# and does not, so run this again after one (the data step is quick the second time, the models re-download).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$SKYRL_DIR"   # the example script and uv's project live here

if [[ -f "$TRAIN_FILE" && -f "$EVAL_FILE" ]]; then
  echo "data already prepared: $DATA_DIR"
else
  DATA_DIR="$DATA_DIR" bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
ls -la "$TRAIN_FILE" "$EVAL_FILE"

for m in Qwen/Qwen3-4B-Base Qwen/Qwen3-1.7B-Base; do
  echo "caching $m"
  uv run --isolated --extra fsdp hf download "$m" >/dev/null
done
echo "done: data in $DATA_DIR, models in the HF cache"
