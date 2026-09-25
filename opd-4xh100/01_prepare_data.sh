#!/usr/bin/env bash
# DAPO-Math-17k (train) and AIME 2024 (eval), as the blog prepared them, plus a 256-row GSM8K eval slice
# (0.8B models sit near the floor on AIME24, so a second, easier eval set gives the curves some signal),
# plus a warm HF cache for the selected pair's student and teacher (OPD_PREPARE_ALL=1: both pairs) so no
# run downloads at step 0. The 9B teachers are 19 GB each.
#   bash skyrl-test/opd-4xh100/01_prepare_data.sh
set -euo pipefail
DATA="${OPD_DATA:-$HOME/data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_locate.sh"; opd_locate || exit 1
GSM8K_EVAL_ROWS="${GSM8K_EVAL_ROWS:-256}"
cd "$SKYRL_DIR"   # the example scripts and uv's project live here

DATA_DIR="$DATA/dapo" bash examples/train/algorithms/dapo/prepare_dapo_data.sh
ls -la "$DATA/dapo"/*cleaned*.parquet
uv run --isolated --extra fsdp examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA/gsm8k"
uv run --isolated --extra fsdp "$HERE/slice_parquet.py" "$DATA/gsm8k/validation.parquet" "$DATA/gsm8k/validation_${GSM8K_EVAL_ROWS}.parquet" --rows "$GSM8K_EVAL_ROWS"

hf_dl() { if command -v hf >/dev/null 2>&1; then hf "$@"; else uv run --isolated --extra fsdp hf "$@"; fi; }
MODELS=("${OPD_STUDENT_MODEL:?source 00_env.sh first}" "${OPD_TEACHER_MODEL:?source 00_env.sh first}")
[[ "${OPD_PREPARE_ALL:-0}" == "1" ]] && MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B Qwen/Qwen3.5-0.8B-Base Qwen/Qwen3.5-9B-Base)
for m in "${MODELS[@]}"; do
  case "$m" in /*|~*) echo "local path $m; nothing to download";; *) echo "caching $m"; hf_dl download "$m" >/dev/null;; esac
done
echo "done: $DATA"
