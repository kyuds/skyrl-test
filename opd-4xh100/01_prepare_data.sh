#!/usr/bin/env bash
# GSM8K, the kit's task (Charlie, 2026-09-28: 0.8B is too small for DAPO): SkyRL's gsm8k_dataset.py writes
# train.parquet (the train split, 7,473 prompts) and validation.parquet (the test split, 1,319), plus a warm HF
# cache for the selected pair's student and teacher (OPD_PREPARE_ALL=1: both pairs) so no run downloads at
# step 0. The 9B teachers are 19 GB each.
#   bash skyrl-test/opd-4xh100/01_prepare_data.sh
set -euo pipefail
DATA="${OPD_DATA:-$HOME/data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_locate.sh"; opd_locate || exit 1
cd "$SKYRL_DIR"   # the example scripts and uv's project live here

uv run --isolated --extra fsdp examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA/gsm8k"
ls -la "$DATA/gsm8k"/train.parquet "$DATA/gsm8k"/validation.parquet

hf_dl() { if command -v hf >/dev/null 2>&1; then hf "$@"; else uv run --isolated --extra fsdp hf "$@"; fi; }
MODELS=("${OPD_STUDENT_MODEL:?source 00_env.sh first}" "${OPD_TEACHER_MODEL:?source 00_env.sh first}")
[[ "${OPD_PREPARE_ALL:-0}" == "1" ]] && MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B Qwen/Qwen3.5-0.8B-Base Qwen/Qwen3.5-9B-Base)
for m in "${MODELS[@]}"; do
  case "$m" in /*|~*) echo "local path $m; nothing to download";; *) echo "caching $m"; hf_dl download "$m" >/dev/null;; esac
done
echo "done: $DATA"
