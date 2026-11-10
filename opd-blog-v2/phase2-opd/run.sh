#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. Phase 2: OPD of M2RL-SFT from Qwen3-32B in thinking mode, an 8x larger teacher of
# another post-training lineage, with everything else exactly as phase 1.1. Expected to break: see README.md in this
# folder. From ~/SkyRL:
#   nohup bash skyrl-test/opd-blog-v2/phase2-opd/run.sh >> ~/opd-store/logs/p2_opd_m2rl_sft_from_qwen3_32b.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-blog-v2/phase2-opd/run.sh --smoke   # 2 tiny steps, W&B project opd-blog-v2-smoke
# On an H100 node the 32B teacher needs two GPUs an engine: TEACHER_ENGINES=2 TEACHER_TP=2 in front of the command.
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run the same command again.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init p2_opd_m2rl_sft_from_qwen3_32b "$@"
TRAIN_FILE="$MATH_TRAIN"

STUDENT_GPUS=4
TEACHER_ENGINES="${TEACHER_ENGINES:-4}"   # a B200 holds the 32B model (66 GB in bf16) and its scoring on one GPU
TEACHER_TP="${TEACHER_TP:-1}"
MAX_RESP=16384
v2_collect < <(
  v2_opd_recipe "$M2RL_SFT" "$STUDENT_GPUS" "$MAX_RESP" 32768 1 ""
  printf '%s\n' \
    trainer.teacher.backend=skyrl \
    "trainer.teacher.model=$QWEN3_32B" \
    "trainer.teacher.inference_engine.num_engines=$TEACHER_ENGINES" \
    "trainer.teacher.inference_engine.tensor_parallel_size=$TEACHER_TP" \
    "trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=$((2048 + MAX_RESP + 1))" \
    trainer.teacher.max_concurrency=64
  v2_bookkeeping "['opd','phase2','nemotron3-nano-rl-math','M2RL-SFT','teacher:Qwen3-32B']"
)
v2_launch opd
