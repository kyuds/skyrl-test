#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. Phase 1.1: OPD of M2RL-SFT from M2RL-RL_Math, thinking on, on M2RL's math set.
# See README.md in this folder. From ~/SkyRL:
#   nohup bash skyrl-test/opd-blog-v2/phase1-opd/run_m2rl.sh >> ~/opd-store/logs/p1_opd_m2rl_sft_from_rl_math.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-blog-v2/phase1-opd/run_m2rl.sh --smoke   # 2 tiny steps, W&B project opd-blog-v2-smoke
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run the same command again.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init p1_opd_m2rl_sft_from_rl_math "$@"
TRAIN_FILE="$MATH_TRAIN"

# 4 GPUs for the student (FSDP and its colocated vLLM engines), 4 for the teacher's vLLM engines.
STUDENT_GPUS=4
TEACHER_ENGINES=4
MAX_RESP=16384       # training answers; eval answers get 32768
v2_collect < <(
  v2_opd_recipe "$M2RL_SFT" "$STUDENT_GPUS" "$MAX_RESP" 32768 1 ""
  printf '%s\n' \
    trainer.teacher.backend=skyrl \
    "trainer.teacher.model=$M2RL_MATH" \
    "trainer.teacher.inference_engine.num_engines=$TEACHER_ENGINES" \
    trainer.teacher.inference_engine.tensor_parallel_size=1 \
    "trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=$((2048 + MAX_RESP + 1))" \
    trainer.teacher.max_concurrency=64
  v2_bookkeeping "['opd','phase1','nemotron3-nano-rl-math','M2RL-SFT','teacher:M2RL-RL_Math']"
)
v2_launch opd
