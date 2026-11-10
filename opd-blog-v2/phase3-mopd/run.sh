#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. Phase 3: MOPD of M2RL-SFT from three M2RL experts, each row scored by the expert of
# its domain (math, code, instruction following), thinking on. See README.md in this folder. From ~/SkyRL:
#   nohup bash skyrl-test/opd-blog-v2/phase3-mopd/run.sh >> ~/opd-store/logs/p3_mopd_m2rl_sft_from_math_code_if.log 2>&1 < /dev/null &
#   bash skyrl-test/opd-blog-v2/phase3-mopd/run.sh --smoke   # 2 tiny steps, W&B project opd-blog-v2-smoke
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run the same command again.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init p3_mopd_m2rl_sft_from_math_code_if "$@"
TRAIN_FILE="$MOPD_TRAIN"

# 4 GPUs for the student, 4 for the teachers: math, the largest share of the rows (38%), gets two engines.
STUDENT_GPUS=4
MAX_RESP=16384
TEACHER_LEN=$((2048 + MAX_RESP + 1))
teacher() {   # $1 name in trainer.teachers (= the rows' skyrl_mopd_domain label), $2 model, $3 engines
  printf '%s\n' \
    "trainer.teachers.$1.backend=skyrl" \
    "trainer.teachers.$1.model=$2" \
    "trainer.teachers.$1.inference_engine.num_engines=$3" \
    "trainer.teachers.$1.inference_engine.tensor_parallel_size=1" \
    "trainer.teachers.$1.inference_engine.engine_init_kwargs.max_model_len=$TEACHER_LEN" \
    "trainer.teachers.$1.max_concurrency=$((32 * $3))"
}
v2_collect < <(
  v2_opd_recipe "$M2RL_SFT" "$STUDENT_GPUS" "$MAX_RESP" 32768 1 ""
  teacher math "$M2RL_MATH" 2
  teacher code "$M2RL_CODE" 1
  teacher if "$M2RL_IF" 1
  printf '%s\n' trainer.algorithm.opd.route_key=skyrl_mopd_domain
  v2_bookkeeping "['mopd','phase3','nemotron3-nano-rl-math-code-if','M2RL-SFT','teacher:M2RL-RL_Math','teacher:M2RL-RL_Coding','teacher:M2RL-RL_IF']"
)
v2_launch opd
