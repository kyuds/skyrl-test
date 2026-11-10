#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. Section 3: MOPD of M2RL-SFT from three M2RL experts, each row scored by the expert of
# its domain (math, code, instruction following), thinking on, with M2RL's MT-OPD recipe. See README.md here.
#   bash skyrl-test/opd-blog-v2/common/job.sh mopd_3_m2rl_sft_from_math_code_if    # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/3-mopd/run.sh --smoke                              # 2 tiny steps
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run it again; it resumes.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init mopd_3_m2rl_sft_from_math_code_if "$@"

# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table. M2RL's multi-teacher OPD (MT-OPD), as published, with three of its five
# teachers. The same block as 2-alignment/opd.sh:
#   [paper]  M2RL paper (https://arxiv.org/abs/2602.12566), appendix A, "Multi-teacher on-policy distillation": 300 steps,
#            batch size 256, group size 4, one update per 1024 rollouts, Adam at 1e-6; section 3.2: "the training prompts
#            are the same as the multi-task RLVR"
#   [script] https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L66-L127
#   [route]  https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/on_policy_distillation.py#L20-L25
#            (each row goes to its domain's teacher)
#   [slime]  https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/utils/arguments.py#L730-L731
#            (defaults the script leaves alone; rollout top-p 1.0 and top-k -1 at #L225-L228)
#   [loss]   https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/loss.py#L318-L337
#   [mean]   https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/cp_utils.py#L70-L76
#   [plan]   the proposal of 2026-11-10 (README.md at the kit root)
TRAIN_FILE="$MOPD_TRAIN"           # [plan] Nemotron3 math, code, IF: 57,800 prompts. [paper] used all five domains
PROMPTS=256                        # [paper] batch size 256; [script] --rollout-batch-size 256
SAMPLES=4                          # [paper] group size 4; [script] --n-samples-per-prompt 4
STEPS=300                          # [paper] 300 steps ([script] says --num-rollout 400; the paper's number is kept)
LR=1e-6                            # [paper]; [script] --lr 1e-6
WARMUP=0                           # [script] --lr-decay-style constant, no warmup
BETAS="[0.9,0.98]"                 # [script] --adam-beta1 0.9 --adam-beta2 0.98
WEIGHT_DECAY=0.1                   # [script] --weight-decay 0.1
GRAD_CLIP=1.0                      # [slime] --clip-grad default 1.0
MAX_PROMPT=2048                    # [script] --rollout-max-prompt-len 2048
MAX_RESP=16384                     # [script] --rollout-max-response-len 16384
TEMPERATURE=1.0                    # [script] --rollout-temperature 1.
TOP_P=1.0                          # [slime] default
TOP_K=-1                           # [slime] default
KL_COEF=1.0                        # [loss] advantage = teacher log-prob - student log-prob, no coefficient
LOSS=importance_sampling           # [paper] one update a step, so slime's PPO ratio is 1: the same gradient
LOSS_REDUCTION=sequence_mean       # [slime] --calculate-per-token-loss off; [mean] each rollout's mean, then the batch's
# ---- the in-training eval: this kit's; AIME only (the final eval runs all six benchmarks) ----
EVAL_EVERY=5
EVAL_N=8
EVAL_RESP=32768                    # M2RL appendix A, "Evaluation": 32K tokens
EVAL_TEMPERATURE=0.6               # same
EVAL_TOP_P=0.95                    # ASSUMED (0-evals/README.md)
EVAL_TOP_K=-1                      # none given
# ---- infrastructure, not hyperparameters ----
STUDENT_GPUS=4                     # the other 4 GPUs run the teachers
MICRO=1
# =====================================================================================

STUDENT="$M2RL_SFT"
# Math, the largest share of the rows (38%), gets two engines; code and IF one each.
teacher() {   # $1 name in trainer.teachers (= the rows' skyrl_mopd_domain label), $2 model, $3 engines
  printf '%s\n' \
    "trainer.teachers.$1.backend=skyrl" \
    "trainer.teachers.$1.model=$2" \
    "trainer.teachers.$1.inference_engine.num_engines=$3" \
    "trainer.teachers.$1.inference_engine.tensor_parallel_size=1" \
    "trainer.teachers.$1.inference_engine.engine_init_kwargs.max_model_len=$((MAX_PROMPT + MAX_RESP + 1))" \
    "trainer.teachers.$1.max_concurrency=$((32 * $3))"
}
v2_collect < <(
  v2_opd_recipe
  teacher math "$M2RL_MATH" 2
  teacher code "$M2RL_CODE" 1
  teacher if "$M2RL_IF" 1
  printf '%s\n' trainer.algorithm.opd.route_key=skyrl_mopd_domain
  v2_bookkeeping "['mopd','section3','nemotron3-nano-rl-math-code-if','M2RL-SFT','teacher:M2RL-RL_Math','teacher:M2RL-RL_Coding','teacher:M2RL-RL_IF']"
)
v2_launch opd
