#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs: 4 for the student, 4 for its teacher. Section 2's OPD: the student is M2RL-SFT on M2RL's math prompts, with M2RL's MT-OPD
# recipe and one teacher; only the teacher (and, for 2c, 2e to 2g, the warm-up; for 2d, the clip) changes. See README.md here.
#   2a  teacher M2RL-RL_Math: RL'd from this very student, so aligned
#   2b  teacher Qwen3-32B (thinking): stronger, but a different lineage, so misaligned
#   2c  as 2b, from the student after its warm-up on Qwen3-32B's answers (sft_2c_m2rl_sft_warmup)
#   2d  as 2b, with each token's teacher signal clipped below at CLIP_MIN nats
#   2e  as 2c, from the student after a light warm-up: the same SFT at a fifth of the learning rate (sft_2e_m2rl_sft_light_warmup)
#   2f  as 2c, from the student after a warm-up on only the answers of at most 8,192 tokens (sft_2f_m2rl_sft_warmup_short_answers)
#   2g  as 2c, from the student after a warm-up on answers sampled at temperature 1.0 (sft_2g_m2rl_sft_warmup_temperature_1)
# From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh opd_2b_m2rl_sft_from_qwen3_32b    # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/2-alignment/opd.sh 2b --smoke                   # 2 tiny steps
# Anything after the variant is passed to SkyRL as an extra override. Preempted? Run it again; it resumes.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
variant="${1:?variant: 2a to 2g}"; shift
case "$variant" in
  2a) v2_init opd_2a_m2rl_sft_from_m2rl_rl_math "$@" ;;
  2b) v2_init opd_2b_m2rl_sft_from_qwen3_32b "$@" ;;
  2c) v2_init opd_2c_m2rl_warmup_from_qwen3_32b "$@" ;;
  2d) v2_init opd_2d_m2rl_sft_from_qwen3_32b_clip "$@" ;;
  2e) v2_init opd_2e_m2rl_light_warmup_from_qwen3_32b "$@" ;;
  2f) v2_init opd_2f_m2rl_warmup_short_answers_from_qwen3_32b "$@" ;;
  2g) v2_init opd_2g_m2rl_warmup_temperature_1_from_qwen3_32b "$@" ;;
  *) echo "unknown variant: $variant (2a to 2g)" >&2; exit 2 ;;
esac

# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table. M2RL's multi-teacher OPD (MT-OPD), here with one teacher; section 3 uses
# the same block with three:
#   [paper]  M2RL paper (https://arxiv.org/abs/2602.12566), appendix A, "Multi-teacher on-policy distillation": 300 steps,
#            batch size 256, group size 4, one update per 1024 rollouts, Adam at 1e-6
#   [script] https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L66-L127
#   [slime]  https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/utils/arguments.py#L730-L731
#            (defaults the script leaves alone; rollout top-p 1.0 and top-k -1 at #L225-L228)
#   [loss]   https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/loss.py#L318-L337
#   [mean]   https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/cp_utils.py#L70-L76
#   [plan]   the proposal of 2026-11-10 (README.md at the kit root)
TRAIN_FILE="$MATH_TRAIN"           # [plan] Nemotron3 math: M2RL's 22,056 math prompts (data/README.md)
PROMPTS=256                        # [paper] batch size 256; [script] --rollout-batch-size 256
SAMPLES=4                          # [paper] group size 4; [script] --n-samples-per-prompt 4
STEPS=250                          # fewer than [script]'s --num-rollout 400 and [paper]'s 300, to save GPU time: 2c's
                                   # in-training AIME was flat from about step 110 (2026-11-20)
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
LOSS=importance_sampling           # [paper] one update per 1024 rollouts = one a step, so slime's PPO ratio is 1 and
                                   # its clip never acts: the same gradient as this loss
LOSS_REDUCTION=sequence_mean       # [slime] --calculate-per-token-loss off; [mean] each rollout's mean, then the batch's
CLIP_MIN=-2.3                      # 2d only; this kit's: about the 1st percentile of 2a's pair (README.md, "2d")
# ---- the in-training eval: this kit's, with the M2RL settings the section's final evals use ----
EVAL_EVERY=5                       # every 5 steps, and before the first
EVAL_N=8                           # samples per AIME problem; 0-evals/ does the 64-sample evals
EVAL_RESP=32768                    # M2RL appendix A, "Evaluation": 32K tokens
EVAL_TEMPERATURE=0.6               # same
EVAL_TOP_P=0.95                    # ASSUMED (0-evals/README.md)
EVAL_TOP_K=-1                      # none given
# ---- infrastructure, not hyperparameters ----
STUDENT_GPUS=4                     # half the node; the teacher runs on the other half (below). One node per run, so
                                   # section 2's runs share the three nodes left (config.sh)
MICRO=1                            # sequences per GPU per micro-batch
# =====================================================================================

STUDENT="$M2RL_SFT"
case "$variant" in
  2a)    TEACHER="$M2RL_MATH" ;;
  2b|2d) TEACHER="$QWEN3_32B" ;;
  2c)    TEACHER="$QWEN3_32B"; STUDENT="$(v2_model kit:sft_2c_m2rl_sft_warmup)" ;;
  2e)    TEACHER="$QWEN3_32B"; STUDENT="$(v2_model kit:sft_2e_m2rl_sft_light_warmup)" ;;
  2f)    TEACHER="$QWEN3_32B"; STUDENT="$(v2_model kit:sft_2f_m2rl_sft_warmup_short_answers)" ;;
  2g)    TEACHER="$QWEN3_32B"; STUDENT="$(v2_model kit:sft_2g_m2rl_sft_warmup_temperature_1)" ;;
esac
# The teacher's engines fill the node's other GPUs, one GPU each, except Qwen3-32B on an H100, which needs two.
TEACHER_TP=1
if [[ "$TEACHER" == "$QWEN3_32B" ]]; then TEACHER_TP="$(v2_tp_32b)"; fi
EXTRA_OPD=()
if [[ $variant == 2d ]]; then EXTRA_OPD+=("trainer.algorithm.opd.advantage_clip_min=$CLIP_MIN"); fi
EXTRA_TAGS=""
if [[ $variant == 2e ]]; then EXTRA_TAGS=",'light-warmup','warmup-lr1e-5'"; fi
if [[ $variant == 2f ]]; then EXTRA_TAGS=",'warmup-short-answers','warmup-answers-le8192'"; fi
if [[ $variant == 2g ]]; then EXTRA_TAGS=",'warmup-temperature-1','warmup-traces-temperature-1'"; fi

v2_collect < <(
  v2_opd_recipe
  v2_teacher "$TEACHER" $(((8 - STUDENT_GPUS) / TEACHER_TP)) "$TEACHER_TP"
  v2_bookkeeping "['opd','section2','$variant','nemotron3-nano-rl-math','teacher:${TEACHER#*/}'$EXTRA_TAGS]"
  printf '%s\n' ${EXTRA_OPD[@]+"${EXTRA_OPD[@]}"}
)
v2_launch opd
