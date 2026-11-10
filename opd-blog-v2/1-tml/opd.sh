#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. 1b: OPD of 1a's output (sft_1a_qwen3_8b_base_ot3, fetched from the Hub when not on
# this node) from Qwen3-8B in thinking mode, Thinking Machines' recipe. See README.md in this folder. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh opd_1b_ot3sft8b_from_qwen3_8b    # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/1-tml/opd.sh --smoke                            # 2 tiny steps, before 1a exists
# Anything else on the command line is passed to SkyRL as an extra override. Preempted? Run it again; it resumes.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init opd_1b_ot3sft8b_from_qwen3_8b "$@"

# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table. Thinking Machines' cookbook at 330b73d8d7 (2025-10-27, the blog's
# commit; its last Qwen3 version, 58e781e78e, has the same values) and the Tinker SDK of that week:
#   [blog]    https://thinkingmachines.ai/blog/on-policy-distillation/
#   [recipe]  https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/on_policy_distillation.py#L50-L72
#   [readme]  https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/README.md#L30-L39
#   [data]    https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/distillation/datasets.py#L170-L179 (and #L221)
#   [rl]      https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/rl/train.py#L116
#   [kl]      https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/distillation/train_on_policy.py#L81-L95
#   [tinker]  https://github.com/thinking-machines-lab/tinker/blob/2c9f564c9f7d9236a0659da89781c09c6509b950/src/tinker/types/sampling_params.py#L18-L24
#             https://github.com/thinking-machines-lab/tinker/blob/2c9f564c9f7d9236a0659da89781c09c6509b950/src/tinker/types/optim_step_request.py#L11-L21
#   [qwen3]   Qwen3 report, section 4.6: https://arxiv.org/abs/2505.09388
TEACHER="$QWEN3_8B"                # [blog] "We actually use Qwen3-8B as a teacher"; [recipe] teacher_model
TRAIN_FILE="$DEEPMATH_TRAIN"       # [recipe] dataset="deepmath"; [data] the question column alone
PROMPTS=512                        # [readme] groups_per_batch=512; [blog] 150 steps is about 77K prompts
SAMPLES=4                          # [recipe] group_size=4; [blog] 4 samples per prompt
STEPS=150                          # [blog] about 150 steps
LR=5e-5                            # [readme] "5e-5 with full finetuning" (1e-4 is for LoRA)
WARMUP=0                           # [rl] a constant rate, no schedule
BETAS="[0.9,0.95]"                 # [rl] AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
WEIGHT_DECAY=0                     # [tinker] AdamParams has no weight decay
GRAD_CLIP=1.0                      # not in [tinker]'s API, so unknown; kept as a guard (README.md, "Differences")
MAX_PROMPT=1024                    # [data] max_prompt_tokens=1024
MAX_RESP=4096                      # [recipe] max_tokens=4096
TEMPERATURE=1.0                    # [tinker] SamplingParams defaults; the recipe sets none
TOP_P=1.0                          # [tinker]
TOP_K=-1                           # [tinker]
KL_COEF=1.0                        # [recipe] kl_penalty_coef=1.0, kl_discount_factor=0.0; [kl] student - teacher log-prob
LOSS=importance_sampling           # [recipe] loss_fn="importance_sampling", num_substeps=1: one update a step
LOSS_REDUCTION=token_mean          # Tinker sums the loss over tokens: the same, up to a constant per step
# ---- the in-training eval: this kit's, with the Qwen settings 1b's final eval uses ----
EVAL_EVERY=5                       # every 5 steps, and before the first
EVAL_N=8                           # samples per AIME problem; 0-evals/ does the 64-sample evals
EVAL_RESP=32768                    # [qwen3] (the final evals give AIME 38,912)
EVAL_TEMPERATURE=0.6               # [qwen3]
EVAL_TOP_P=0.95                    # [qwen3]
EVAL_TOP_K=20                      # [qwen3]
# ---- infrastructure, not hyperparameters ----
STUDENT_GPUS=4                     # the other 4 GPUs run the teacher
TEACHER_ENGINES=4                  # 1 GPU each
MICRO=2                            # sequences per GPU per micro-batch
# =====================================================================================

if [[ $SMOKE == 1 ]]; then
  # The smoke test checks the plumbing before 1a has finished: the Base model with fixed settings stands in for 1a.
  STUDENT="$(v2_fix_base "$QWEN3_8B_BASE" "$QWEN3_8B" qwen3_8b_base_fixed)"
else
  STUDENT="$(v2_model kit:sft_1a_qwen3_8b_base_ot3)"
fi
v2_collect < <(
  v2_opd_recipe
  v2_teacher "$TEACHER" "$TEACHER_ENGINES" 1
  v2_bookkeeping "['opd','section1','1b','deepmath','teacher:Qwen3-8B']"
)
v2_launch opd
