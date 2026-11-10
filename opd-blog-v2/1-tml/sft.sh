#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. 1a: SFT of Qwen3-8B-Base on 384,000 OpenThoughts-3 rows, Thinking Machines'
# recipe. See README.md in this folder. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh sft_1a_qwen3_8b_base_ot3      # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/1-tml/sft.sh --smoke                         # 2 steps on 512 rows, nothing uploaded
# Anything else on the command line is passed to SkyRL's SFT config as an extra override. Preempted? Run it again:
# it resumes from its last checkpoint.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init sft_1a_qwen3_8b_base_ot3 "$@"

# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table. Thinking Machines' cookbook at 330b73d8d7 (2025-10-27, the blog's
# commit; the current cookbook uses Qwen3.5, and its last Qwen3 version, 58e781e78e, has the same values):
#   [blog]    https://thinkingmachines.ai/blog/on-policy-distillation/
#   [recipe]  https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/off_policy_reasoning.py#L87-L101
#   [readme]  https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/README.md#L17
#   [train]   https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/supervised/train.py#L49-L69
#   [sched]   https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/utils/lr_scheduling.py#L11
#   [tinker]  https://github.com/thinking-machines-lab/tinker/blob/2c9f564c9f7d9236a0659da89781c09c6509b950/src/tinker/types/optim_step_request.py#L11-L21
BASE_MODEL="$QWEN3_8B_BASE"        # [blog] the student; [recipe] model_name="Qwen/Qwen3-8B-Base"
OT3_DOMAINS=""                     # [recipe] streams every domain (math, code, science); "" keeps all, "math" only math
OT3_ROWS=384000                    # [recipe] max_prompts = 128 * 3000
SFT_MAX_LEN=16384                  # [recipe] max_length=16384; longer rows are cut
SFT_BATCH=128                      # [recipe] batch_size=128, so 3000 steps
SFT_EPOCHS=1                       # [recipe] num_epochs=1
SFT_LR=1e-4                        # [readme] "1e-4 with full finetuning" (the recipe's 1e-3 default is for LoRA)
SFT_SCHEDULER=linear               # [recipe] lr_schedule="linear"; [sched] 1 - step/total, down to 0
SFT_WARMUP=0                       # [train] none
SFT_BETAS="[0.9,0.95]"             # [train] adam_beta1=0.9, adam_beta2=0.95; adam_eps=1e-8 is torch's default too
SFT_WEIGHT_DECAY=0                 # [tinker] AdamParams has no weight decay
SFT_GRAD_CLIP=1.0                  # not in [tinker]'s API, so unknown; kept as a guard (README.md, "Differences")
# Loss on the answer tokens, summed ([recipe] TrainOnWhat.ALL_ASSISTANT_MESSAGES; SkyRL's SFT loss is the same sum).
# ---- infrastructure, not hyperparameters ----
SFT_MICRO=2                        # rows per GPU per micro-batch, packed
SFT_CKPT_EVERY=250
SFT_HF_EVERY=1000                  # exports at 1000, 2000 and 3000: the SFT curve, not only its end
SFT_TAGS="['sft','section1','1a','openthoughts3','Qwen3-8B-Base']"
# =====================================================================================

# The SFT starts from Qwen3-8B-Base with Qwen3-8B's settings files: tokenizer, end tokens, 40,960 context
# (common/fix_base_settings.py says why). SkyRL's export keeps those files, so 1a's output ends its turns at <|im_end|>.
SFT_BASE="$(v2_fix_base "$BASE_MODEL" "$QWEN3_8B" qwen3_8b_base_fixed)"
SFT_DATA="$V2_DATA/sft_ot3_${OT3_DOMAINS:-all}_${OT3_ROWS}"
SFT_DATA="${SFT_DATA//,/_}"
if [[ $SMOKE == 1 ]]; then SFT_DATA="$V2_DATA/sft_ot3_smoke"; fi
if [[ $V2_DRY_RUN != 1 ]]; then
  if [[ $SMOKE == 1 ]]; then
    bash "$V2_KIT/data/prepare.sh" sft-ot3-smoke "$SFT_DATA"
  else
    bash "$V2_KIT/data/prepare.sh" sft-ot3 "$SFT_DATA" "$OT3_DOMAINS" "$OT3_ROWS"
  fi
fi
v2_sft
