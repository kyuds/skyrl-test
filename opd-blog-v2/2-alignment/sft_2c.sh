#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. 2c's warm-up: SFT of M2RL-SFT on Qwen3-32B's correct answers (traces.sh), so the
# student starts closer to its teacher before OPD. See README.md in this folder. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh sft_2c_m2rl_sft_warmup     # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/2-alignment/sft_2c.sh --smoke             # 2 steps, nothing uploaded
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init sft_2c_m2rl_sft_warmup "$@"

DATA="$SFT_2C"
if [[ $SMOKE == 1 ]]; then DATA="${SFT_2C}_smoke"; fi   # traces.sh --smoke writes it
if [[ $V2_DRY_RUN != 1 && ! -f "$DATA.stats.json" ]]; then
  echo "no warm-up data at $DATA: run 2-alignment/traces.sh$([[ $SMOKE == 1 ]] && echo ' --smoke') first" >&2
  exit 1
fi
ROWS=17000   # the dry run's guess; on a node, the rows traces.sh kept
if [[ $V2_DRY_RUN != 1 ]]; then ROWS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$DATA.stats.json")"; fi

# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table. No paper gives a warm-up recipe. The closest published one is M2RL's own
# SFT of M2RL-SFT on teacher answers, its multi-teacher off-policy distillation (MT-OFF), so this uses that:
#   [mtoff]  M2RL paper (https://arxiv.org/abs/2602.12566), appendix A, "Multi-teacher off-policy distillation":
#            "fine-tuned with the learning rate of 5e-5 and the batch size of 512"
#   [sft]    the same appendix, "Supervised fine-tuning" (how M2RL-SFT itself was trained): Adam, weight decay 0.1,
#            "10% training steps are used for learning rate warmup"
#   [script] https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L121-L127
SFT_BASE="$M2RL_SFT"               # already a chat model: its export ends turns at <|im_end|>
SFT_ROWS="$ROWS"
SFT_DATA="$DATA"                   # traces.sh: Qwen3-32B's answers, one per prompt, cut at SFT_MAX_LEN
SFT_LR=5e-5                        # [mtoff]
SFT_BATCH=512                      # [mtoff]
SFT_EPOCHS=1                       # ASSUMED: [mtoff] gives none
SFT_WARMUP=$(( (ROWS / SFT_BATCH + 9) / 10 ))   # [sft] 10% of the steps, rounded up
SFT_SCHEDULER=constant_with_warmup # ASSUMED: [mtoff] and [sft] give no decay
SFT_BETAS="[0.9,0.98]"             # ASSUMED: not given for SFT; [script]'s, M2RL's optimizer for the OPD that follows
SFT_WEIGHT_DECAY=0.1               # [sft]
SFT_GRAD_CLIP=1.0                  # slime's default (2-alignment/opd.sh, [slime])
SFT_MAX_LEN=18432                  # this kit's: a 2,048-token prompt + a 16,384-token answer, the OPD lengths
# Loss on the answer tokens. ---- infrastructure, not hyperparameters ----
SFT_MICRO=1
SFT_CKPT_EVERY=10
SFT_HF_EVERY=1000                  # only the final export
SFT_TAGS="['sft','section2','2c','warmup','qwen3-32b-traces','M2RL-SFT']"
# =====================================================================================
v2_sft
