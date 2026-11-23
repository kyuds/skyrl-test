#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. The warm-ups of 2c, 2e, 2f and 2g: SFT of M2RL-SFT on Qwen3-32B's correct answers
# (traces.sh), so the student starts closer to its teacher before OPD. See README.md in this folder.
#   2c  the warm-up at M2RL's own learning rate for this kind of SFT
#   2e  the light warm-up: the same, at a fifth of that learning rate
#   2f  the short-answer warm-up: 2c's, on only the answers of at most 8,192 tokens
#   2g  2c's, on answers sampled at temperature 1.0 (traces.sh 2g)
# From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh sft_2g_m2rl_sft_warmup_temperature_1   # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/2-alignment/sft_2c.sh 2c --smoke                      # 2 steps, nothing uploaded
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
variant="${1:?variant: 2c, 2e, 2f or 2g}"; shift
case "$variant" in
  2c) v2_init sft_2c_m2rl_sft_warmup "$@" ;;
  2e) v2_init sft_2e_m2rl_sft_light_warmup "$@" ;;
  2f) v2_init sft_2f_m2rl_sft_warmup_short_answers "$@" ;;
  2g) v2_init sft_2g_m2rl_sft_warmup_temperature_1 "$@" ;;
  *) echo "unknown variant: $variant (2c, 2e, 2f or 2g)" >&2; exit 2 ;;
esac

# 2f's change, a hyperparameter (in the block below too): only the answers (loss-masked tokens) of at most this many
# tokens. The filtered store is built from the full one on first use (filter_traces.py).
MAX_ANSWER=""
if [[ $variant == 2f ]]; then MAX_ANSWER=8192; fi
DATA="$SFT_2C"
if [[ $variant == 2g ]]; then DATA="$SFT_2G"; fi         # 2g's change: its own answers (traces.sh 2g)
if [[ -n $MAX_ANSWER ]]; then DATA="${SFT_2C}_answers_le${MAX_ANSWER}"; fi
if [[ $SMOKE == 1 ]]; then DATA="${DATA}_smoke"; fi     # traces.sh <variant> --smoke writes it
if [[ $V2_DRY_RUN != 1 && $SMOKE != 1 && -n $MAX_ANSWER && ! -f "$DATA.stats.json" && -f "$SFT_2C.stats.json" ]]; then
  (cd "$SKYRL_DIR" && uv run --isolated --extra fsdp python "$V2_KIT/2-alignment/filter_traces.py" --data "$SFT_2C" \
    --max-answer-tokens "$MAX_ANSWER" --out "$DATA")
fi
if [[ $V2_DRY_RUN != 1 && ! -f "$DATA.stats.json" ]]; then
  echo "no warm-up data at $DATA: run 2-alignment/traces.sh $([[ $variant == 2g ]] && echo 2g || echo 2c)$([[ $SMOKE == 1 ]] && echo ' --smoke') first" >&2
  exit 1
fi
ROWS=17000   # the dry run's guess; on a node, the rows traces.sh kept
if [[ -n $MAX_ANSWER ]]; then ROWS=6756; fi   # the dry run's guess: Qwen3-32B's token counts (2026-11-23)
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
SFT_LR=5e-5                        # 2c, 2f and 2g: [mtoff]
if [[ $variant == 2e ]]; then
  SFT_LR=1e-5                      # 2e: this kit's (README.md, "2e's light warm-up"). [mtoff]'s teachers are RL'd from
fi                                 # this student, Qwen3-32B is not: at 5e-5, 25 steps moved 2c's student further
                                   # than its 250 OPD steps at 1e-6 did, and it started at 26% looping answers
SFT_BATCH=512                      # [mtoff]
SFT_EPOCHS=1                       # ASSUMED: [mtoff] gives none
SFT_WARMUP=$(( (ROWS / SFT_BATCH + 9) / 10 ))   # [sft] 10% of the steps, rounded up
SFT_SCHEDULER=constant_with_warmup # ASSUMED: [mtoff] and [sft] give no decay
SFT_BETAS="[0.9,0.98]"             # ASSUMED: not given for SFT; [script]'s, M2RL's optimizer for the OPD that follows
SFT_WEIGHT_DECAY=0.1               # [sft]
SFT_GRAD_CLIP=1.0                  # slime's default (2-alignment/opd.sh, [slime])
SFT_MAX_LEN=18432                  # this kit's: a 2,048-token prompt + a 16,384-token answer, the OPD lengths
SFT_MAX_ANSWER="${MAX_ANSWER:-16384}" # 2f: 8,192, this kit's (README.md, "2f's short-answer warm-up"); else traces.sh's limit
# Loss on the answer tokens. ---- infrastructure, not hyperparameters ----
SFT_MICRO=1
SFT_CKPT_EVERY=10
SFT_HF_EVERY=1000                  # only the final export
SFT_TAGS="['sft','section2','2c','warmup','qwen3-32b-traces','M2RL-SFT']"
if [[ $variant == 2e ]]; then SFT_TAGS="['sft','section2','2e','light-warmup','lr1e-5','qwen3-32b-traces','M2RL-SFT']"; fi
if [[ $variant == 2f ]]; then SFT_TAGS="['sft','section2','2f','warmup-short-answers','answers-le8192','qwen3-32b-traces','M2RL-SFT']"; fi
if [[ $variant == 2g ]]; then SFT_TAGS="['sft','section2','2g','warmup-temperature-1','traces-temperature-1','qwen3-32b-traces','M2RL-SFT']"; fi
# =====================================================================================
v2_sft
