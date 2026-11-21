#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. 2c's warm-up data: Qwen3-32B (thinking) answers every one of M2RL's 22,056 math
# prompts twice; per prompt, the first answer that is correct and finished within 16,384 tokens is kept, then
# tokenized for SFT of M2RL-SFT. See README.md in this folder. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh traces_2c_qwen3_32b    # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/2-alignment/traces.sh --smoke         # 16 prompts, 512 tokens; writes nothing for 2c
# Lands in ~/opd-store/evals/<job>/ (requests, responses, grades, traces.jsonl) and $SFT_2C (pretokenized).
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
v2_init traces_2c_qwen3_32b "$@"

# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table. No paper gives a warm-up recipe: the sizes here are this kit's.
#   [plan]   the proposal of 2026-11-10: "Collect Nemotron3 answers from Qwen3-32B"
#   [qwen3]  Qwen3 report (https://arxiv.org/abs/2505.09388), section 4.6: thinking mode at temperature 0.6, top-p 0.95,
#            top-k 20
MODEL="$QWEN3_32B"                 # [plan] 2b's teacher, so the warm-up moves the student toward it
PROMPT_FILE="$MATH_TRAIN"          # [plan] Nemotron3 math: M2RL's 22,056 prompts, the ones 2c's OPD trains on
N_PROMPTS=22056                    # all of them (the proposal's "20k" was a round number for this set)
SAMPLES=2                          # this kit's: a second try keeps more of the hard prompts
TEMPERATURE=0.6                    # [qwen3]
TOP_P=0.95                         # [qwen3]
TOP_K=20                           # [qwen3]
MAX_TOKENS=16384                   # this kit's: 2c's OPD rollout length, so the warm-up teaches answers that fit it
# Kept: per prompt, the first sample that is correct (common/kit_main.py's grader) and finished (closed its thinking
# and stopped at <|im_end|>). Tokenized with M2RL-SFT's tokenizer, cut at 18,432 tokens (a 2,048 prompt + 16,384).
# ---- infrastructure ----
TP="$(v2_tp_32b)"                  # 2 GPUs per engine on an H100, 1 on a B200
# =====================================================================================

OUT="$SFT_2C"
KEEP_ALL=0
if [[ $SMOKE == 1 ]]; then
  # 512 tokens is too short for any thinking answer to finish, so the smoke test keeps every prompt's first answer,
  # right or wrong, finished or not: it checks the plumbing through tokenizing, not the filter.
  N_PROMPTS=16; MAX_TOKENS=512; OUT="${SFT_2C}_smoke"; KEEP_ALL=1
fi
DIR="$EVALS/$RUN"
if [[ $V2_DRY_RUN == 1 ]]; then
  echo "traces $RUN: $MODEL (tp $TP) on $N_PROMPTS prompts of $PROMPT_FILE x $SAMPLES, temperature $TEMPERATURE, top-p $TOP_P, top-k $TOP_K, $MAX_TOKENS tokens -> $OUT"
  exit 0
fi
bash "$V2_KIT/data/prepare.sh"
if [[ $SMOKE == 1 ]]; then rm -rf "$DIR"; fi
mkdir -p "$DIR"
cd "$SKYRL_DIR"
[[ -f "$DIR/requests.jsonl" ]] || uv run --isolated --extra fsdp python "$V2_KIT/0-evals/benchmarks.py" prompts \
  --parquet "$PROMPT_FILE" --n "$N_PROMPTS" --samples "$SAMPLES" --seed 0 --out "$DIR"
[[ -f "$DIR/responses.jsonl" ]] || env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp \
  python "$V2_KIT/0-evals/generate.py" --model "$MODEL" --requests "$DIR/requests.jsonl" --tp "$TP" \
  --out "$DIR/responses.jsonl" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --max-tokens "$MAX_TOKENS"
uv run --isolated --extra fsdp --with math-verify==0.9.0 python "$V2_KIT/0-evals/grade.py" --dir "$DIR" --tools "$TOOLS"
uv run --isolated --extra fsdp python - "$DIR" "$KEEP_ALL" <<'PY'
import json, os, sys
d, keep_all = sys.argv[1], sys.argv[2] == "1"
requests = {r["id"]: r for r in map(json.loads, open(os.path.join(d, "requests.jsonl")))}
responses = {(r["id"], r["sample"]): r for r in map(json.loads, open(os.path.join(d, "responses.jsonl")))}
kept = {}
for g in sorted(map(json.loads, open(os.path.join(d, "graded.jsonl"))), key=lambda g: (g["id"], g["sample"])):
    if (keep_all or (g["correct"] and g["finish_reason"] == "stop")) and g["id"] not in kept:
        kept[g["id"]] = responses[(g["id"], g["sample"])]["response"]
with open(os.path.join(d, "traces.jsonl"), "w") as f:
    for rid, response in kept.items():
        question = requests[rid]["messages"][-1]["content"]
        f.write(json.dumps({"question": question, "response": response}) + "\n")
print(f"kept {len(kept)} of {len(requests)} prompts: one {'answer' if keep_all else 'correct, finished answer'} each")
if not kept:
    sys.exit("no answer was kept: nothing to train the warm-up on (check graded.jsonl and the cut rate)")
PY
uv run --isolated --extra fsdp python "$V2_KIT/data/prepare_sft.py" traces --traces "$DIR/traces.jsonl" \
  --tokenizer "$M2RL_SFT" --max-length 18432 --out "$OUT" $([[ $KEEP_ALL == 1 ]] && echo --keep-unfinished)
echo "=== $RUN done: $OUT ($(tr -d '\n ' < "$OUT.stats.json" | cut -c1-160))"
