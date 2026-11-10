#!/usr/bin/env bash
# Runs on the VM, on all 8 GPUs. One eval: build the requests, sample them with vLLM, grade, and log the scores to W&B
# (project opd-blog-v2-evals). See README.md in this folder. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh eval_qwen3_8b_math       # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/0-evals/run.sh <job> <model> <preset> <benches> [--smoke]
#     <model>    a Hub id, kit:<SFT job> (its output, from the Hub if not on this node), run:<job> (its last export)
#     <preset>   qwen or m2rl: the authors' generation settings, in HYPERPARAMETERS below
#     <benches>  comma-separated names or sets: math (aime24, aime25), code (lcb_v5, lcb_v6), if (ifeval, ifbench), all
#   --smoke: 2 samples a problem, 512-token answers, W&B project opd-blog-v2-smoke. It checks the plumbing, not scores.
# Everything lands in ~/opd-store/evals/<job>/; a rerun reuses whatever step already finished.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
job="${1:?usage: run.sh <job> <model> <preset> <benches> [--smoke]}"
spec="${2:?model}"; preset="${3:?preset: qwen or m2rl}"; benches="${4:?benches}"
shift 4
v2_init "$job" "$@"
# ================================== HYPERPARAMETERS ==================================
# README.md in this folder has the same table.
#   [qwen3]  Qwen3 report (https://arxiv.org/abs/2505.09388), section 4.6: thinking mode at temperature 0.6, top-p 0.95,
#            top-k 20; 32,768 tokens, except AIME'24 and AIME'25 at 38,912; AIME scored over 64 samples a problem
#   [m2rl]   M2RL paper (https://arxiv.org/abs/2602.12566), appendix A, "Evaluation": 32K tokens, temperature 0.6,
#            Avg@K with K = 64 for AIME, 8 for LiveCodeBench v5, 1 for v6, 4 for IFEval, 1 for IFBench. Top-p is not
#            given for these evals; 0.95 is ASSUMED from the same appendix's GenRM settings.
# The samples per problem (K) are in benchmarks.py (BENCH_SAMPLES), the same for both presets.
case "$preset" in
  qwen) TEMP=0.6                   # [qwen3]
        TOP_P=0.95                 # [qwen3]
        TOP_K=20                   # [qwen3]
        MAX_TOKENS=32768           # [qwen3]
        # One length per job: AIME's for a math-only job ([qwen3]), 32,768 for anything else.
        if [[ $benches == math || $benches == aime24 || $benches == aime25 || $benches == aime24,aime25 ]]; then
          MAX_TOKENS=38912         # [qwen3]
        fi ;;
  m2rl) TEMP=0.6                   # [m2rl]
        TOP_P=0.95                 # ASSUMED (see above)
        TOP_K=-1                   # [m2rl] none given
        MAX_TOKENS=32768 ;;        # [m2rl] "32K"
  *) echo "unknown preset: $preset (qwen or m2rl)" >&2; exit 2 ;;
esac
# Thinking on for both: each model's own chat template, enable_thinking=True ([qwen3], [m2rl]).
# ---- infrastructure ----
TP=1                               # GPUs per vLLM engine; Qwen3-32B needs 2 on an H100 (common/run_lib.sh, v2_tp_32b)
if [[ "$spec" == "$QWEN3_32B" ]]; then TP="$(v2_tp_32b)"; fi
# =====================================================================================
SAMPLES_ARG=()
if [[ $SMOKE == 1 ]]; then MAX_TOKENS=512; SAMPLES_ARG=(--samples 2); fi
DIR="$EVALS/$RUN"
# The graders, at the commits this kit was tested with on 2026-11-09.
LCB_REPO=https://github.com/LiveCodeBench/LiveCodeBench.git; LCB_COMMIT=28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
IFB_REPO=https://github.com/allenai/IFBench.git;              IFB_COMMIT=cd189a9c2889526d8af901d2103fd3e1ffcea6df
GRADER_DEPS=(--with math-verify==0.9.0 --with absl-py==2.3.1 --with langdetect==1.0.9 --with nltk==3.9.2
             --with immutabledict==4.2.2 --with emoji==2.15.0 --with syllapy==0.8.0)
if [[ $V2_DRY_RUN == 1 ]]; then
  echo "eval $RUN: model=$(v2_model "$spec") preset=$preset benches=$benches temperature=$TEMP top_p=$TOP_P top_k=$TOP_K max_tokens=$MAX_TOKENS tp=$TP"
  exit 0
fi

tool() {   # $1 folder under $TOOLS, $2 repo, $3 commit
  if [[ "$(git -C "$TOOLS/$1" rev-parse HEAD 2>/dev/null || true)" != "$3" ]]; then
    rm -rf "${TOOLS:?}/$1"
    git clone -q "$2" "$TOOLS/$1"
    git -C "$TOOLS/$1" checkout -q "$3"
  fi
}
bash "$V2_KIT/data/prepare.sh"
tool LiveCodeBench "$LCB_REPO" "$LCB_COMMIT"
tool IFBench "$IFB_REPO" "$IFB_COMMIT"
MODEL="$(v2_model "$spec")"
if [[ $SMOKE == 1 ]]; then rm -rf "$DIR"; fi
mkdir -p "$DIR"
echo "=== eval $RUN  model=$MODEL ($spec)  preset=$preset  benches=$benches  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$SKYRL_DIR"

if [[ ! -f "$DIR/requests.jsonl" ]]; then
  uv run --isolated --extra fsdp python "$V2_KIT/0-evals/benchmarks.py" bench --benches "$benches" \
    --data "$V2_DATA" --tools "$TOOLS" --out "$DIR" ${SAMPLES_ARG[@]+"${SAMPLES_ARG[@]}"}
fi
if [[ ! -f "$DIR/responses.jsonl" ]]; then
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp python "$V2_KIT/0-evals/generate.py" \
    --model "$MODEL" --requests "$DIR/requests.jsonl" --out "$DIR/responses.jsonl" \
    --temperature "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --max-tokens "$MAX_TOKENS" --tp "$TP"
fi
uv run --isolated --extra fsdp "${GRADER_DEPS[@]}" python "$V2_KIT/0-evals/grade.py" --dir "$DIR" --tools "$TOOLS"

# The scores on W&B, so they can be read from anywhere. Logging must not fail a finished eval.
EVAL_PROJECT="${V2_PROJECT}-evals"; [[ $SMOKE == 0 ]] || EVAL_PROJECT="${V2_PROJECT}-smoke"
uv run --isolated --extra fsdp python - "$DIR/scores.json" "$EVAL_PROJECT" "$RUN" "$spec" "$MODEL" "$preset" \
  "$TEMP" "$TOP_P" "$TOP_K" "$MAX_TOKENS" <<'EOF' || echo "warning: could not log the scores to W&B; they are in $DIR/scores.json"
import json, sys
import wandb
scores_path, project, name, spec, model, preset, temp, top_p, top_k, max_tokens = sys.argv[1:]
scores = json.load(open(scores_path))
run = wandb.init(project=project, name=name, job_type="eval", config={
    "model": spec, "model_path": model, "preset": preset, "temperature": float(temp), "top_p": float(top_p),
    "top_k": int(top_k), "max_tokens": int(max_tokens)})
run.summary.update({f"{bench}/{key}": value for bench, s in scores.items() for key, value in s.items()})
run.finish()
EOF
echo "=== $RUN done: $DIR/scores.json"
