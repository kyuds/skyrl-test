#!/usr/bin/env bash
# Launch a teacher as one stand-alone `vllm serve` per GPU (round-robin from the client), wait until every
# server serves it, and record the URLs for the run scripts. Standalone: needs no sourced env.
#
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair post                 # Qwen3.5-9B on GPUs 0,1, ports 8100-8101
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair base                 # Qwen3.5-9B-Base on GPUs 0,1, ports 8000-8001
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair post --gpus 0        # one GPU only
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --model Qwen/Qwen3.5-4B --gpus 0,1 --port-base 9000
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --status                     # what is serving
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair post --stop          # kill that pair's servers
#   bash skyrl-test/opd-4xh100/02_serve_teacher.sh --stop-all
#
# Defaults come from the environment 00_env.sh exports (OPD_PAIR, OPD_TEACHER_GPUS, OPD_LOGS) and fall back
# to base / 0,1 / ~/logs. Each pair has its own default port range (base 8000+, post 8100+) so both can run
# at once. The URL list is written to $OPD_LOGS/teacher_<pair>.urls and the GPUs to teacher_<pair>.gpus;
# the run scripts read both (the second to refuse starting a student on a GPU a teacher holds).
#
# The OPD branch runs no preflight checks yet (see the TODO in main_opd.py), so this script does the two
# that matter for a vLLM teacher: the model is served, and max_model_len covers
# max_prompt_length (2048) + max_generate_length (8192) + 1.
set -euo pipefail

PAIR="${OPD_PAIR:-base}"; MODEL=""; GPUS_CSV="${OPD_TEACHER_GPUS:-0,1}"; PORT_BASE=""; ACTION="start"
LOGS="${OPD_LOGS:-$HOME/logs}"
MAX_MODEL_LEN="${OPD_TEACHER_MAX_MODEL_LEN:-10496}"   # >= 2048 + 8192 + 1
REQUIRED=$((2048 + 8192 + 1))
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pair) PAIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --stop) ACTION="stop"; shift ;;
    --stop-all) ACTION="stop-all"; shift ;;
    --status) ACTION="status"; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
case "$PAIR" in
  base) : "${MODEL:=${OPD_TEACHER_MODEL:-Qwen/Qwen3.5-9B-Base}}"; : "${PORT_BASE:=${OPD_TEACHER_PORT_BASE:-8000}}" ;;
  post) : "${MODEL:=${OPD_TEACHER_MODEL:-Qwen/Qwen3.5-9B}}";      : "${PORT_BASE:=${OPD_TEACHER_PORT_BASE:-8100}}" ;;
  *)    [[ -n "$MODEL" ]] || { echo "--pair must be base or post, or pass --model"; exit 2; }; : "${PORT_BASE:=9000}" ;;
esac
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
MANIFEST="$LOGS/teacher_${PAIR}.urls"
mkdir -p "$LOGS"

stop_pair() {  # $1 = pair tag
  local f
  for f in "$LOGS"/teacher_"$1"_gpu*.pid; do
    [[ -f "$f" ]] || continue
    kill "$(cat "$f")" 2>/dev/null && echo "stopped $(basename "$f" .pid) (pid $(cat "$f"))"; rm -f "$f"
  done
  rm -f "$LOGS/teacher_$1.urls" "$LOGS/teacher_$1.gpus"
}
case "$ACTION" in
  stop) stop_pair "$PAIR"; exit 0 ;;
  stop-all) for f in "$LOGS"/teacher_*.urls; do [[ -f "$f" ]] && stop_pair "$(basename "$f" .urls | sed 's/^teacher_//')"; done; exit 0 ;;
  status)
    for f in "$LOGS"/teacher_*.urls; do
      [[ -f "$f" ]] || { echo "no teacher manifests in $LOGS"; break; }
      echo "$(basename "$f" .urls): $(cat "$f")"
      for u in $(tr -d "[]'" < "$f" | tr ',' ' '); do
        if body=$(curl -sf "$u/v1/models" 2>/dev/null); then echo "  $u  UP  $(echo "$body" | tr -d '\n' | cut -c1-120)"; else echo "  $u  DOWN"; fi
      done
    done; exit 0 ;;
esac

# ---- start ----
URLS=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"; port=$((PORT_BASE + i)); tag="teacher_${PAIR}_gpu${gpu}"; log="$LOGS/$tag.log"; pid="$LOGS/$tag.pid"
  URLS+=("http://127.0.0.1:${port}")
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then echo "port $port already serving; leaving it"; continue; fi
  echo "serving $MODEL on GPU $gpu, port $port, max_model_len $MAX_MODEL_LEN; log: $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup uv run --isolated --extra fsdp vllm serve "$MODEL" \
    --port "$port" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    > "$log" 2>&1 &
  echo $! > "$pid"
done

# Readiness: poll every server's /v1/models (prime-rl's check) for up to 20 minutes.
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"; port=$((PORT_BASE + i)); tag="teacher_${PAIR}_gpu${gpu}"; log="$LOGS/$tag.log"; pid="$LOGS/$tag.pid"
  ready=0
  for _ in $(seq 1 240); do
    if body=$(curl -sf "http://127.0.0.1:${port}/v1/models" 2>/dev/null); then
      echo "$body" | uv run --isolated --extra fsdp python - "$MODEL" "$REQUIRED" "$port" <<'PY'
import json, sys
body = json.load(sys.stdin); model, required, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cards = {c["id"]: c for c in body.get("data", [])}
if model not in cards:
    sys.exit(f"port {port} does not serve {model!r}; it serves {list(cards)}")
mml = cards[model].get("max_model_len")
print(f"teacher ready on port {port}: {model} max_model_len={mml} (need >= {required})")
if mml is not None and mml < required:
    sys.exit("max_model_len too small for prompt + response + 1; pass --max-model-len")
PY
      ready=1; break
    fi
    if [[ -f "$pid" ]] && ! kill -0 "$(cat "$pid")" 2>/dev/null; then echo "teacher on GPU $gpu exited early; tail of $log:"; tail -30 "$log"; exit 1; fi
    sleep 5
  done
  [[ $ready == 1 ]] || { echo "teacher on port $port not ready after 20 min; tail of $log:"; tail -30 "$log"; exit 1; }
done
printf "['%s'" "${URLS[0]}" > "$MANIFEST"; for u in "${URLS[@]:1}"; do printf ",'%s'" "$u" >> "$MANIFEST"; done; printf "]\n" >> "$MANIFEST"
echo "$GPUS_CSV" > "$LOGS/teacher_${PAIR}.gpus"
echo "teacher '$PAIR' = $MODEL on GPUs $GPUS_CSV at $(cat "$MANIFEST")  (manifest: $MANIFEST)"
