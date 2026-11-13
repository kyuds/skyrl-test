#!/usr/bin/env bash
# Runs on a teacher node, as the job serve_teachers_<node> (config.sh). Serves every model config.sh's
# V2_TEACHER_SERVERS puts on this node, with SkyRL's own teacher launch (teachers/serve.py), until it is stopped:
# the job never finishes. The OPD runs on other nodes score against it (README.md in this folder).
# From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/common/job.sh serve_teachers_h100_5     # as orchestrate.sh runs it
#   bash skyrl-test/opd-blog-v2/orchestrate.sh stop h100-5              # from the Mac: stops it
# --smoke is accepted and changes nothing: a served teacher is the same for a smoke run as for a real one.
set -euo pipefail
source "$(dirname "$0")/../common/run_lib.sh"
node="${1:?the node whose teachers to serve, e.g. h100-5}"; shift
v2_init "serve_teachers_${node//-/_}" "$@"
RUN="serve_teachers_${node//-/_}"   # the same with --smoke: a smoke run's OPD scores against the real servers

keys="$(echo "$V2_TEACHER_SERVERS" | awk -v n="$node" '$2 == n {print $1}')"
[[ -n "$keys" ]] || { echo "config.sh's V2_TEACHER_SERVERS puts no teacher on $node" >&2; exit 2; }
OVERRIDES=("trainer.log_path=$LOGS/skyrl/$RUN")
PORTS=()
gpus=0
for key in $keys; do
  model="$(v2_teacher_field "$key" 4)"; engines="$(v2_teacher_field "$key" 5)"; tp="$(v2_teacher_field "$key" 6)"
  OVERRIDES+=(
    "trainer.teachers.$key.backend=skyrl"
    "trainer.teachers.$key.model=$model"
    "trainer.teachers.$key.inference_engine.num_engines=$engines"
    "trainer.teachers.$key.inference_engine.tensor_parallel_size=$tp"
    "trainer.teachers.$key.inference_engine.engine_init_kwargs.max_model_len=$V2_TEACHER_MAX_LEN"
  )
  PORTS+=(--port "$key=$(v2_teacher_field "$key" 3)")
  gpus=$((gpus + engines * tp))
done
# Several teachers route by row label, so SkyRL wants a route key; the servers never see one.
OVERRIDES+=(trainer.algorithm.opd.route_key=skyrl_mopd_domain)
# The runs' top-K overlap metrics (config.sh): SkyRL's teacher launch then lets each server return V2_TOPK logprobs a
# position, and serve.py's preflight checks that it does.
OVERRIDES+=("trainer.algorithm.opd.topk_metrics.num_rows=$V2_TOPK_ROWS" "trainer.algorithm.opd.topk_metrics.top_k=$V2_TOPK")

if [[ $V2_DRY_RUN == 1 ]]; then
  echo "serve ${PORTS[*]}"
  printf '%s\n' "${OVERRIDES[@]}"
  exit 0
fi
have="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ $gpus -le $have ]] || { echo "$node's teachers need $gpus GPUs; it has $have" >&2; exit 2; }
echo "=== $RUN: $keys on $gpus of $have GPUs  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  %s\n' "${OVERRIDES[@]}"
cd "$SKYRL_DIR"
# Download first. SkyRL gives a server 600 s to answer /health, and on a fresh node (or after a preemption, which
# wipes the NVMe cache) Qwen3-32B's 64 GB download alone takes longer: it failed that way on 2026-11-11.
for key in $keys; do
  model="$(v2_teacher_field "$key" 4)"
  echo "downloading $model (nothing to do if it is in the cache)"
  uv run --isolated --extra fsdp hf download "$model" > /dev/null
done
# As v2_launch: no HF token in the process (every teacher is public), Ray ships this directory to its workers.
# 1800 s for a server to answer /health, not SkyRL's 600 (SkyRL passes SKYRL_* variables on to its Ray workers). On a
# node whose NVMe array is new or wiped by a preemption, the JIT and compile caches under ~/.cache are empty: a
# Qwen3-32B engine (TP 2) then spends about 7 minutes building FlashInfer's all-reduce kernel before it captures its
# CUDA graphs, and took 557 s to start on h100-7 on 2026-11-11, which with the launch before it missed the 600.
exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800 \
  uv run --isolated --extra fsdp \
  python "$V2_KIT/teachers/serve.py" "${PORTS[@]}" --status-file "$OPD_STORE/teachers/$RUN.status" -- "${OVERRIDES[@]}"
