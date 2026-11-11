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
# As v2_launch: no HF token in the process (every teacher is public), Ray ships this directory to its workers.
exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp \
  python "$V2_KIT/teachers/serve.py" "${PORTS[@]}" --status-file "$OPD_STORE/teachers/$RUN.status" -- "${OVERRIDES[@]}"
