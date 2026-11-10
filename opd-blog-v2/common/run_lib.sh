# Sourced by every run script. Runs on the VM; with V2_DRY_RUN=1 it also runs on your Mac and only prints the
# overrides, which is how common/check_configs.sh checks them against SkyRL's real config classes.
#
#   v2_init <run name> "$@"   parses --smoke, sets RUN, PROJECT and the paths, and on the VM loads 00_env.sh
#   v2_bookkeeping            prints the overrides every run shares: logging, checkpoints, exports, resume
#   v2_opd_recipe             prints the OPD recipe every phase shares (see the phase READMEs)
#   v2_launch grpo|opd ...    runs common/kit_main.py with the overrides, then marks the run done
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash"; return 1 2>/dev/null || exit 1; }
V2_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DRY_RUN="${V2_DRY_RUN:-0}"

v2_init() {
  RUN="$1"; shift
  SMOKE=0
  EXTRA=()
  for arg in "$@"; do
    if [[ "$arg" == --smoke ]]; then SMOKE=1; else EXTRA+=("$arg"); fi
  done
  if [[ $V2_DRY_RUN == 1 ]]; then
    # Paths as they are on a VM, so the printed overrides read like the real ones. Nothing is touched.
    V2_KIT="$(cd "$V2_LIB_DIR/.." && pwd)"
    source "$V2_KIT/config.sh"
    OPD_STORE=/home/user/opd-store; LOGS="$OPD_STORE/logs"; CKPT_ROOT="$OPD_STORE/ckpts"; EXPORT_ROOT="$OPD_STORE/exports"
    V2_DATA="$OPD_STORE/data/v2"
    MATH_TRAIN="$V2_DATA/math_train.parquet"; MOPD_TRAIN="$V2_DATA/mopd_train.parquet"
    AIME24="$V2_DATA/aime24.parquet"; AIME25="$V2_DATA/aime25.parquet"
  else
    source "$V2_LIB_DIR/00_env.sh"
  fi
  PROJECT="$V2_PROJECT"
  if [[ $SMOKE == 1 ]]; then
    RUN="smoke_$RUN"
    PROJECT="${V2_PROJECT}-smoke"   # keeps the real project to the real runs
  fi
}

# Logging, checkpoints, exports and resume: the same for every run. A checkpoint every 10 steps (the last 2 kept)
# bounds what a preemption costs; an HF export every 50 steps keeps a usable model near any eval, which the teacher
# runs of 2026-11 lacked when their best step fell between exports.
v2_bookkeeping() {
  local tags="$1"
  printf '%s\n' \
    trainer.logger=wandb \
    "trainer.project_name=$PROJECT" \
    "trainer.run_name=$RUN" \
    "trainer.tags=$tags" \
    "trainer.ckpt_path=$CKPT_ROOT/$RUN" \
    "trainer.export_path=$EXPORT_ROOT/$RUN" \
    "trainer.log_path=$LOGS/skyrl/$RUN" \
    trainer.ckpt_interval=10 \
    trainer.max_ckpts_to_keep=2 \
    trainer.hf_save_interval=50 \
    trainer.resume_mode=latest \
    trainer.eval_before_train=true \
    trainer.eval_interval=25 \
    trainer.eval_batch_size=1024 \
    trainer.epochs=20
}

# The OPD recipe every phase shares: M2RL's multi-teacher OPD (arXiv 2602.12566, appendix A, and its
# slime/scripts/multi_teacher_opd.sh) on SkyRL's OPD defaults. Arguments:
#   $1 student model   $2 student GPUs   $3 max response tokens (training)   $4 max response tokens (eval)
#   $5 micro batch per GPU (training)    $6 chat_template_kwargs override or ""
v2_opd_recipe() {
  local student="$1" gpus="$2" max_resp="$3" eval_resp="$4" micro="$5" template_kwargs="$6"
  local max_prompt=2048
  printf '%s\n' \
    "trainer.policy.model.path=$student" \
    "data.train_data=['$TRAIN_FILE']" \
    "data.val_data=['$AIME24','$AIME25']" \
    environment.env_class=opd_v2_math \
    trainer.strategy=fsdp \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_nodes=1 \
    "trainer.placement.policy_num_gpus_per_node=$gpus" \
    "generator.inference_engine.num_engines=$gpus" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    "generator.inference_engine.engine_init_kwargs.max_model_len=$((max_prompt + eval_resp + 1))" \
    generator.batched=false \
    trainer.train_batch_size=256 \
    generator.n_samples_per_prompt=4 \
    trainer.policy_mini_batch_size=256 \
    trainer.update_epochs_per_batch=1 \
    trainer.max_training_steps=300 \
    "trainer.max_prompt_length=$max_prompt" \
    "generator.sampling_params.max_generate_length=$max_resp" \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.top_k=-1 \
    "generator.eval_sampling_params.max_generate_length=$eval_resp" \
    generator.eval_sampling_params.temperature=0.6 \
    generator.eval_sampling_params.top_p=0.95 \
    generator.eval_n_samples_per_prompt=16 \
    trainer.algorithm.opd.kl_coef=1.0 \
    trainer.algorithm.opd.use_task_reward=false \
    trainer.algorithm.policy_loss_type=importance_sampling \
    trainer.algorithm.loss_reduction=token_mean \
    "trainer.algorithm.max_seq_len=$((max_prompt + max_resp))" \
    trainer.algorithm.use_kl_loss=false \
    trainer.policy.optimizer_config.lr=1e-6 \
    trainer.policy.optimizer_config.num_warmup_steps=0 \
    trainer.policy.optimizer_config.weight_decay=0.1 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    "trainer.micro_train_batch_size_per_gpu=$micro" \
    "trainer.micro_forward_batch_size_per_gpu=$((micro * 2))"
  if [[ -n "$template_kwargs" ]]; then printf '%s\n' "$template_kwargs"; fi
}

# Tiny settings for --smoke: 2 steps of 8 prompts x 2 samples, 512-token answers, one eval sample per problem.
# It checks that the models load, the teachers answer and the step finishes; it says nothing about the recipe.
v2_smoke_overrides() {
  printf '%s\n' \
    trainer.train_batch_size=8 \
    generator.n_samples_per_prompt=2 \
    trainer.policy_mini_batch_size=8 \
    trainer.max_training_steps=2 \
    generator.sampling_params.max_generate_length=512 \
    generator.eval_sampling_params.max_generate_length=512 \
    generator.eval_n_samples_per_prompt=1 \
    trainer.eval_interval=2 \
    trainer.ckpt_interval=2 \
    trainer.hf_save_interval=2 \
    trainer.resume_mode=none \
    trainer.policy.optimizer_config.num_warmup_steps=0 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.micro_forward_batch_size_per_gpu=1
}

# Reads newline-separated overrides on stdin into OVERRIDES, adding the smoke settings and the extra arguments.
v2_collect() {
  OVERRIDES=()
  local line
  while IFS= read -r line; do [[ -n "$line" ]] && OVERRIDES+=("$line"); done
  if [[ $SMOKE == 1 ]]; then
    while IFS= read -r line; do OVERRIDES+=("$line"); done < <(v2_smoke_overrides)
  fi
  OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})
}

v2_launch() {
  local mode="$1"
  if [[ $V2_DRY_RUN == 1 ]]; then
    echo "$mode"
    printf '%s\n' "${OVERRIDES[@]}"
    return 0
  fi
  if [[ $SMOKE == 0 && -f "$LOGS/$RUN.done" ]]; then
    echo "$RUN already finished ($(cat "$LOGS/$RUN.done")); not training again. Delete $LOGS/$RUN.done to retrain."
    return 0
  fi
  if [[ $SMOKE == 1 ]]; then rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN" "$LOGS/$RUN.smoke_ok"; fi
  bash "$V2_KIT/data/prepare.sh"
  echo "=== run=$RUN  project=$PROJECT  mode=$mode  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  %s\n' "${OVERRIDES[@]}"
  # HF_TOKEN is kept out of the training process: SkyRL logs the value of what it copies into the Ray runtime env, and
  # training needs no token (every model is public). Only common/upload_export.sh uses it.
  cd "$SKYRL_DIR"   # uv resolves the project, and Ray ships this directory to its workers
  # math-verify 0.9.0, the version tested; SkyRL's lock holds its ANTLR runtime at 4.9.3 (through omegaconf), the
  # parser common/kit_main.py's grader was checked against.
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp --with math-verify==0.9.0 \
    python "$V2_KIT/common/kit_main.py" "$mode" "${OVERRIDES[@]}"
  if [[ $SMOKE == 1 ]]; then
    rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/$RUN.smoke_ok"
    echo "=== $RUN passed"
  else
    date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/$RUN.done"
    echo "=== $RUN finished training at $(cat "$LOGS/$RUN.done"). Exports: $EXPORT_ROOT/$RUN"
  fi
}
