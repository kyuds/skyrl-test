# Shared by 02_run_dapo.sh, 03_run_opd.sh and upload_export.sh. Holds what the DAPO runs and the OPD runs have in
# common in the blog's recipe, so the two cannot drift: the task (DAPO-Math-17k prompts, AIME 2024 eval,
# prompt 2048 / response 8192), the batch shape (512 prompts x 16 samples), the sampling settings and the eval
# cadence. The values are the repo's own scripts for the blog runs, flag for flag:
#   examples/train/algorithms/dapo/run_dapo_aime_qwen3_4b_aime.sh, run_dapo_qwen3_1.7b_aime.sh
#   examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_{4b,1.7b}.sh
# What differs between the two methods (loss, LR, mini-batching, teacher) is in the two run scripts.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_locate.sh"; opd_locate || exit 1
: "${OPD_STORE:?source 00_env.sh first}"
PROJECT="${OPD_PROJECT:-opd_gcp_spot_full_repro}"
LOGS="${OPD_LOGS:-$OPD_STORE/logs}"
DATA_DIR="${OPD_DATA:-$OPD_STORE/data}/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
EVAL_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
# Checkpoints (resume only) and HF exports (the teacher, and what gets uploaded). Both on the boot disk by
# default. OPD_CKPT_ROOT may also be a gs:// path: SkyRL writes checkpoints to GCS directly, and that survives
# losing the VM altogether. Exports stay local: vLLM loads the teacher from them and `hf upload` reads them.
CKPT_ROOT="${OPD_CKPT_ROOT:-$OPD_STORE/ckpts/$PROJECT}"
EXPORT_ROOT="$OPD_STORE/exports/$PROJECT"
mkdir -p "$LOGS"

# 4b | 1.7b -> MODEL_ID (the base model a run starts from) and MODEL_TAG (its name in run names)
opd_model() {
  case "${1:-}" in
    4b)        MODEL_ID="Qwen/Qwen3-4B-Base";   MODEL_TAG="4b" ;;
    1.7b|1p7b) MODEL_ID="Qwen/Qwen3-1.7B-Base"; MODEL_TAG="1p7b" ;;
    *) return 1 ;;
  esac
}

opd_check_data() {
  local f
  for f in "$TRAIN_FILE" "$EVAL_FILE"; do
    [[ -f "$f" ]] || { echo "missing $f: run skyrl-test/opd-gcp-spot-full-repro/01_prepare_data.sh first" >&2; return 1; }
  done
}

# ---- the task and batch shape, shared by DAPO and OPD ----
# Checkpoint every 5 steps instead of the repo scripts' 10 (it changes no result; on a spot VM it halves what a
# preemption costs), keeping the last two. HF exports every 10 steps as in the repo scripts: the teacher is
# the DAPO run's step-90 export, and the last step of a run is always exported.
TASK_OPTS=(
  data.train_data="['$TRAIN_FILE']"
  data.val_data="['$EVAL_FILE']"
  environment.env_class=aime
  trainer.max_prompt_length=2048
  generator.sampling_params.max_generate_length=8192
  generator.eval_sampling_params.max_generate_length=8192
  generator.sampling_params.temperature=1.0
  generator.sampling_params.top_p=1.0
  generator.eval_sampling_params.temperature=1.0
  generator.eval_sampling_params.top_p=0.7
  generator.n_samples_per_prompt=16
  generator.eval_n_samples_per_prompt=32
  trainer.train_batch_size=512
  trainer.epochs=20
  trainer.update_epochs_per_batch=1
  trainer.eval_batch_size=1024
  trainer.eval_before_train=true
  trainer.eval_interval=5
  trainer.policy.optimizer_config.weight_decay=0.1
  trainer.strategy=fsdp
  trainer.placement.colocate_all=true
  generator.batched=true
  generator.inference_engine.backend=vllm
  generator.inference_engine.run_engines_locally=true
  generator.inference_engine.weight_sync_backend=nccl
  generator.inference_engine.tensor_parallel_size=1
  generator.inference_engine.gpu_memory_utilization=0.8
  trainer.ckpt_interval="${OPD_CKPT_INTERVAL:-5}"
  trainer.max_ckpts_to_keep=2
  trainer.hf_save_interval=10
  trainer.resume_mode=latest
  trainer.logger=wandb
)

# $1 = run name, $2 = GPUs the trained model takes (FSDP ranks, and one colocated vLLM engine on each).
# Fills RUN_OPTS with the placement and the run's names and paths.
opd_run_opts() {
  RUN_OPTS=(
    trainer.placement.policy_num_nodes=1
    trainer.placement.policy_num_gpus_per_node="$2"
    generator.inference_engine.num_engines="$2"
    trainer.project_name="$PROJECT"
    trainer.run_name="$1"
    trainer.ckpt_path="$CKPT_ROOT/$1"
    trainer.export_path="$EXPORT_ROOT/$1"
    trainer.log_path="$LOGS/skyrl/$1"
  )
}

# The training process. HF_TOKEN is removed from its environment: SkyRL copies HF_TOKEN into the Ray runtime
# env and logs the value while doing so ("Exporting `HF_TOKEN` to ray runtime env: <value>", in
# prepare_runtime_environment), so a run started with the token set writes it into its own log. Nothing in the
# training needs it (public base models, teacher loaded from a local export); only the upload step does.
# OPD_FORWARD_HF_TOKEN=1 keeps it, for a private Hub repo as OPD_TEACHER_MODEL; the token then lands in the log.
opd_uv_run() {
  cd "$SKYRL_DIR"   # uv resolves the project (and its extras) from the cwd
  if [[ "${OPD_FORWARD_HF_TOKEN:-0}" == "1" ]]; then
    uv run --isolated --extra fsdp "$@"
  else
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp "$@"
  fi
}

# Upload a finished run's final export. Returns 3 when the upload fails, so a caller can tell "trained but not
# uploaded" (re-running the same script retries just the upload) from a failed training run.
opd_upload() {
  [[ "${OPD_UPLOAD:-1}" == "1" ]] || { echo "OPD_UPLOAD=0: $1 is not uploaded"; return 0; }
  [[ "$1" != smoke_* ]] || { echo "smoke run: $1 is not uploaded"; return 0; }
  if [[ -f "$LOGS/$1.uploaded" ]]; then echo "already uploaded: $(cat "$LOGS/$1.uploaded")"; return 0; fi
  if bash "$OPD_KIT_DIR/upload_export.sh" --run "$1"; then return 0; fi
  echo "TRAINING FINISHED BUT THE UPLOAD FAILED for $1. The export is intact under $EXPORT_ROOT/$1." >&2
  echo "Retry: bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --run $1" >&2
  return 3
}

# A run that already finished is not started again: training is skipped and only a missing upload is retried.
# Run names are fixed (no timestamp), which is what lets a preempted run resume by re-running the same command;
# OPD_TAG=<suffix> gives a fresh run of the same kind, and removing $LOGS/<run>.done re-opens a finished one.
opd_skip_if_done() {
  [[ -f "$LOGS/$1.done" ]] || return 0
  echo "$1 already finished ($(cat "$LOGS/$1.done")); not training again. OPD_TAG=<suffix> starts a fresh run."
  local rc=0; opd_upload "$1" || rc=$?
  exit "$rc"
}

# Called right after the training command returned 0.
opd_finish() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/$1.done"
  echo "=== $1 finished training at $(cat "$LOGS/$1.done")"
  opd_upload "$1"
}
