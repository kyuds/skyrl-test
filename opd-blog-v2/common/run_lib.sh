# Sourced by every run script. Runs on the VM; with V2_DRY_RUN=1 it also runs on your Mac and only prints the
# overrides, which is how check_configs.sh checks them against SkyRL's real config classes.
#
#   v2_init <run> "$@"        parses --smoke, sets RUN, PROJECT and the paths, and on the VM loads 00_env.sh
#   v2_model <spec>           a model path for: a Hub id, kit:<job> (an SFT output) or run:<job> (a run's last export)
#   v2_fix_base <base> <tok> <name>    a Base model with fixed settings (fix_base_settings.py), as $MODELS/<name>
#   v2_opd_recipe             prints the OPD overrides from the run script's HYPERPARAMETERS block
#   v2_tp_32b                 tensor parallelism for Qwen3-32B on this node's GPUs (2 on an H100, 1 on a B200)
#   v2_teacher <model> <engines> <tp>    prints one teacher's overrides
#   v2_bookkeeping <tags>     prints the overrides every run shares: logging, checkpoints, exports, resume, eval
#   v2_collect                reads overrides on stdin into OVERRIDES, adding --smoke's and the extra arguments
#   v2_launch opd             runs common/kit_main.py with OVERRIDES, then uploads the run's last export
#   v2_sft                    SFT from the SFT_* variables: train, check the result ends turns at <|im_end|>, upload it
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
    V2_DATA="$OPD_STORE/data/v2"; MODELS="$OPD_STORE/models"; JOBS="$OPD_STORE/jobs"; EVALS="$OPD_STORE/evals"
    TOOLS="$OPD_STORE/tools"; SKYRL_DIR=/home/user/SkyRL
    MATH_TRAIN="$V2_DATA/math_train.parquet"; MOPD_TRAIN="$V2_DATA/mopd_train.parquet"
    DEEPMATH_TRAIN="$V2_DATA/deepmath_train.parquet"
    AIME24="$V2_DATA/aime24.parquet"; AIME25="$V2_DATA/aime25.parquet"
    SFT_2C="$V2_DATA/sft_2c_qwen3_32b_traces"
  else
    source "$V2_LIB_DIR/00_env.sh"
  fi
  PROJECT="$V2_PROJECT"
  if [[ $SMOKE == 1 ]]; then
    RUN="smoke_$RUN"
    PROJECT="${V2_PROJECT}-smoke"   # keeps the real project to the real runs
  fi
}

# Prints a local path (or Hub id) for a model spec. kit:<job> is an SFT job's output: this node's copy in $MODELS,
# or else the Hub folder <job> of $V2_HF_REPO, downloaded into $MODELS first. run:<job> is a training job's highest
# complete export on this node. Anything else is a Hub id, passed through.
v2_model() {
  local spec="$1" name n best=""
  case "$spec" in
    kit:*)
      name="${spec#kit:}"
      if [[ $V2_DRY_RUN == 1 || -f "$MODELS/$name/config.json" ]]; then echo "$MODELS/$name"; return 0; fi
      echo "fetching $name from https://huggingface.co/$V2_HF_REPO/tree/main/$name" >&2
      rm -rf "$MODELS/.hub/$name"
      (cd "$SKYRL_DIR" && uv run --isolated --extra fsdp hf download "$V2_HF_REPO" --include "$name/*" \
        --local-dir "$MODELS/.hub") >&2 || { echo "could not download $name from $V2_HF_REPO" >&2; return 1; }
      [[ -f "$MODELS/.hub/$name/config.json" ]] || { echo "$V2_HF_REPO has no model in folder $name" >&2; return 1; }
      mv "$MODELS/.hub/$name" "$MODELS/$name"
      echo "$MODELS/$name" ;;
    run:*)
      name="${spec#run:}"
      if [[ $V2_DRY_RUN == 1 ]]; then echo "$EXPORT_ROOT/$name/global_step_N/policy"; return 0; fi
      for n in $(ls "$EXPORT_ROOT/$name" 2>/dev/null | sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p' | sort -n); do
        if [[ -f "$EXPORT_ROOT/$name/global_step_$n/policy/config.json" ]] &&
           compgen -G "$EXPORT_ROOT/$name/global_step_$n/policy/*.safetensors" >/dev/null; then best="$n"; fi
      done
      [[ -n "$best" ]] || { echo "no complete export of $name under $EXPORT_ROOT/$name on this node" >&2; return 1; }
      echo "$EXPORT_ROOT/$name/global_step_$best/policy" ;;
    *) echo "$spec" ;;
  esac
}

# Prints $MODELS/<name>: a Base model's weights with the post-trained Qwen3's settings files (tokenizer, end tokens,
# context; common/fix_base_settings.py), written on first use. 1a's SFT starts from it; its output keeps them.
v2_fix_base() {
  local base="$1" tokenizer_from="$2" name="$3"
  if [[ $V2_DRY_RUN != 1 && ! -f "$MODELS/$name/config.json" ]]; then
    (cd "$SKYRL_DIR" && uv run --isolated --extra fsdp python "$V2_KIT/common/fix_base_settings.py" \
      --weights "$base" --tokenizer-from "$tokenizer_from" --out "$MODELS/$name") >&2 || return 1
  fi
  echo "$MODELS/$name"
}

# ---------------------------------------------------------------------------------------------------------------
# OPD
# ---------------------------------------------------------------------------------------------------------------

# Every OPD run script sets each of these in its HYPERPARAMETERS block, with the source of each value; nothing has a
# default here, so a setting cannot come from somewhere you did not look. v2_opd_recipe stops on a missing one.
V2_OPD_VARS="STUDENT TRAIN_FILE PROMPTS SAMPLES STEPS LR WARMUP BETAS WEIGHT_DECAY GRAD_CLIP MAX_PROMPT MAX_RESP
  TEMPERATURE TOP_P TOP_K KL_COEF LOSS LOSS_REDUCTION EVAL_EVERY EVAL_N EVAL_RESP EVAL_TEMPERATURE EVAL_TOP_P
  EVAL_TOP_K STUDENT_GPUS MICRO"

v2_opd_recipe() {
  local v
  for v in $V2_OPD_VARS; do
    [[ -n "${!v:-}" ]] || { echo "the run script did not set $v" >&2; return 1; }
  done
  printf '%s\n' \
    "trainer.policy.model.path=$STUDENT" \
    "data.train_data=['$TRAIN_FILE']" \
    "data.val_data=['$AIME24','$AIME25']" \
    environment.env_class=opd_v2_math \
    trainer.strategy=fsdp \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_nodes=1 \
    "trainer.placement.policy_num_gpus_per_node=$STUDENT_GPUS" \
    "generator.inference_engine.num_engines=$STUDENT_GPUS" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    "generator.inference_engine.engine_init_kwargs.max_model_len=$((MAX_PROMPT + EVAL_RESP + 1))" \
    generator.batched=false \
    "trainer.train_batch_size=$PROMPTS" \
    "generator.n_samples_per_prompt=$SAMPLES" \
    "trainer.policy_mini_batch_size=$PROMPTS" \
    trainer.update_epochs_per_batch=1 \
    "trainer.max_training_steps=$STEPS" \
    "trainer.max_prompt_length=$MAX_PROMPT" \
    "generator.sampling_params.max_generate_length=$MAX_RESP" \
    "generator.sampling_params.temperature=$TEMPERATURE" \
    "generator.sampling_params.top_p=$TOP_P" \
    "generator.sampling_params.top_k=$TOP_K" \
    "generator.eval_sampling_params.max_generate_length=$EVAL_RESP" \
    "generator.eval_sampling_params.temperature=$EVAL_TEMPERATURE" \
    "generator.eval_sampling_params.top_p=$EVAL_TOP_P" \
    "generator.eval_sampling_params.top_k=$EVAL_TOP_K" \
    "generator.eval_n_samples_per_prompt=$EVAL_N" \
    "trainer.algorithm.opd.kl_coef=$KL_COEF" \
    trainer.algorithm.opd.use_task_reward=false \
    "trainer.algorithm.policy_loss_type=$LOSS" \
    "trainer.algorithm.loss_reduction=$LOSS_REDUCTION" \
    "trainer.algorithm.max_seq_len=$((MAX_PROMPT + MAX_RESP))" \
    trainer.algorithm.use_kl_loss=false \
    "trainer.policy.optimizer_config.lr=$LR" \
    "trainer.policy.optimizer_config.num_warmup_steps=$WARMUP" \
    trainer.policy.optimizer_config.scheduler=constant_with_warmup \
    "trainer.policy.optimizer_config.adam_betas=$BETAS" \
    "trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY" \
    "trainer.policy.optimizer_config.max_grad_norm=$GRAD_CLIP" \
    "trainer.micro_train_batch_size_per_gpu=$MICRO" \
    "trainer.micro_forward_batch_size_per_gpu=$((MICRO * 2))"
}

# Tensor parallelism for Qwen3-32B: 2 GPUs per engine on an 80 GB H100 (its 64 GB of weights leave no room for a long
# context on one), 1 on a 180 GB B200. Read from the node's first GPU; the Mac's dry run assumes an H100.
v2_tp_32b() {
  local mib
  if [[ $V2_DRY_RUN == 1 ]]; then echo 2; return 0; fi
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "${mib:-0}" -lt 100000 ]]; then echo 2; else echo 1; fi
}

# One teacher (trainer.teacher), thinking mode like the student, on its own engines. A teacher only scores, so its
# context is the training rollout's, not the eval's.
v2_teacher() {
  local model="$1" engines="$2" tp="$3"
  printf '%s\n' \
    trainer.teacher.backend=skyrl \
    "trainer.teacher.model=$model" \
    "trainer.teacher.inference_engine.num_engines=$engines" \
    "trainer.teacher.inference_engine.tensor_parallel_size=$tp" \
    "trainer.teacher.inference_engine.engine_init_kwargs.max_model_len=$((MAX_PROMPT + MAX_RESP + 1))" \
    "trainer.teacher.max_concurrency=$((16 * engines))"
}

# Logging, checkpoints, exports and resume: the same for every run. A checkpoint every 10 steps (the last 2 kept)
# bounds what a preemption costs; an HF export every 50 steps keeps a usable model near any eval.
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
    "trainer.eval_interval=$EVAL_EVERY" \
    trainer.eval_batch_size=1024 \
    trainer.epochs=20
}

# Tiny settings for --smoke: 2 steps of 8 prompts x 2 samples, 512-token answers, one eval sample per problem.
# It checks that the models load, the teacher answers and the step finishes; it says nothing about the recipe.
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

# Runs the trainer, then uploads the run's last export to the Hub folder <run>_step<N> (V2_UPLOAD=0 skips it). A
# failed upload only warns: the export is still on this node, where the run's final eval reads it.
v2_launch() {
  local mode="$1"
  if [[ $V2_DRY_RUN == 1 ]]; then
    echo "$mode"
    printf '%s\n' "${OVERRIDES[@]}"
    return 0
  fi
  if [[ $SMOKE == 1 ]]; then rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN"; fi
  bash "$V2_KIT/data/prepare.sh"
  echo "=== run=$RUN  project=$PROJECT  mode=$mode  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  %s\n' "${OVERRIDES[@]}"
  # HF_TOKEN is kept out of the training process: SkyRL logs the value of what it copies into the Ray runtime env, and
  # training needs no token (every model it reads is public or already on disk).
  cd "$SKYRL_DIR"   # uv resolves the project, and Ray ships this directory to its workers
  # math-verify 0.9.0, the version tested; SkyRL's lock holds its ANTLR runtime at 4.9.3 (through omegaconf), the
  # parser common/kit_main.py's grader was checked against.
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp --with math-verify==0.9.0 \
    python "$V2_KIT/common/kit_main.py" "$mode" "${OVERRIDES[@]}"
  if [[ $SMOKE == 1 ]]; then
    rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN"
    echo "=== $RUN passed"
    return 0
  fi
  echo "=== $RUN finished training at $(date -u +%Y-%m-%dT%H:%M:%SZ). Exports: $EXPORT_ROOT/$RUN"
  if [[ "${V2_UPLOAD:-1}" == 1 ]]; then
    bash "$V2_KIT/common/upload_export.sh" --run "$RUN" || echo "warning: the upload failed; the export is on this node"
  fi
}

# ---------------------------------------------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------------------------------------------

# Uploads a folder to the Hub folder $2 of $V2_HF_REPO, trying again on failure (several nodes commit to one repo).
v2_upload_dir() {
  local dir="$1" folder="$2" attempt
  for attempt in 1 2 3 4 5; do
    if (cd "$SKYRL_DIR" && uv run --isolated --extra fsdp hf upload "$V2_HF_REPO" "$dir" "$folder" \
          --commit-message "Upload $folder (skyrl-test/opd-blog-v2)"); then
      echo "uploaded: https://huggingface.co/$V2_HF_REPO/tree/main/$folder"
      return 0
    fi
    echo "upload attempt $attempt failed"
    [[ $attempt == 5 ]] || sleep "${V2_UPLOAD_RETRY_SECS:-120}"
  done
  return 1
}

# Full fine-tuning with SkyRL's SFT trainer on pretokenized data. SkyRL's export carries the starting model's settings
# files (config, generation config, tokenizer), so the result ends its turns at <|im_end|> when the start does: 1a
# starts from Qwen3-8B-Base with its settings fixed (common/fix_base_settings.py), 2c from M2RL-SFT. The job checks
# that before uploading. The last export is linked as $MODELS/$RUN and uploaded to the Hub folder $RUN. A preempted
# run resumes from its last checkpoint; a finished one is not trained again.
# The run script sets every one of V2_SFT_VARS in its HYPERPARAMETERS block; nothing has a default here.
V2_SFT_VARS="SFT_BASE SFT_DATA SFT_ROWS SFT_LR SFT_SCHEDULER SFT_WARMUP SFT_BETAS SFT_WEIGHT_DECAY SFT_GRAD_CLIP
  SFT_BATCH SFT_MICRO SFT_MAX_LEN SFT_EPOCHS SFT_CKPT_EVERY SFT_HF_EVERY SFT_TAGS"
v2_sft() {
  local out="$MODELS/$RUN" export_dir v
  for v in $V2_SFT_VARS; do
    [[ -n "${!v:-}" ]] || { echo "the run script did not set $v" >&2; return 1; }
  done
  OVERRIDES=(
    strategy=fsdp
    "model.path=$SFT_BASE"
    "pretokenized_dataset_paths=['$SFT_DATA']"
    "max_length=$SFT_MAX_LEN"
    "num_epochs=$SFT_EPOCHS"
    "batch_size=$SFT_BATCH"
    "micro_train_batch_size_per_gpu=$SFT_MICRO"
    remove_microbatch_padding=true
    sampler=random
    seed=42
    "optimizer_config.lr=$SFT_LR"
    "optimizer_config.scheduler=$SFT_SCHEDULER"
    "optimizer_config.num_warmup_steps=$SFT_WARMUP"
    "optimizer_config.adam_betas=$SFT_BETAS"
    "optimizer_config.weight_decay=$SFT_WEIGHT_DECAY"
    "optimizer_config.max_grad_norm=$SFT_GRAD_CLIP"
    placement.num_nodes=1
    placement.num_gpus_per_node=8
    fsdp_config.cpu_offload=false
    logger=wandb
    "project_name=$PROJECT"
    "run_name=$RUN"
    "tags=$SFT_TAGS"
    "ckpt_path=$CKPT_ROOT/$RUN"
    "ckpt_interval=$SFT_CKPT_EVERY"
    max_ckpts_to_keep=1
    resume_from=latest
    "hf_save_interval=$SFT_HF_EVERY"
    "export_path=$EXPORT_ROOT/$RUN"
  )
  if [[ $SMOKE == 1 ]]; then
    # The real micro-batch on real rows, so a smoke run also checks the memory of a full-length micro-batch.
    OVERRIDES+=(max_training_steps=2 batch_size=16 hf_save_interval=2 ckpt_interval=0 'resume_from=""')
  else
    # SkyRL at the pinned commit gives the LR scheduler no total when the length is set in epochs
    # (skyrl/train/sft_trainer.py, init: num_training_steps=None), and a linear schedule then fails at startup. The
    # same total as max_training_steps fixes it without changing the run: SkyRL counts ceil(rows / batch) steps an
    # epoch, so this is exactly SFT_EPOCHS epochs, and the schedule reaches 0 at the last step.
    OVERRIDES+=("max_training_steps=$(( SFT_EPOCHS * ((SFT_ROWS + SFT_BATCH - 1) / SFT_BATCH) ))")
  fi
  OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})
  if [[ $V2_DRY_RUN == 1 ]]; then
    echo sft
    printf '%s\n' "${OVERRIDES[@]}"
    return 0
  fi

  if [[ -f "$out/config.json" ]]; then
    echo "$out is already built; not training again (delete it to retrain)"
  else
    if [[ $SMOKE == 1 ]]; then rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN"; fi
    echo "=== run=$RUN  project=$PROJECT  mode=sft  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  %s\n' "${OVERRIDES[@]}"
    (cd "$SKYRL_DIR" && env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN uv run --isolated --extra fsdp \
      python -m skyrl.train.main_sft "${OVERRIDES[@]}")
    export_dir="$(v2_model "run:$RUN")"
    echo "=== $RUN finished training at $(date -u +%Y-%m-%dT%H:%M:%SZ): $export_dir"
    # The OPD student and the evals need a model that ends its turn at <|im_end|> (fix_base_settings.py says why).
    python3 - "$export_dir" <<'PY'
import json, os, sys
d = sys.argv[1]
gen = json.load(open(os.path.join(d, "generation_config.json"))).get("eos_token_id")
gen = gen if isinstance(gen, list) else [gen]
tok = json.load(open(os.path.join(d, "tokenizer_config.json"))).get("eos_token")
tok = tok.get("content") if isinstance(tok, dict) else tok
if 151645 not in gen or tok != "<|im_end|>":
    sys.exit(f"{d} does not end its turn at <|im_end|> (generation eos {gen}, tokenizer eos {tok}): was the SFT "
             "started from a model with fixed settings (common/fix_base_settings.py)?")
context = json.load(open(os.path.join(d, "config.json")))["max_position_embeddings"]
print(f"{d} ends its turn at <|im_end|>; context {context}")
PY
    ln -sfn "$export_dir" "$out"
  fi
  export_dir="$(cd "$out" && pwd -P)"
  if [[ $SMOKE == 1 ]]; then
    rm -rf "${CKPT_ROOT:?}/$RUN" "${EXPORT_ROOT:?}/$RUN" "${out:?}"
    echo "=== $RUN passed"
    return 0
  fi
  # Other nodes fetch it from the Hub (v2_model kit:<job>), so a failed upload fails the job.
  v2_upload_dir "$export_dir" "$RUN"
}
