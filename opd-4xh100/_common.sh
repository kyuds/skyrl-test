# Shared flags for the OPD runs. The blog's OPD settings (LR 1e-5, no mini-batching, n=16, temperature 1,
# eval at top_p 0.7 with 32 samples, 2048/8192 lengths, batched single-turn generation) on the Qwen3.5 pair
# selected by 00_env.sh, with the Qwen3.5-on-FSDP flags from examples/train/models/run_qwen3.5_0.8b.sh
# (text-only model on policy, ref and engines; no microbatch packing; NCCL weight sync). The teacher is
# launched by the run itself (trainer.teacher.backend=skyrl, see TEACHER_OPTS) as SkyRL inference servers
# behind a RemoteInferenceClient on $OPD_TEACHER_NUM_GPUS GPUs; the student trains on $OPD_NUM_STUDENT_GPUS
# colocated GPUs. The run joins the node's Ray cluster, which places both from its ledger. AIME24 (the blog's eval,
# env aime) and a GSM8K slice (env gsm8k, from the parquet's env_class column) are both evaluated, from the kit's
# own copies that opd_build_eval_set rebuilds at the start of every run.
# Sourced by 02/03; each sets RUN_NAME and the batch shape before calling opd_run "$@".
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_locate.sh"; opd_locate || exit 1
: "${OPD_STUDENT_MODEL:?source 00_env.sh first (it selects the pair)}"
DATA="${OPD_DATA:-$HOME/data}"
TRAIN_FILE="$DATA/dapo/dapo-math-17k-cleaned.parquet"
# The eval set: AIME24 and the first GSM8K_EVAL_ROWS GSM8K validation rows. The runs read the kit's copies under
# $DATA/opd-eval, never the sources: SkyRL concatenates the val files, and the sources as written do not
# concatenate (SkyRL's DAPO prep writes AIME through pandas 3, whose strings are large_string; the GSM8K script
# writes through datasets, whose strings are string; seen 2026-09-27). See make_eval_set.py.
AIME_SRC="$DATA/dapo/aime-2024-cleaned.parquet"
GSM8K_SRC="$DATA/gsm8k/validation.parquet"
GSM8K_EVAL_ROWS="${GSM8K_EVAL_ROWS:-256}"
AIME_FILE="$DATA/opd-eval/aime24.parquet"
GSM8K_FILE="$DATA/opd-eval/gsm8k_validation_${GSM8K_EVAL_ROWS}.parquet"
NUM_GPUS="${OPD_NUM_STUDENT_GPUS:-2}"

# Rebuilds $AIME_FILE and $GSM8K_FILE from their sources and checks that SkyRL can concatenate them (the same
# datasets calls PromptDataset makes). A few hundred rows, deterministic and written atomically, so every run
# starts from the same eval files whatever an earlier preparation or run left behind. Called by opd_run, by
# 04_run_grpo.sh and at the end of 01_prepare_data.sh.
opd_build_eval_set() {
  local src
  for src in "$AIME_SRC" "$GSM8K_SRC"; do
    [[ -f "$src" ]] || { echo "missing $src: run skyrl-test/opd-4xh100/01_prepare_data.sh first" >&2; return 1; }
  done
  (cd "$SKYRL_DIR" && uv run --isolated --extra fsdp python "$OPD_KIT_DIR/make_eval_set.py" \
    --aime-src "$AIME_SRC" --aime-out "$AIME_FILE" \
    --gsm8k-src "$GSM8K_SRC" --gsm8k-out "$GSM8K_FILE" --gsm8k-rows "$GSM8K_EVAL_ROWS")
}

# ---- teacher: launched by the run (trainer.teacher.backend=skyrl, on kyuds/opd-teacher-launching) ----
# The run brings up OPD_TEACHER_NUM_GPUS TP-1 vLLM servers for the teacher in a placement group of their own
# (never the student's colocate group), before the student's engines and workers exist, drives them through a
# RemoteInferenceClient exactly like the student's engines, and takes them down when the run ends. The
# launched block's own defaults are not repeated here: gpu_memory_utilization 0.9, prefix caching off, Ray
# Prometheus stats off, and max_model_len = longest input + longest response + 1 (so the smoke's 2k responses
# get a 4097 context and the full run's 8k a 10241). Everything teacher-related is this one array.
TEACHER_OPTS=(
  trainer.teacher.backend=skyrl
  trainer.teacher.model="${OPD_TEACHER_MODEL}"
  trainer.teacher.inference_engine.num_engines="${OPD_TEACHER_NUM_GPUS:-2}"   # one server per GPU: a 9B fits an H100 with room for its KV cache, and two replicas prefill more than one TP-2 server
  trainer.teacher.inference_engine.tensor_parallel_size=1
  trainer.teacher.inference_engine.language_model_only=true                    # Qwen3.5 is multimodal; text-only, like the student's engines
  trainer.teacher.max_concurrency="${OPD_TEACHER_MAX_CONCURRENCY:-256}"        # in flight across the servers; each is also capped at SKYRL_GENERATE_CONCURRENCY_PER_ENGINE (512)
)

opd_run() {
  opd_build_eval_set || return 1
  echo "$RUN_NAME" > "${OPD_LOGS:-$HOME/logs}/last_opd_run"
  echo "run_name=$RUN_NAME  pair=${OPD_PAIR}  student=${OPD_STUDENT_MODEL} on ${NUM_GPUS} GPUs  teacher=${OPD_TEACHER_MODEL} on ${OPD_TEACHER_NUM_GPUS:-2} GPUs, launched by the run  thinking=${OPD_THINKING:-false}"
  cd "$SKYRL_DIR"   # uv resolves the project (and its extras) from the cwd
  # Joins the node's Ray cluster (Anyscale's, or one from `ray start --head`). The teacher servers are Ray
  # actors holding GPUs of their own, so Ray places the student's $NUM_GPUS-GPU group on the others;
  # nothing here picks device ids. (A driver-side CUDA_VISIBLE_DEVICES is ignored by a running cluster.)
  # If another run still holds GPUs, the placement groups wait and fail after SKYRL_RAY_PG_TIMEOUT_IN_S (180 s).
  # OPD_UV_WITH (unquoted on purpose: it is zero or two words) overrides Ray to the cluster's version.
  uv run --isolated --extra fsdp ${OPD_UV_WITH:-} -m skyrl.train.entrypoints.main_opd \
    data.train_data="['$TRAIN_FILE']" \
    data.val_data="['$AIME_FILE','$GSM8K_FILE']" \
    trainer.policy.model.path="${OPD_STUDENT_MODEL}" \
    trainer.policy.language_model_only=true \
    trainer.ref.language_model_only=true \
    generator.inference_engine.language_model_only=true \
    trainer.remove_microbatch_padding=false \
    "${TEACHER_OPTS[@]}" \
    trainer.algorithm.opd.kl_coef=1.0 \
    trainer.algorithm.opd.use_task_reward=false \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
    trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
    generator.inference_engine.num_engines="$NUM_GPUS" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    generator.batched=true \
    generator.chat_template_kwargs="{enable_thinking: ${OPD_THINKING:-false}}" \
    environment.env_class=aime \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.eval_sampling_params.temperature=1.0 \
    generator.eval_sampling_params.top_p=0.7 \
    trainer.update_epochs_per_batch=1 \
    trainer.policy.optimizer_config.lr=1e-5 \
    trainer.policy.optimizer_config.num_warmup_steps=0 \
    trainer.policy.optimizer_config.weight_decay=0.1 \
    trainer.micro_forward_batch_size_per_gpu=2 \
    trainer.micro_train_batch_size_per_gpu=2 \
    trainer.logger=wandb \
    trainer.project_name="${OPD_PROJECT:-opd_4xh100}" \
    trainer.run_name="$RUN_NAME" \
    trainer.ckpt_path="$HOME/ckpts/${OPD_PROJECT:-opd_4xh100}/$RUN_NAME" \
    trainer.export_path="$HOME/exports/${OPD_PROJECT:-opd_4xh100}/$RUN_NAME" \
    "$@"
}
