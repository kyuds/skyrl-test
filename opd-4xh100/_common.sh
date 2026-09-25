# Shared flags for the OPD runs. The blog's OPD settings (LR 1e-5, no mini-batching, n=16, temperature 1,
# eval at top_p 0.7 with 32 samples, 2048/8192 lengths, batched single-turn generation) on the Qwen3.5 pair
# selected by 00_env.sh, with the Qwen3.5-on-FSDP flags from examples/train/models/run_qwen3.5_0.8b.sh
# (text-only model on policy, ref and engines; no microbatch packing; NCCL weight sync), the teacher on the
# local vLLM servers and the student on $OPD_NUM_STUDENT_GPUS colocated GPUs. AIME24 (the blog's eval,
# env aime) and a GSM8K slice (env gsm8k, from the parquet's env_class column) are both evaluated.
# Sourced by 03/04; each sets RUN_NAME and the batch shape before calling opd_run "$@".
DATA="${OPD_DATA:-$HOME/data}"
TRAIN_FILE="$DATA/dapo/dapo-math-17k-cleaned.parquet"
AIME_FILE="$DATA/dapo/aime-2024-cleaned.parquet"
GSM8K_FILE="$DATA/gsm8k/validation_${GSM8K_EVAL_ROWS:-256}.parquet"
NUM_GPUS="${OPD_NUM_STUDENT_GPUS:-2}"

teacher_urls() {  # the manifest 02_serve_teacher.sh wrote for this pair, e.g. ['http://127.0.0.1:8100','http://127.0.0.1:8101']
  local manifest="${OPD_LOGS:-$HOME/logs}/teacher_${OPD_PAIR}.urls"
  [[ -f "$manifest" ]] || { echo "no teacher manifest at $manifest; run: bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair ${OPD_PAIR}" >&2; return 1; }
  tr -d '\n' < "$manifest"
}

refuse_gpu_collision() {  # $1 = GPUs the student wants; refuses if a running teacher holds any of them
  local want="$1" f pair held u up
  for f in "${OPD_LOGS:-$HOME/logs}"/teacher_*.gpus; do
    [[ -f "$f" ]] || continue
    pair="$(basename "$f" .gpus)"; pair="${pair#teacher_}"; held="$(cat "$f")"; up=0
    for u in $(tr -d "[]'" < "${OPD_LOGS:-$HOME/logs}/teacher_${pair}.urls" 2>/dev/null | tr ',' ' '); do curl -sf "$u/v1/models" >/dev/null 2>&1 && up=1; done
    [[ $up == 1 ]] || continue
    for g in ${want//,/ }; do
      for h in ${held//,/ }; do
        [[ "$g" == "$h" ]] && { echo "GPU $g is held by the running '$pair' teacher; stop it (02_serve_teacher.sh --pair $pair --stop) or pick other GPUs" >&2; return 1; }
      done
    done
  done
  return 0  # explicit: the last [[ ]] above is false whenever there is no collision, and set -e would trip on it
}

opd_run() {
  local urls; urls="$(teacher_urls)"
  echo "$RUN_NAME" > "${OPD_LOGS:-$HOME/logs}/last_opd_run"
  echo "run_name=$RUN_NAME  pair=${OPD_PAIR}  student=${OPD_STUDENT_MODEL}  teacher=${OPD_TEACHER_MODEL} @ ${urls}  thinking=${OPD_THINKING:-false}"
  for u in $(echo "$urls" | tr -d "[]'" | tr ',' ' '); do
    curl -sf "$u/v1/models" >/dev/null || { echo "teacher not reachable at $u; run: bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair ${OPD_PAIR}"; exit 1; }
  done
  refuse_gpu_collision "${OPD_STUDENT_GPUS:-2,3}"
  CUDA_VISIBLE_DEVICES="${OPD_STUDENT_GPUS:-2,3}" \
  uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
    data.train_data="['$TRAIN_FILE']" \
    data.val_data="['$AIME_FILE','$GSM8K_FILE']" \
    trainer.policy.model.path="${OPD_STUDENT_MODEL}" \
    trainer.policy.language_model_only=true \
    trainer.ref.language_model_only=true \
    generator.inference_engine.language_model_only=true \
    trainer.remove_microbatch_padding=false \
    trainer.teacher.backend=vllm \
    trainer.teacher.model="${OPD_TEACHER_MODEL}" \
    trainer.teacher.server_urls="${urls}" \
    trainer.teacher.max_concurrency="${OPD_TEACHER_MAX_CONCURRENCY:-256}" \
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
