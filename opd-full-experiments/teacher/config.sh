# The teacher runs, and everything your Mac and the VMs must agree on. Sourced on both sides, so nothing here may call
# gcloud, and it has to work on the Mac's bash 3.2. The VMs read this file from GitHub: to change a setting, edit it
# here and push.

# One line per run: <node> <base model>. A run takes all 8 GPUs of its node. Each size comes as Base and instruct, so
# the OPD runs can compare teacher lineage. The H100s train the smaller models and the B200s the 8B pair. To change a
# run, change its line: its W&B name and Hub folder come from the model name.
TEACHER_RUNS="${TEACHER_RUNS:-
h100-1 Qwen/Qwen3-1.7B-Base
h100-2 Qwen/Qwen3-1.7B
h100-3 Qwen/Qwen3-4B-Base
h100-4 Qwen/Qwen3-4B
b200-1 Qwen/Qwen3-8B-Base
b200-2 Qwen/Qwen3-8B
}"

# ---- naming ----
TEACHER_METHOD=dapo                     # SkyRL's DAPO recipe, GRPO advantages included. See 02_train_teacher.sh
TEACHER_DATA=nemotron_math              # in run names
TEACHER_DATA_LONG=nemotron-cascade-rl-math   # in W&B tags, and the training rows' data_source
TEACHER_PROJECT=opd_full_experiments    # W&B project
# One Hub repo, one folder per model: a repo id cannot have a second slash, so kyuds/opd_full_experiments/<model> is a
# folder inside kyuds/opd_full_experiments.
TEACHER_HF_REPO=kyuds/opd_full_experiments

# ---- the run ----
TEACHER_STEPS=1000
# Demystifying's GRPO shape, 128 prompts x 8 rollouts, about 9 passes over the data in 1000 steps. The earlier DAPO
# runs' 512 x 16 would be 35 passes and take 2 to 4 times as long per step.
TEACHER_PROMPTS=128
TEACHER_SAMPLES=8
# 64 prompts x 8 rollouts is the 512 rollouts per optimizer step of the earlier runs, so 2 optimizer steps per training
# step. Warmup stays 10 training steps, as in the earlier runs.
TEACHER_MINI_BATCH=64
TEACHER_WARMUP=20
TEACHER_EVAL_INTERVAL=25
# A training step takes about 3 minutes, so a preemption costs at most about 30 minutes.
TEACHER_CKPT_INTERVAL=10
TEACHER_HF_INTERVAL=100

# ---- code ----
# The VMs check out this exact commit, including after a preemption, so every run and every resume use the same code
# even if the branch moves. It is the head of kyuds/opd-entrypoint (PR #2256) on 2026-10-31, which the OPD runs build on.
SKYRL_REPO=https://github.com/NovaSky-AI/SkyRL.git
SKYRL_COMMIT=0a1a769a2b032e932a659cd4d32328c4f8f0fae9
KIT_REPO=https://github.com/kyuds/skyrl-test.git
KIT_BRANCH=opd-full-experiments

# Qwen/Qwen3-1.7B-Base -> qwen3_1p7b_base
model_tag() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | sed 's/\./p/g'; }
# Method, data and base model: dapo_nemotron_math_qwen3_1p7b_base
run_name() { echo "${TEACHER_METHOD}_${TEACHER_DATA}_$(model_tag "$1")"; }
