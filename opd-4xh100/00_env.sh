# Source this on the 4xH100 node, from anywhere:
#   OPD_PAIR=post source skyrl-test/opd-4xh100/00_env.sh
# Finds the SkyRL checkout next to this kit (<root>/SkyRL beside <root>/skyrl-test, or the kit cloned
# inside SkyRL/; SKYRL_DIR overrides), requires it to be on the PR branch (kyuds/opd-entrypoint), and
# checks uv, the GPUs and the W&B key; exports the knobs the other scripts read. Never prints the key.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash (it uses BASH_SOURCE to find itself)"; return 1 2>/dev/null || exit 1; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_locate.sh"
opd_locate || return 1 2>/dev/null || exit 1
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

# Ray workers get the same uv environment the driver runs in (install doc's "configure Ray to use uv").
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# The two experiments; `post` (the post-trained pair) runs first. Each pair shares one tokenizer (verified
# 2026-09-25 from the HF file hashes: 0.8B-Base and 9B-Base share tokenizer.json fe000e3e..., 0.8B and 9B
# share 5f9e4d49...); the pairs do NOT share one with each other, so never mix a Base teacher with a
# post-trained student. OPD_TEACHER_MODEL overrides the teacher (e.g. the 2B or 4B of the same variant).
export OPD_PAIR="${OPD_PAIR:-post}"
case "$OPD_PAIR" in
  base) export OPD_STUDENT_MODEL="Qwen/Qwen3.5-0.8B-Base"; export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-Qwen/Qwen3.5-9B-Base}" ;;
  post) export OPD_STUDENT_MODEL="Qwen/Qwen3.5-0.8B";      export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-Qwen/Qwen3.5-9B}" ;;
  *) echo "OPD_PAIR must be base or post, got '$OPD_PAIR'"; return 1 2>/dev/null || exit 1 ;;
esac
# Qwen3.5 templates default to thinking mode. Off by default for both pairs so responses stay inside the
# 8k budget and the two experiments see the same prompt format; OPD_THINKING=true to let the models think.
export OPD_THINKING="${OPD_THINKING:-false}"

# GPU split: the teacher is stand-alone vLLM servers (one per GPU, round-robin; 02_serve_teacher.sh), the
# student trains on the other GPUs (colocate_all: FSDP policy + one vLLM engine per GPU, TP 1). Default:
# teacher on 0,1 and student on 2,3. A 0.8B student generates far faster than a 9B server prefills, so the
# teacher is the step's bottleneck and gets two GPUs; two data-parallel student ranks also make the
# blog's batch of 512 exact (with 3 ranks and n=16 it would have to be a multiple of 3). The run scripts
# find the teacher through the per-pair manifest 02_serve_teacher.sh writes and refuse to start on a GPU
# a running teacher holds. Ports: base 8000+, post 8100+ unless overridden. SkyRL learns the split only
# through CUDA_VISIBLE_DEVICES, so each run starts its own Ray instance (RAY_ADDRESS=local in the run
# scripts) instead of joining a cluster that is already up and has all four GPUs registered.
export OPD_TEACHER_GPUS="${OPD_TEACHER_GPUS:-0,1}"
export OPD_STUDENT_GPUS="${OPD_STUDENT_GPUS:-2,3}"
export OPD_NUM_STUDENT_GPUS="${OPD_NUM_STUDENT_GPUS:-2}"

export OPD_DATA="${OPD_DATA:-$HOME/data}"
export OPD_PROJECT="${OPD_PROJECT:-opd_4xh100}"
export OPD_LOGS="${OPD_LOGS:-$HOME/logs}"
mkdir -p "$OPD_LOGS"

echo "kit     : $OPD_KIT_DIR"
echo "skyrl   : $SKYRL_DIR @ $(git -C "$SKYRL_DIR" rev-parse --abbrev-ref HEAD) $(git -C "$SKYRL_DIR" rev-parse --short HEAD)"
echo "uv      : $(uv --version 2>/dev/null || echo MISSING)"
echo "gpus    :"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
echo "pair    : $OPD_PAIR  student=$OPD_STUDENT_MODEL  teacher=$OPD_TEACHER_MODEL  thinking=$OPD_THINKING"
echo "teacher : GPUs $OPD_TEACHER_GPUS (02_serve_teacher.sh --pair $OPD_PAIR); manifest: $OPD_LOGS/teacher_$OPD_PAIR.urls$( [[ -f $OPD_LOGS/teacher_$OPD_PAIR.urls ]] && echo " = $(cat $OPD_LOGS/teacher_$OPD_PAIR.urls)" || echo " (not serving yet)")"
echo "student : GPUs $OPD_STUDENT_GPUS ($OPD_NUM_STUDENT_GPUS)"
echo "wandb   : key set (${#WANDB_API_KEY} chars)   project: $OPD_PROJECT"
echo "data    : $OPD_DATA   logs: $OPD_LOGS"
