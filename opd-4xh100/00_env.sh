# Source this on the 4xH100 node, from anywhere:
#   OPD_PAIR=post source skyrl-test/opd-4xh100/00_env.sh
# Finds the SkyRL checkout next to this kit (<root>/SkyRL beside <root>/skyrl-test, or the kit cloned
# inside SkyRL/; SKYRL_DIR overrides), requires it to be on the branch with the launched teacher
# (kyuds/opd-teacher-launching: PR #2256's entrypoint plus trainer.teacher.backend=skyrl), checks uv, the
# GPUs, the W&B key and that a Ray cluster is running; exports the knobs the other scripts read.
# Never prints the key.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash (it uses BASH_SOURCE to find itself)"; return 1 2>/dev/null || exit 1; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_locate.sh"
opd_locate || return 1 2>/dev/null || exit 1
if [[ ! -f "$SKYRL_DIR/skyrl/train/opd/teacher_launch.py" ]]; then
  echo "$SKYRL_DIR has the OPD entrypoint but not the launched teacher (skyrl/train/opd/teacher_launch.py):"
  echo "  git -C \"$SKYRL_DIR\" checkout kyuds/opd-teacher-launching"
  return 1 2>/dev/null || exit 1
fi
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

# Ray workers get the same uv environment the driver runs in (install doc's "configure Ray to use uv").
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# The runs join the node's Ray cluster instead of starting their own: the OPD run launches its teacher as
# SkyRL inference servers, i.e. Ray actors that hold their GPUs in Ray's ledger, so the student lands on
# the others without anyone naming device ids. (A private instance per run does not work on an Anyscale
# node: its RAY_OVERRIDE_RESOURCES pins any new raylet to all four GPUs; seen 2026-09-25.) As in SkyRL's
# Anyscale CI, let SkyRL's own ray.init(runtime_env=...) merge with the job-level runtime env the
# workspace installs.
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
if [[ -z "${RAY_ADDRESS:-}" && ! -f /tmp/ray/ray_current_cluster ]]; then
  echo "no Ray cluster is running on this node (no /tmp/ray/ray_current_cluster, RAY_ADDRESS unset)."
  echo "Anyscale workspaces start one; elsewhere: (cd \"$SKYRL_DIR\" && uv run --isolated --extra fsdp ray start --head)"
  return 1 2>/dev/null || exit 1
fi

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

# GPU split, by count only (Ray picks the devices): the OPD run gives the teacher OPD_TEACHER_NUM_GPUS
# GPUs (servers of TP 1) and trains the student on OPD_NUM_STUDENT_GPUS (colocate_all: FSDP policy + one
# vLLM engine per GPU). A 0.8B student generates far faster than a 9B server prefills, so the teacher is
# the step's bottleneck and gets two GPUs; two data-parallel student ranks also make the blog's batch of
# 512 exact (with 3 ranks and n=16 it would have to be a multiple of 3). GRPO, with no teacher, takes all four.
export OPD_TEACHER_NUM_GPUS="${OPD_TEACHER_NUM_GPUS:-2}"
export OPD_NUM_STUDENT_GPUS="${OPD_NUM_STUDENT_GPUS:-2}"

export OPD_DATA="${OPD_DATA:-$HOME/data}"
export OPD_PROJECT="${OPD_PROJECT:-opd_4xh100}"
export OPD_LOGS="${OPD_LOGS:-$HOME/logs}"
mkdir -p "$OPD_LOGS"

echo "kit     : $OPD_KIT_DIR"
echo "skyrl   : $SKYRL_DIR @ $(git -C "$SKYRL_DIR" rev-parse --abbrev-ref HEAD) $(git -C "$SKYRL_DIR" rev-parse --short HEAD)"
echo "uv      : $(uv --version 2>/dev/null || echo MISSING)"
echo "ray     : cluster at $(cat /tmp/ray/ray_current_cluster 2>/dev/null || echo "${RAY_ADDRESS}")"
echo "gpus    :"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
echo "pair    : $OPD_PAIR  student=$OPD_STUDENT_MODEL  teacher=$OPD_TEACHER_MODEL  thinking=$OPD_THINKING"
echo "teacher : $OPD_TEACHER_NUM_GPUS GPUs ($OPD_TEACHER_NUM_GPUS TP-1 servers), launched by the OPD run (trainer.teacher.backend=skyrl)"
echo "student : $OPD_NUM_STUDENT_GPUS GPUs"
echo "wandb   : key set (${#WANDB_API_KEY} chars)   project: $OPD_PROJECT"
echo "data    : $OPD_DATA   logs: $OPD_LOGS"
