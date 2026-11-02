# Runs on the VM. The run scripts source this themselves; source it by hand, from bash, to use them in a shell:
#   source skyrl-test/opd-full-experiments/teacher/00_env.sh
# Finds the SkyRL checkout around this kit (~/SkyRL/skyrl-test, as node_setup.sh lays it out), loads the node's NCCL
# settings and your keys, checks that Ray is up, and sets the paths. Never prints a key.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash (it uses BASH_SOURCE to find itself)"; return 1 2>/dev/null || exit 1; }
TEACHER_KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TEACHER_KIT/config.sh"
export SKYRL_DIR="${SKYRL_DIR:-$(cd "$TEACHER_KIT/../../.." && pwd)}"
[[ -f "$SKYRL_DIR/examples/train/algorithms/dapo/main_dapo.py" ]] || {
  echo "no SkyRL checkout at $SKYRL_DIR: launch_all.sh sets one up, or set SKYRL_DIR"; return 1 2>/dev/null || exit 1
}

# ~/.opd_cluster_env: written by node_setup.sh (uv on PATH, NCCL over plain sockets for a single node).
# ~/.opd_secrets:     written by launch_all.sh from your Mac's environment (WANDB_API_KEY, HF_TOKEN).
if [[ -f "$HOME/.opd_cluster_env" ]]; then source "$HOME/.opd_cluster_env"; fi
if [[ -f "$HOME/.opd_secrets" ]]; then source "$HOME/.opd_secrets"; fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv is not on PATH: launch_all.sh sets it up"; return 1 2>/dev/null || exit 1; }
: "${WANDB_API_KEY:?no WANDB_API_KEY: launch_all.sh copies it from your Mac}"
: "${HF_TOKEN:?no HF_TOKEN: launch_all.sh copies it from your Mac}"

# Ray workers get the same uv environment the driver runs in. A run joins the node's Ray cluster, which node_setup.sh
# starts from SkyRL's own environment, so its Ray is the lockfile's Ray.
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
if [[ -z "${RAY_ADDRESS:-}" && ! -f /tmp/ray/ray_current_cluster ]]; then
  echo "no Ray cluster is running on this node: launch_all.sh starts it"; return 1 2>/dev/null || exit 1
fi

# Everything a run must keep is on the boot disk: a preempted spot VM is stopped, its boot disk survives and the run
# resumes from its last checkpoint. The local NVMe array is wiped by a preemption, so it only holds caches.
export OPD_STORE="${OPD_STORE:-$HOME/opd-store}"
export LOGS="$OPD_STORE/logs"
export CKPT_ROOT="$OPD_STORE/ckpts"
export EXPORT_ROOT="$OPD_STORE/exports"
# Not TRAIN_FILE or DATA_DIR: prepare_dapo_data.sh reads those from the environment, and would download the raw
# dataset over the cleaned file.
export DAPO_DIR="$OPD_STORE/data/dapo"
export DAPO_TRAIN="$DAPO_DIR/dapo-math-17k-cleaned.parquet"
export EVAL_2024="$DAPO_DIR/aime-2024-cleaned.parquet"
export EVAL_2025="$DAPO_DIR/aime-2025-cleaned.parquet"
mkdir -p "$LOGS" "$CKPT_ROOT" "$EXPORT_ROOT" "$DAPO_DIR"

echo "skyrl : $SKYRL_DIR @ $(git -C "$SKYRL_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')   kit: $(git -C "$TEACHER_KIT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "gpus  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ')"
echo "store : $OPD_STORE ($(df -h --output=avail "$OPD_STORE" 2>/dev/null | tail -1 | tr -d ' ') free)"
