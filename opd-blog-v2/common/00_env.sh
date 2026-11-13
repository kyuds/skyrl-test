# Runs on the VM. Every run script sources this; source it by hand, from bash, to use the paths in a shell:
#   source skyrl-test/opd-blog-v2/common/00_env.sh
# Finds the SkyRL checkout around this kit (~/SkyRL/skyrl-test, as gcp/node_setup.sh lays it out), loads the node's NCCL
# settings and your keys, checks that Ray is up, and sets the paths. Never prints a key.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash (it uses BASH_SOURCE to find itself)" >&2; return 1 2>/dev/null || exit 1; }
V2_KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$V2_KIT/config.sh"
export SKYRL_DIR="${SKYRL_DIR:-$(cd "$V2_KIT/../.." && pwd)}"
[[ -f "$SKYRL_DIR/skyrl/train/entrypoints/main_opd.py" ]] || {
  echo "no SkyRL checkout with main_opd.py at $SKYRL_DIR: gcp/02_setup_nodes.sh sets one up, or set SKYRL_DIR" >&2; return 1 2>/dev/null || exit 1
}

# ~/.opd_cluster_env: written by node_setup.sh (uv on PATH, NCCL over plain sockets for a single node).
# ~/.opd_secrets:     written by gcp/02_setup_nodes.sh from your Mac's environment (WANDB_API_KEY, HF_TOKEN).
if [[ -f "$HOME/.opd_cluster_env" ]]; then source "$HOME/.opd_cluster_env"; fi
if [[ -f "$HOME/.opd_secrets" ]]; then source "$HOME/.opd_secrets"; fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv is not on PATH: gcp/02_setup_nodes.sh sets it up" >&2; return 1 2>/dev/null || exit 1; }
: "${WANDB_API_KEY:?no WANDB_API_KEY: gcp/02_setup_nodes.sh copies it from your Mac}"
: "${HF_TOKEN:?no HF_TOKEN: gcp/02_setup_nodes.sh copies it from your Mac}"

# Ray workers get the same uv environment the driver runs in, `--with math-verify` included. A run joins the node's
# Ray cluster, which node_setup.sh starts from SkyRL's own environment, so its Ray is the lockfile's Ray.
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
if [[ -z "${RAY_ADDRESS:-}" && ! -f /tmp/ray/ray_current_cluster ]]; then
  echo "no Ray cluster is running on this node: gcp/02_setup_nodes.sh starts it" >&2; return 1 2>/dev/null || exit 1
fi

# Everything a run must keep is on the boot disk: a preempted spot VM is stopped, its boot disk survives and the run
# resumes from its last checkpoint. The local NVMe array is wiped by a preemption, so it only holds caches (the HF
# cache with the downloaded models among them).
export OPD_STORE="${OPD_STORE:-$HOME/opd-store}"
export LOGS="$OPD_STORE/logs"
export CKPT_ROOT="$OPD_STORE/ckpts"
export EXPORT_ROOT="$OPD_STORE/exports"
export V2_DATA="$OPD_STORE/data/v2"
export MODELS="$OPD_STORE/models"      # SFT outputs, and Base models with fixed settings (fix_base_settings.py)
export JOBS="$OPD_STORE/jobs"          # common/job.sh's records: <job>.started, .done, .failed, .pid
export EVALS="$OPD_STORE/evals"        # 0-evals/run.sh: requests, responses and scores, one folder per eval job
export TOOLS="$OPD_STORE/tools"        # graders cloned at pinned commits (0-evals/README.md)
# Written by data/prepare_data.py. One prompt format for every math row, M2RL's.
export MATH_TRAIN="$V2_DATA/math_train.parquet"     # M2RL's math set: DAPO-17k + Skywork-OR1 (no OmniMath)
export MOPD_TRAIN="$V2_DATA/mopd_train.parquet"     # math + code + IF, each row labelled with its teacher
export DEEPMATH_TRAIN="$V2_DATA/deepmath_train.parquet"   # Thinking Machines' OPD prompts (section 1)
export AIME24="$V2_DATA/aime24.parquet"
export AIME25="$V2_DATA/aime25.parquet"
# Written by data/prepare_sft.py: pretokenized SFT data (input_ids + loss_mask), parquet shards in a folder. 1a's
# OpenThoughts-3 folder is named by 1-tml/sft.sh from its settings (sft_ot3_<domains>_<rows>).
export SFT_2C="$V2_DATA/sft_2c_qwen3_32b_traces"    # 2c's warm-up: Qwen3-32B's correct answers to M2RL's math prompts
mkdir -p "$LOGS" "$CKPT_ROOT" "$EXPORT_ROOT" "$V2_DATA" "$MODELS" "$JOBS" "$EVALS" "$TOOLS"

echo "skyrl : $SKYRL_DIR @ $(git -C "$SKYRL_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')   kit: $(git -C "$V2_KIT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "gpus  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ')"
echo "store : $OPD_STORE ($(df -h --output=avail "$OPD_STORE" 2>/dev/null | tail -1 | tr -d ' ') free)"
