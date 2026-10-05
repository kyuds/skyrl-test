# Source this on the VM, from bash, in the shell you launch the runs from:
#   source skyrl-test/opd-gcp-spot-full-repro/00_env.sh
# Finds the SkyRL checkout around this kit (~/SkyRL/skyrl-test, as gcp/02_setup_node.sh lays it out), loads the
# node's NCCL settings and your keys, checks that Ray is up, and exports the knobs the run scripts read.
# Never prints a key.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash (it uses BASH_SOURCE to find itself)"; return 1 2>/dev/null || exit 1; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_locate.sh"
opd_locate || return 1 2>/dev/null || exit 1

# ~/.opd_cluster_env: written by gcp/node_setup.sh (uv on PATH, NCCL over plain sockets for a single node).
# ~/.opd_secrets:     written by gcp/04_push_secrets.sh from your Mac's environment (WANDB_API_KEY, HF_TOKEN).
# Both are optional here: exporting the same variables by hand works as well.
[[ -f "$HOME/.opd_cluster_env" ]] && source "$HOME/.opd_cluster_env"
[[ -f "$HOME/.opd_secrets" ]] && source "$HOME/.opd_secrets"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv is not on PATH: run gcp/02_setup_node.sh from your Mac (or install uv)"; return 1 2>/dev/null || exit 1; }
: "${WANDB_API_KEY:?export WANDB_API_KEY first (or run gcp/04_push_secrets.sh from your Mac)}"

# A finished run uploads its final HF export to the Hub as a public repo under $HF_USER (upload_export.sh).
# OPD_UPLOAD=0 turns that off, and then no HF token is needed: the base models are public.
export OPD_UPLOAD="${OPD_UPLOAD:-1}"
export HF_USER="${HF_USER:-kyuds}"
if [[ "$OPD_UPLOAD" == "1" ]]; then
  : "${HF_TOKEN:?export HF_TOKEN (a write token) first, or OPD_UPLOAD=0 to skip the uploads}"
fi

# Ray workers get the same uv environment the driver runs in (install doc's "configure Ray to use uv").
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# The runs join the node's Ray cluster (gcp/03_start_ray.sh starts it from SkyRL's own environment, so its Ray
# is the lockfile's Ray and nothing needs a version override). Ray's ledger is what keeps the OPD teacher and
# the student on different GPUs; the scripts only pass counts.
if [[ -z "${RAY_ADDRESS:-}" && ! -f /tmp/ray/ray_current_cluster ]]; then
  echo "no Ray cluster is running on this node (no /tmp/ray/ray_current_cluster, RAY_ADDRESS unset)."
  echo "From your Mac: bash skyrl-test/opd-gcp-spot-full-repro/gcp/03_start_ray.sh"
  return 1 2>/dev/null || exit 1
fi

# Everything a run must keep lives under OPD_STORE, on the boot disk: a preempted spot VM is stopped, its boot
# disk survives and the run resumes from its last checkpoint. The local NVMe array (/mnt/local_storage) is wiped
# by a preemption, so it only holds caches (~/.cache points at it).
export OPD_STORE="${OPD_STORE:-$HOME/opd-store}"
export OPD_DATA="${OPD_DATA:-$OPD_STORE/data}"
export OPD_LOGS="${OPD_LOGS:-$OPD_STORE/logs}"
export OPD_PROJECT="${OPD_PROJECT:-opd_gcp_spot_full_repro}"
mkdir -p "$OPD_DATA" "$OPD_LOGS" || { echo "cannot create $OPD_STORE"; return 1 2>/dev/null || exit 1; }

# The node's 8 GPUs, by count only (Ray picks the devices). A DAPO run takes all 8. An OPD run splits them:
# the student trains on OPD_NUM_STUDENT_GPUS (colocated FSDP ranks + one vLLM engine per GPU) and the run
# launches the teacher on OPD_TEACHER_NUM_GPUS more (one TP-1 server each). So the runs go one at a time.
export OPD_DAPO_NUM_GPUS="${OPD_DAPO_NUM_GPUS:-8}"
export OPD_NUM_STUDENT_GPUS="${OPD_NUM_STUDENT_GPUS:-4}"
export OPD_TEACHER_NUM_GPUS="${OPD_TEACHER_NUM_GPUS:-4}"

echo "kit     : $OPD_KIT_DIR"
echo "skyrl   : $SKYRL_DIR @ $(git -C "$SKYRL_DIR" rev-parse --abbrev-ref HEAD) $(git -C "$SKYRL_DIR" rev-parse --short HEAD)"
echo "uv      : $(uv --version 2>/dev/null || echo MISSING)"
echo "ray     : cluster at $(cat /tmp/ray/ray_current_cluster 2>/dev/null || echo "${RAY_ADDRESS:-?}")"
echo "gpus    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ' || echo 'nvidia-smi failed')"
echo "split   : DAPO $OPD_DAPO_NUM_GPUS | OPD student $OPD_NUM_STUDENT_GPUS + teacher $OPD_TEACHER_NUM_GPUS"
echo "nccl    : NCCL_NET=${NCCL_NET:-<unset>} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-<unset>}"
echo "store   : $OPD_STORE  ($(df -h --output=avail "$OPD_STORE" 2>/dev/null | tail -1 | tr -d ' ' || echo '?') free)"
echo "wandb   : key set (${#WANDB_API_KEY} chars)   project: $OPD_PROJECT"
if [[ "$OPD_UPLOAD" == "1" ]]; then echo "hub     : token set (${#HF_TOKEN} chars)   uploads go to $HF_USER/opd-gcp-spot-full-repro-<run>-step<N>, public"
else echo "hub     : OPD_UPLOAD=0, nothing is uploaded"; fi
