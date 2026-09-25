# Source this on the Anyscale workspace, from the SkyRL checkout:
#   source skyrl-test/anyscale-pr2198/00_env.sh
# Mirrors the env_vars block of SkyRL/ci/anyscale_gpu_e2e_test.yaml. Never prints the key.
# Set WANDB_API_KEY first: Workspace > Dependencies > Environment variables (cluster-wide), or
# `export WANDB_API_KEY=...` in this shell -- SkyRL forwards it into the Ray runtime env itself
# (skyrl/train/utils/utils.py, prepare_runtime_environment), so the driver shell is enough.

: "${WANDB_API_KEY:?export WANDB_API_KEY first (see the comment at the top of this file)}"

# Every ci/anyscale_*.yaml sets this so SkyRL's own ray.init(runtime_env=...) wins over the
# job-level runtime env Anyscale installs. Assumed to matter for workspaces the same way.
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
# The install doc's "configure Ray to use uv" line: Ray workers get the same uv environment the
# driver runs in. Harmless if the image already sets it.
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Single-node workspace (head = g6.12xlarge, 4xL4): $HOME is visible to every Ray worker.
# If the head node has no GPUs, point this at shared storage instead, e.g. /mnt/cluster_storage/data.
export PR2198_DATA="${PR2198_DATA:-$HOME/data}"
export PR2198_PROJECT="${PR2198_PROJECT:-pr2198_smoke}"

echo "branch : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "uv     : $(uv --version 2>/dev/null || echo MISSING)"
echo "gpus   :"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
echo "wandb  : key set (${#WANDB_API_KEY} chars)"
echo "data   : $PR2198_DATA   project: $PR2198_PROJECT"
