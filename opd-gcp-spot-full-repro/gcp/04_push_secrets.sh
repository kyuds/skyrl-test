#!/usr/bin/env bash
# Run on your Mac, once per VM. Copies WANDB_API_KEY and HF_TOKEN from this shell's environment into
# ~/.opd_secrets on the VM (mode 600), which 00_env.sh sources. The values travel over ssh's stdin: they are
# never on a command line, never printed, and never in this repo. The file is on the boot disk, so it
# survives a preemption. HF_TOKEN must be a write token (the runs upload their exports).
#   source ~/.zprofile; export HF_TOKEN=...   # however you keep them
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/04_push_secrets.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
: "${WANDB_API_KEY:?export WANDB_API_KEY in this shell first}"
: "${HF_TOKEN:?export HF_TOKEN (a Hugging Face write token) in this shell first}"
need_vm
printf 'export WANDB_API_KEY=%q\nexport HF_TOKEN=%q\n' "$WANDB_API_KEY" "$HF_TOKEN" |
  gssh --command 'umask 077; cat > ~/.opd_secrets; chmod 600 ~/.opd_secrets; echo "wrote ~/.opd_secrets on $(hostname): $(wc -l < ~/.opd_secrets | tr -d " ") keys"'
