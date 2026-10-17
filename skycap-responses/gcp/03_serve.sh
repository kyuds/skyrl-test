#!/usr/bin/env bash
# Run on your Mac, after 02_setup_node.sh and again after every preemption or reboot. Starts, on the VM and
# detached: vLLM (the kit's 01_engine.sh) and two skycap servers in token mode (02_skycap.sh), :8080 plain and
# :8081 with --length-as-completed, then waits until all three answer. Skips what is already up.
#   bash skyrl-test/skycap-responses/gcp/03_serve.sh
#   bash skyrl-test/skycap-responses/gcp/03_serve.sh --restart       # stop all three first
#   bash skyrl-test/skycap-responses/gcp/03_serve.sh --stop          # stop them and leave them stopped
# The kit's knobs pass through: SKYCAP_MODEL, SKYCAP_MAX_MODEL_LEN, and SKYCAP_ENGINE_ARGS (extra `vllm serve`
# flags), e.g. on an 8-GPU node:
#   SKYCAP_MODEL=Qwen/Qwen3-32B SKYCAP_ENGINE_ARGS="--tensor-parallel-size 4" bash skyrl-test/skycap-responses/gcp/03_serve.sh --restart
# The first start downloads the model (a few minutes for an 8B one).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
push_setup
case "${1:-}" in
  "") phase=serve ;;
  --restart) phase="serve --restart" ;;
  --stop) phase=unserve ;;
  *) echo "usage: $0 [--restart | --stop]" >&2; exit 2 ;;
esac
env=""
for name in SKYCAP_MODEL SKYCAP_MAX_MODEL_LEN SKYCAP_ENGINE_ARGS; do
  if [[ -n "${!name:-}" ]]; then env="$env $name=$(printf %q "${!name}")"; fi
done
gssh --command "${env# } bash ~/skycap_node_setup.sh $phase"
[[ "$phase" == unserve ]] || echo "Next: bash skyrl-test/skycap-responses/gcp/tunnel.sh, then drive from here (README, 'On GCP')"
