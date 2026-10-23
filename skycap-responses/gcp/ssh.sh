#!/usr/bin/env bash
# Run on your Mac. ssh into the VM; arguments go to `gcloud compute ssh`.
#   bash skyrl-test/skycap-responses/gcp/ssh.sh                                   # a shell
#   bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'nvidia-smi'            # one command
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
gssh "$@"
