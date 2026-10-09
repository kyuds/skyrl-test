#!/usr/bin/env bash
# Run on your Mac. ssh into the VM; arguments go to `gcloud compute ssh`.
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/ssh.sh                                   # a shell
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/ssh.sh --command 'nvidia-smi'            # one command
#   bash skyrl-test/opd-gcp-spot-full-repro/gcp/ssh.sh -- -L 8265:localhost:8265         # shell + Ray dashboard on localhost:8265
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
need_vm
gssh "$@"
