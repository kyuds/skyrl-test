#!/usr/bin/env bash
# Run on your Mac. ssh into a node; further arguments go to `gcloud compute ssh`.
#   bash skyrl-test/opd-blog-v2/gcp/ssh.sh h100-1                          # a shell
#   bash skyrl-test/opd-blog-v2/gcp/ssh.sh h100-1 --command 'nvidia-smi'   # one command
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
[[ $# -ge 1 ]] || { echo "usage: $0 <node> [gcloud compute ssh arguments]" >&2; exit 2; }
resolve_node "$1"
shift
gssh "$@"
