#!/usr/bin/env bash
# Runs on the VM. Writes the training and eval data (data/README.md). Every run script calls it first; it skips the
# work when the files are already there, so it costs nothing on a second call. By hand, from ~/SkyRL (the first call
# downloads about 8 GB):
#   bash skyrl-test/opd-blog-v2/data/prepare.sh
set -euo pipefail
source "$(dirname "$0")/../common/00_env.sh" >/dev/null
cd "$SKYRL_DIR"   # uv resolves the project (and its extras) from the cwd

if [[ -f "$MATH_TRAIN" && -f "$MOPD_TRAIN" && -f "$AIME24" && -f "$AIME25" ]]; then
  echo "data already prepared in $V2_DATA"
else
  uv run --isolated --extra fsdp python "$V2_KIT/data/prepare_data.py" --out "$V2_DATA"
fi
