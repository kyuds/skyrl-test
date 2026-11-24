#!/usr/bin/env bash
# Runs on the VM. Writes the data (data/README.md). Each part is skipped when it is already there, so a second call
# costs nothing; the run scripts call it first. From ~/SkyRL:
#   bash skyrl-test/opd-blog-v2/data/prepare.sh            # RL prompts and AIME: the blend, DeepMath (~8 GB down)
#   bash skyrl-test/opd-blog-v2/data/prepare.sh sft-ot3 <out> <domains> <rows>
#                                       # OpenThoughts-3 for SFT (28 GB down, ~20 min); 1-tml/sft.sh passes its settings
#   bash skyrl-test/opd-blog-v2/data/prepare.sh sft-ot3-smoke <out>   # 512 rows of the first shard, any domain
set -euo pipefail
source "$(dirname "$0")/../common/00_env.sh" >/dev/null
cd "$SKYRL_DIR"   # uv resolves the project (and its extras) from the cwd
what="${1:-rl}"

if [[ -f "$MATH_TRAIN" && -f "$MOPD_TRAIN" && -f "$DEEPMATH_TRAIN" && -f "$AIME24" && -f "$AIME25" ]]; then
  echo "RL data already prepared in $V2_DATA"
else
  uv run --isolated --extra fsdp python "$V2_KIT/data/prepare_data.py" --out "$V2_DATA"
fi

case "$what" in
  rl) ;;
  sft-ot3)
    out="${2:?out folder}"; domains="${3-}"; rows="${4:?rows}"
    if [[ -f "$out.stats.json" ]]; then
      echo "SFT data already prepared in $out: $(tr -d '\n ' < "$out.stats.json" | cut -c1-200)"
    else
      uv run --isolated --extra fsdp python "$V2_KIT/data/prepare_sft.py" ot3 --out "$out" --domains "$domains" --rows "$rows"
    fi ;;
  sft-ot3-smoke)
    out="${2:?out folder}"
    [[ -f "$out.stats.json" ]] ||
      uv run --isolated --extra fsdp python "$V2_KIT/data/prepare_sft.py" ot3 --out "$out" --rows 512 --max-shards 1 ;;
  *) echo "unknown part: $what (rl, sft-ot3 or sft-ot3-smoke)" >&2; exit 2 ;;
esac
