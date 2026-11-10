#!/usr/bin/env bash
# Run on your Mac, before pushing a change to a run script or moving SKYRL_COMMIT. Builds every run's config, normal and
# --smoke, with SkyRL's own parser and validators (as the run would), and checks the GPU budget and lengths, which
# SkyRL only finds out on the node. Prints what each run will do. Needs the SkyRL checkout at the pinned commit.
#   bash skyrl-test/opd-blog-v2/common/check_configs.sh
#   SKYRL_DIR=/path/to/SkyRL bash skyrl-test/opd-blog-v2/common/check_configs.sh
set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$KIT_DIR/config.sh"
SKYRL_DIR="${SKYRL_DIR:-$(cd "$KIT_DIR/../.." && pwd)/SkyRL}"
[[ -x "$SKYRL_DIR/.venv/bin/python" ]] || { echo "no SkyRL checkout with a .venv at $SKYRL_DIR; set SKYRL_DIR" >&2; exit 2; }
head="$(git -C "$SKYRL_DIR" rev-parse HEAD)"
if [[ "$head" != "$SKYRL_COMMIT" ]]; then
  echo "WARNING: $SKYRL_DIR is at ${head:0:10}, the VMs run ${SKYRL_COMMIT:0:10}. The check is only as good as that match." >&2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
files=()
while read -r node run script; do
  [[ -n "$node" ]] || continue
  for variant in "" --smoke; do
    out="$tmp/${run}${variant:+_smoke}"
    V2_DRY_RUN=1 bash "$KIT_DIR/$script" $variant > "$out"
    files+=("$out")
  done
done <<<"$V2_RUNS"
cd "$SKYRL_DIR"
set +e
.venv/bin/python -I -B "$KIT_DIR/common/check_configs.py" "$KIT_DIR" "${files[@]}" 2>&1 | grep -v -E '^\s*$|UserWarning|warnings.warn'
exit "${PIPESTATUS[0]}"
