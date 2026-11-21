#!/usr/bin/env bash
# Run on your Mac, before pushing a change to a run script or moving SKYRL_COMMIT. Builds every job's config, normal and
# --smoke, with SkyRL's own parser and validators (as the job would): the OPD runs' and the SFT runs'. It checks the GPU
# budget and lengths, which SkyRL only finds out on the node, and prints what each job will do (evals included).
# Needs the SkyRL checkout at the pinned commit.
#   bash skyrl-test/opd-blog-v2/check_configs.sh
#   SKYRL_DIR=/path/to/SkyRL bash skyrl-test/opd-blog-v2/check_configs.sh
set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
while read -r stage node job deps script args; do
  for variant in "" --smoke; do
    out="$tmp/${job}${variant:+_smoke}"
    V2_DRY_RUN=1 bash "$KIT_DIR/$script" $args $variant > "$out"
    files+=("$out")
  done
done < <(v2_jobs)
cd "$SKYRL_DIR"
set +e
.venv/bin/python -I -B "$KIT_DIR/check_configs.py" "$KIT_DIR" "${files[@]}" 2>&1 | grep -v -E '^\s*$|UserWarning|warnings.warn'
exit "${PIPESTATUS[0]}"
