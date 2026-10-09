# Finds this kit and the SkyRL checkout. Sourced by every script that runs on the VM; safe to source twice.
# Layouts supported:
#   nested:  <SkyRL>/skyrl-test/opd-gcp-spot-full-repro     (what gcp/02_setup_node.sh creates: ~/SkyRL/skyrl-test)
#   sibling: <root>/SkyRL and <root>/skyrl-test/opd-gcp-spot-full-repro
# SKYRL_DIR in the environment overrides the search. The checkout must carry both halves of the reproduction:
# the OPD entrypoint with the launched teacher (PR #2256, branch kyuds/opd-entrypoint) and the DAPO recipe.
opd_locate() {
  OPD_KIT_DIR="${OPD_KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  local up2 need f; up2="$(cd "$OPD_KIT_DIR/../.." && pwd)"
  need=(skyrl/train/entrypoints/main_opd.py skyrl/train/opd/teacher_launch.py examples/train/algorithms/dapo/main_dapo.py)
  if [[ -z "${SKYRL_DIR:-}" ]]; then
    if [[ -f "$up2/${need[0]}" ]]; then SKYRL_DIR="$up2"
    elif [[ -f "$up2/SkyRL/${need[0]}" ]]; then SKYRL_DIR="$up2/SkyRL"
    fi
  fi
  if [[ -z "${SKYRL_DIR:-}" ]]; then
    echo "cannot find a SkyRL checkout with the OPD entrypoint: looked at $up2 and $up2/SkyRL." >&2
    echo "Set SKYRL_DIR, and make sure the checkout is on the OPD branch (git checkout kyuds/opd-entrypoint)." >&2
    return 1
  fi
  for f in "${need[@]}"; do
    [[ -f "$SKYRL_DIR/$f" ]] || { echo "$SKYRL_DIR is missing $f: git -C \"$SKYRL_DIR\" checkout kyuds/opd-entrypoint" >&2; return 1; }
  done
  export OPD_KIT_DIR SKYRL_DIR
}
