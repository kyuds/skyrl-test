# Finds this kit and the SkyRL checkout. Sourced by every script; safe to source twice.
# Layouts supported:
#   sibling: <root>/SkyRL and <root>/skyrl-test/opd-4xh100   (the 4xH100 node)
#   nested:  <SkyRL>/skyrl-test/opd-4xh100                     (the workspace rule: cloned inside SkyRL/)
# SKYRL_DIR in the environment overrides the search.
opd_locate() {
  OPD_KIT_DIR="${OPD_KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  local up2; up2="$(cd "$OPD_KIT_DIR/../.." && pwd)"
  if [[ -z "${SKYRL_DIR:-}" ]]; then
    if [[ -f "$up2/skyrl/train/entrypoints/main_opd.py" ]]; then SKYRL_DIR="$up2"
    elif [[ -f "$up2/SkyRL/skyrl/train/entrypoints/main_opd.py" ]]; then SKYRL_DIR="$up2/SkyRL"
    fi
  fi
  if [[ -z "${SKYRL_DIR:-}" || ! -f "${SKYRL_DIR}/skyrl/train/entrypoints/main_opd.py" ]]; then
    echo "cannot find the SkyRL checkout with the OPD entrypoint: looked at $up2 and $up2/SkyRL${SKYRL_DIR:+ and SKYRL_DIR=$SKYRL_DIR}." >&2
    echo "Set SKYRL_DIR to the SkyRL checkout and make sure it is on the OPD branch (git checkout kyuds/opd-entrypoint)." >&2
    return 1
  fi
  export OPD_KIT_DIR SKYRL_DIR
}
