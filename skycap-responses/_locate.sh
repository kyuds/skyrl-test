# Finds this kit and the SkyRL checkout. Sourced by every script; safe to source twice.
# Layouts supported:
#   sibling: <root>/SkyRL and <root>/skyrl-test/skycap-responses   (the ~/dev/skyrl workspace)
#   nested:  <SkyRL>/skyrl-test/skycap-responses                     (cloned inside SkyRL/, as on a workspace)
# SKYRL_DIR in the environment overrides the search.
kit_locate() {
  KIT_DIR="${KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  local up2 marker="skycap/src/skycap/dialects/openai_responses.py"
  up2="$(cd "$KIT_DIR/../.." && pwd)"
  if [[ -z "${SKYRL_DIR:-}" ]]; then
    if [[ -d "$up2/skycap" ]]; then SKYRL_DIR="$up2"
    elif [[ -d "$up2/SkyRL/skycap" ]]; then SKYRL_DIR="$up2/SkyRL"
    fi
  fi
  if [[ -z "${SKYRL_DIR:-}" || ! -d "$SKYRL_DIR/skycap" ]]; then
    echo "cannot find the SkyRL checkout: looked at $up2 and $up2/SkyRL. Set SKYRL_DIR." >&2
    return 1
  fi
  if [[ ! -f "$SKYRL_DIR/$marker" ]]; then
    echo "$SKYRL_DIR has no Responses dialect ($marker):" >&2
    echo "  git -C \"$SKYRL_DIR\" checkout kyuds/skycap-responses" >&2
    return 1
  fi
  export KIT_DIR SKYRL_DIR
}
