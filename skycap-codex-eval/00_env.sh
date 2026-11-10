# Sourced by this kit's scripts. Builds on skycap-responses (its 00_env.sh finds SkyRL, uv, Node, Codex and the
# viewer, and sets SKYCAP_MODEL, SKYCAP_MAX_MODEL_LEN and the engine URL), and adds this kit's own knobs.
# The eval runs its own skycap server (EVAL_PORT) and viewer (EVAL_VIEWER_PORT), so its records stay apart from
# skycap-responses' :8080 and :8090 ones.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash"; return 1 2>/dev/null || exit 1; }
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EVAL_DIR/../skycap-responses/00_env.sh" || { return 1 2>/dev/null || exit 1; }
export EVAL_DIR
export EVAL_PORT="${EVAL_PORT:-8082}"
export EVAL_VIEWER_PORT="${EVAL_VIEWER_PORT:-8766}"
# One record directory per eval run: records/<run>/, which skycap-viewer lists as its own run.
export EVAL_RECORDS="${EVAL_RECORDS:-$EVAL_DIR/records}"
mkdir -p "$EVAL_RECORDS" "$EVAL_DIR/runs"
