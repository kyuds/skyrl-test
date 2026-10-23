# Source this before running the kit, from anywhere:
#   source skyrl-test/skycap-responses/00_env.sh
# Finds the SkyRL checkout (it must be on kyuds/skycap-responses), checks uv, and exports the knobs the
# other scripts read. Override any of them before sourcing, e.g. SKYCAP_MODEL=Qwen/Qwen3.5-0.8B.
[[ -n "${BASH_VERSION:-}" ]] || { echo "source this file from bash (it uses BASH_SOURCE to find itself)"; return 1 2>/dev/null || exit 1; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_locate.sh"
kit_locate || return 1 2>/dev/null || exit 1
# uv's installer puts it in ~/.local/bin, which a non-interactive shell (gcloud compute ssh --command) lacks.
command -v uv >/dev/null || [[ ! -x "$HOME/.local/bin/uv" ]] || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv is not on PATH"; return 1 2>/dev/null || exit 1; }

# The model vLLM serves and skycap renders for. Qwen3's template has no `developer` role, so Codex-shaped
# requests exercise the developer fold; Qwen3.5's renderer raises on `developer` without it.
export SKYCAP_MODEL="${SKYCAP_MODEL:-Qwen/Qwen3-8B}"
export SKYCAP_MAX_MODEL_LEN="${SKYCAP_MAX_MODEL_LEN:-32768}"
# vLLM, on this node.
export SKYCAP_ENGINE_PORT="${SKYCAP_ENGINE_PORT:-8000}"
export SKYCAP_ENGINE_URL="${SKYCAP_ENGINE_URL:-http://127.0.0.1:$SKYCAP_ENGINE_PORT}"
# The skycap server the driver talks to. 02_skycap.sh takes SKYCAP_PORT per server, so several servers
# (e.g. token mode and text mode) can run side by side.
export SKYCAP_PORT="${SKYCAP_PORT:-8080}"
export SKYCAP_URL="${SKYCAP_URL:-http://127.0.0.1:$SKYCAP_PORT}"
# Every server writes records under here, one directory per server: what skycap-viewer reads.
export SKYCAP_RECORDS="${SKYCAP_RECORDS:-$KIT_DIR/records}"
# Node, Codex and skycap-viewer, if _tools.sh installed them here.
export SKYCAP_TOOLS="${SKYCAP_TOOLS:-$KIT_DIR/.tools}"
[[ -d "$SKYCAP_TOOLS/node/bin" ]] && export PATH="$SKYCAP_TOOLS/node/bin:$PATH"
mkdir -p "$SKYCAP_RECORDS" "$KIT_DIR/runs"
echo "skycap-responses: SkyRL at $SKYRL_DIR ($(git -C "$SKYRL_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)), model $SKYCAP_MODEL"
