#!/usr/bin/env bash
# Installs what the kit needs besides SkyRL into $SKYCAP_TOOLS (skyrl-test/skycap-responses/.tools, ignored by
# git): Node 22 (from nodejs.org), the Codex CLI (npm @openai/codex, version CODEX_VERSION, default latest), and
# skycap-viewer (a clone of github.com/NovaSky-AI/skycap-viewer). Skips what is already there.
#
#   CODEX_VERSION=0.154.0 bash skyrl-test/skycap-responses/_tools.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
NODE_VERSION="${NODE_VERSION:-22.15.0}"
mkdir -p "$SKYCAP_TOOLS"

if ! command -v node >/dev/null || ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a==22&&b>=15)?0:1)'; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) platform=linux-x64 ;;
    Linux-aarch64) platform=linux-arm64 ;;
    Darwin-arm64) platform=darwin-arm64 ;;
    Darwin-x86_64) platform=darwin-x64 ;;
    *) echo "no Node build for $(uname -s)-$(uname -m)"; exit 1 ;;
  esac
  echo "installing Node $NODE_VERSION ($platform) into $SKYCAP_TOOLS/node"
  rm -rf "$SKYCAP_TOOLS/node"
  curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-$platform.tar.gz" | tar -xz -C "$SKYCAP_TOOLS"
  mv "$SKYCAP_TOOLS/node-v$NODE_VERSION-$platform" "$SKYCAP_TOOLS/node"
  export PATH="$SKYCAP_TOOLS/node/bin:$PATH"
fi
echo "node $(node --version)"

if ! command -v codex >/dev/null; then
  echo "installing @openai/codex@${CODEX_VERSION:-latest}"
  npm install -g --prefix "$SKYCAP_TOOLS/node" "@openai/codex@${CODEX_VERSION:-latest}"
fi
echo "codex: $(codex --version 2>/dev/null | tail -1)"

if [[ ! -d "$SKYCAP_TOOLS/skycap-viewer" ]]; then
  git clone --depth 1 https://github.com/NovaSky-AI/skycap-viewer "$SKYCAP_TOOLS/skycap-viewer"
fi
echo "skycap-viewer $(git -C "$SKYCAP_TOOLS/skycap-viewer" rev-parse --short HEAD)"
