#!/usr/bin/env bash
# vLLM serving $SKYCAP_MODEL with the token-in/token-out route skycap calls (/inference/v1/generate).
# Long-running: start it with nohup (README). Extra arguments go to `vllm serve`, e.g. --tensor-parallel-size 2.
#
# --enable-scale-out registers /inference/v1/generate on `vllm serve` (vLLM 0.30's
# vllm/entrypoints/launchers/cli_args.py; SkyRL pins 0.31.0, assumed to keep the flag). Token mode needs no tool
# or reasoning parser: skycap renders and parses with `renderers` itself, and that route sees only token ids.
# Text mode does: its upstream is this server's own /v1/responses, which returns reasoning and function_call items
# only with them. SKYCAP_ENGINE_PARSERS sets them (Qwen3's by default; empty for none); they touch only vLLM's
# OpenAI routes, never /inference/v1/generate.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
cd "$SKYRL_DIR"
parsers="${SKYCAP_ENGINE_PARSERS---reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes}"
# shellcheck disable=SC2086 # a list of flags
exec uv run --isolated --extra fsdp vllm serve "$SKYCAP_MODEL" \
  --host 127.0.0.1 --port "$SKYCAP_ENGINE_PORT" \
  --max-model-len "$SKYCAP_MAX_MODEL_LEN" \
  --enable-scale-out \
  $parsers \
  "$@"
