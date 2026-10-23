#!/usr/bin/env bash
# A skycap server in front of the vLLM engine from 01_engine.sh. Long-running: start it with nohup (README). It
# records every trajectory to $SKYCAP_RECORDS/<name>, where <name> is SKYCAP_NAME (default: port<SKYCAP_PORT>,
# or text-<SKYCAP_PORT> in text mode), which is the run skycap-viewer lists it under.
#
#   SKYCAP_MODE=tokens                       tokens (default): render here, sample through /inference/v1/generate;
#                                            text: forward each call to vLLM's own /v1/responses (or chat) and
#                                            record what it answered (messages only, no tokens)
#   SKYCAP_PORT=8080                         where it listens
#   SKYCAP_NAME=...                          the record directory's name
# Extra arguments go to `skycap serve`, e.g. --require-api-key or --sampling-overrides '{"top_k": 20}'.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
mode="${SKYCAP_MODE:-tokens}"
if [[ "$mode" == text ]]; then
  name="${SKYCAP_NAME:-text-$SKYCAP_PORT}"
  args=(serve --mode text --upstream-url "$SKYCAP_ENGINE_URL/v1")
else
  name="${SKYCAP_NAME:-port$SKYCAP_PORT}"
  args=(serve --mode tokens --upstream-url "$SKYCAP_ENGINE_URL" --tokenizer "$SKYCAP_MODEL" --max-model-len "$SKYCAP_MAX_MODEL_LEN")
fi
args+=(--host 127.0.0.1 --port "$SKYCAP_PORT" --record-dir "$SKYCAP_RECORDS/$name")
echo "skycap ($mode mode) on :$SKYCAP_PORT -> $SKYCAP_ENGINE_URL, records in $SKYCAP_RECORDS/$name"
cd "$SKYRL_DIR/skycap"
exec uv run --isolated --extra tokens skycap "${args[@]}" "$@"
