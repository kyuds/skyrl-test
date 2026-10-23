#!/usr/bin/env bash
# Run ON the VM (or any GPU node with skycap-responses' vLLM up). Codex over a task set through its own skycap
# server, in eval mode; every option is 01_eval.py's (--help). Long-running: start it with nohup (README).
#   bash skyrl-test/skycap-codex-eval/01_eval.sh --tasks smoke --repeats 2
#   bash skyrl-test/skycap-codex-eval/01_eval.sh --tasks codecontests --n 16 --repeats 4 --concurrency 8
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"
command -v codex >/dev/null || { echo "codex is not on PATH: bash $KIT_DIR/_tools.sh"; exit 1; }
cd "$EVAL_DIR"
# --no-project: SkyRL's pyproject.toml is a parent directory, and this needs none of it.
exec uv run --no-project --isolated --with httpx --with pyarrow python 01_eval.py "$@"
