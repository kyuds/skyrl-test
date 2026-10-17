#!/usr/bin/env bash
# The real Codex CLI on one task, through a skycap trajectory: creates the trajectory, points Codex at it as a
# custom Responses provider, runs `codex exec` in a scratch repo, then finishes the trajectory and prints what
# it recorded. Needs `codex` on PATH (_tools.sh installs one).
#
#   bash skyrl-test/skycap-responses/04_codex.sh ["the task"]
#   bash skyrl-test/skycap-responses/04_codex.sh --summary <trajectory id>     # print a finished run's summary again
#
# Codex settings, as the design's binding has them (plan/skycap-dialects/codex-harness.md section 4):
# retries off (Codex sends no retry marker, so a retry would fork), a long stream idle timeout (skycap sends
# nothing until the turn is sampled), the context window set to the engine's, function tools only.
# Use a server started with SKYCAP_LENGTH_AS_COMPLETED=1: Codex resamples a cut-off reported as incomplete.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_env.sh"

# What the trajectory recorded, read from the server into $2 (a file: a real run's document is too big for argv).
summarize() {
  curl -sf "$SKYCAP_URL/trajectories/$1" -o "$2" || { echo "no trajectory $1 on $SKYCAP_URL" >&2; return 1; }
  python3 - "$2" <<'EOF'
import json, sys
document = json.load(open(sys.argv[1]))
calls = sorted((c for n in document["nodes"] for c in n["calls"]), key=lambda c: c["t_start"])
rows = (document.get("samples") or {}).get("rows") or []
print(f"paths={len(rows)} unbridged={sum(c.get('bridged') is False for c in calls)} calls={len(calls)} "
      f"finish_reasons={[c['finish_reason'] for c in calls]} dialects={sorted({str(c.get('dialect')) for c in calls})}")
print(f"tokens: prompt max {max((c['usage']['prompt_tokens'] for c in calls), default=0)}, "
      f"sampled {sum(c['usage']['completion_tokens'] for c in calls)}")
for failure in document.get("failures", []):
    print(f"failure: {failure.get('status')} {failure.get('error', '')[:300]}")
EOF
}
if [[ "${1:-}" == "--summary" ]]; then
  [[ -n "${2:-}" ]] || { echo "usage: $0 --summary <trajectory id>" >&2; exit 2; }
  summarize "$2" "$(mktemp)"
  exit
fi

command -v codex >/dev/null || { echo "codex is not on PATH: bash $KIT_DIR/_tools.sh"; exit 1; }
task="${1:-Create fizzbuzz.py that prints FizzBuzz for 1 to 20, run it, and show the output.}"
codex_version="$(codex --version 2>/dev/null | tail -1)"

run="$KIT_DIR/runs/codex-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run/home" "$run/repo"
created="$(curl -sf -X POST "$SKYCAP_URL/trajectories" -H 'Content-Type: application/json' \
  -d "{\"meta\": {\"scenario\": \"codex\", \"codex\": \"$codex_version\"}}")"
field() { python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2]) or "")' "$created" "$1"; }
trajectory="$(field id)"
base_url="$(field base_url)"
# The trajectory's own key; skycap checks it only when started with --require-api-key. Never printed.
export SKYCAP_TRAJECTORY_KEY="$(field api_key)"

cat > "$run/home/config.toml" <<EOF
model = "$SKYCAP_MODEL"
model_provider = "skycap"
model_context_window = $SKYCAP_MAX_MODEL_LEN
web_search = "disabled"

[model_providers.skycap]
name = "skycap"
base_url = "$base_url"
wire_api = "responses"
env_key = "SKYCAP_TRAJECTORY_KEY"
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 3600000
EOF
echo "trajectory $trajectory; Codex $codex_version; logs in $run"

status=0
CODEX_HOME="$run/home" codex exec \
  --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  --disable multi_agent --disable goals \
  --json -C "$run/repo" \
  "$task" > "$run/codex.jsonl" 2> "$run/codex.log" || status=$?
echo "codex exited $status"

curl -sf -X POST "$SKYCAP_URL/trajectories/$trajectory/finish" -H 'Content-Type: application/json' \
  -d "{\"annotations\": {\"scenario\": \"codex\", \"codex\": \"$codex_version\", \"codex_exit\": $status}}" \
  -o "$run/finished.json"
summarize "$trajectory" "$run/trajectory.json"
python3 -c 'import json,sys; print("record:", (json.load(open(sys.argv[1])).get("record") or {}).get("path"))' "$run/finished.json"
echo "Codex's own log: $run/codex.log; its events: $run/codex.jsonl"
