# Codex retry-header probe

A Python standard-library POC that runs **the real Codex binary** against a local
fake Responses API. The server records incoming headers, deliberately fails the
first two requests, then returns a canned successful SSE response. Python does
not implement client retries: every captured retry comes from Codex.

## Run

Requires Python 3.9+ and `codex` on PATH. No pip install, login, or API key needed.

```sh
python3 retry_probe.py
```

Test a particular failure or your own compiled OSS binary:

```sh
python3 retry_probe.py --scenario disconnect
python3 retry_probe.py --codex /absolute/path/to/codex-rs/target/debug/codex
python3 retry_probe.py --scenario 503 --fail-first 3 --request-retries 4
```

The default matrix is:

| Scenario | Injected failure |
| --- | --- |
| `503` | HTTP 503 before streaming starts |
| `429` | HTTP 429 before streaming starts; some builds may not retry it |
| `disconnect` | Read the request, close the connection without an HTTP response |
| `stream-close` | HTTP 200 + `response.created`, then EOF before completion |
| `stream-idle` | HTTP 200 + `response.created`, then stall past the 1s idle timeout |

Each scenario starts a fresh Codex process and localhost server. The retry budgets
default to 3 each for HTTP requests and stream recovery. `--timeout` defaults to
60 seconds per scenario; `--help` lists all options. A model name is needed by
Codex's configuration, but no real model is invoked.

## Read the evidence

The script prints JSON and writes `captures/<timestamp>/<scenario>/`:

- `requests.jsonl`: every received request in arrival order, including all header
  pairs (lowercased names, duplicates preserved), timestamp, path, injected fault,
  and SHA-256 of the exact request body bytes. Authorization/cookies are redacted;
  bodies are not saved.
- `summary.json`: Codex version, request count, completion/timeout status,
  headers containing `retry` or `attempt`, and **all header value changes** relative
  to the first request, including added/removed headers.
- `codex.log`: Codex output, reconnect messages, and errors.
- `config.toml`: exact temporary provider configuration used.

For example, three requests with `503`, `503`, `success`, empty
`retry_named_headers`, and empty `header_changes_from_first` show that retries
occurred and all captured headers remained identical. A request ID that stays
constant is not an explicit retry marker. Inspect the full captures for other
possible header names or semantics.

One request means no retry was observed; do not interpret that as proof that
retries lack headers. Exit status is 0 only when **every selected scenario**
observed a retry, reached the success response, and Codex exited successfully.
Otherwise it is 1, and captures/logs explain which scenario was inconclusive.
An all-scenario run can therefore exit 1 solely because the build does not retry
429 responses.

## Observed locally

Tested with `codex-cli 0.163.0-alpha.10.2`:

| Scenario | Requests | Outcome | Header changes on retries |
| --- | --- | --- | --- |
| `503` | 3 | Completed | None |
| `429` | 1 | Failed without retry | No retry to compare |
| `disconnect` | 3 | Completed | None |
| `stream-close` | 3 | Completed | None |
| `stream-idle` | 3 | Completed | None |

No `retry`/`attempt` header appeared in any capture. Stream recovery changed the
raw body hash; HTTP 503 and disconnect retries preserved it. Raw hash differences
do not establish a semantic body change (for example, serialization can differ).
The saved local evidence is under `captures/verified/` after this development run.

## Scope

This measures HTTP/SSE retries on the custom-provider path of the binary you
select. WebSockets are disabled. It does not establish behavior for WebSocket
reconnects, ChatGPT authentication refresh, DNS/TLS/connect failures, partial
generated-text recovery, or every possible header convention. Test a compiled
OSS release to make a claim about that release; the desktop-bundled binary may
differ from OSS.

Inference traffic goes to `127.0.0.1` with a dummy key. An isolated temporary
`CODEX_HOME` and working directory avoid your normal config/auth/project setup;
they are deleted after each case. Codex may still attempt unrelated background
requests (this build attempted a featured-plugin fetch); this script is not a
network sandbox. Capture directories are gitignored.

The provider settings (`base_url`, `request_max_retries`, `stream_max_retries`,
`stream_idle_timeout_ms`, `supports_websockets`) are documented in the
[official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
