#!/usr/bin/env python3
"""Observe real Codex HTTP retries against a local, fault-injecting Responses server."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


SCENARIOS = ("503", "429", "disconnect", "stream-close", "stream-idle")
SENSITIVE = {"authorization", "proxy-authorization", "cookie", "set-cookie"}


def event(kind, **data):
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **data}) + "\n\n").encode()


def success_stream():
    item = {"id": "msg_probe", "type": "message", "role": "assistant",
            "status": "completed", "content": [
                {"type": "output_text", "text": "probe complete", "annotations": []}]}
    response = {"id": "resp_probe", "object": "response", "created_at": int(time.time()),
                "status": "completed", "output": [item],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
    return b"".join([
        event("response.created", response={**response, "status": "in_progress", "output": []}),
        event("response.output_item.added", output_index=0,
              item={**item, "status": "in_progress", "content": []}),
        event("response.content_part.added", item_id="msg_probe", output_index=0,
              content_index=0, part={"type": "output_text", "text": "", "annotations": []}),
        event("response.output_text.delta", item_id="msg_probe", output_index=0,
              content_index=0, delta="probe complete"),
        event("response.output_text.done", item_id="msg_probe", output_index=0,
              content_index=0, text="probe complete"),
        event("response.content_part.done", item_id="msg_probe", output_index=0,
              content_index=0, part=item["content"][0]),
        event("response.output_item.done", output_index=0, item=item),
        event("response.completed", response=response),
    ])


def handler_for(scenario, fail_first, records, capture_file, lock):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.path.split("?")[0] != "/v1/responses":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with lock:
                attempt = len(records) + 1
                headers = [(k.lower(), "<redacted>" if k.lower() in SENSITIVE else v)
                           for k, v in self.headers.raw_items()]
                record = {"attempt": attempt, "time": time.time(), "path": self.path,
                          "headers": headers, "body_bytes": len(body),
                          "body_sha256": hashlib.sha256(body).hexdigest(),
                          "injected": scenario if attempt <= fail_first else "success"}
                records.append(record)
                with capture_file.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            self.close_connection = True
            try:
                if attempt <= fail_first:
                    if scenario in ("503", "429"):
                        payload = json.dumps({"error": {"message": "Injected by retry probe",
                                                       "type": "server_error"}}).encode()
                        self.send_response(int(scenario))
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    if scenario == "disconnect":
                        # Read the request, then close without any HTTP response.
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(event("response.created", response={
                        "id": f"resp_failed_{attempt}", "status": "in_progress", "output": []}))
                    self.wfile.flush()
                    if scenario == "stream-idle":
                        time.sleep(3)  # Client idle timeout is set to 1 second.
                    # EOF without response.completed exercises stream recovery.
                    return
                payload = success_stream()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def summarize(records, returncode, timed_out, fail_first):
    baseline = dict(records[0]["headers"]) if records else {}
    comparisons = []
    for record in records:
        headers = dict(record["headers"])
        comparisons.append({
            "attempt": record["attempt"], "injected": record["injected"],
            "seconds_since_first": round(record["time"] - records[0]["time"], 3),
            "same_body_as_first": record["body_sha256"] == records[0]["body_sha256"],
            "retry_named_headers": {k: v for k, v in headers.items()
                                    if re.search(r"retry|attempt", k, re.I)},
            "header_changes_from_first": {
                k: {"first": baseline.get(k), "this": headers.get(k)}
                for k in sorted(baseline.keys() | headers.keys())
                if baseline.get(k) != headers.get(k)},
        })
    return {"requests_seen": len(records), "retry_observed": len(records) > 1,
            "success_response_sent": len(records) > fail_first,
            "codex_exit_code": returncode, "timed_out": timed_out,
            "attempts": comparisons}


def run(args, scenario, output, version):
    case_dir = output / scenario
    case_dir.mkdir(parents=True, exist_ok=False)
    records = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(
        scenario, args.fail_first, records, case_dir / "requests.jsonl", threading.Lock()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="codex-retry-probe-") as tmp:
            home = Path(tmp) / "home"
            work = Path(tmp) / "work"
            home.mkdir()
            work.mkdir()
            # Isolate config, auth, hooks, MCP servers, and project instructions.
            config = f'''model_provider = "probe"
model = "gpt-5-codex"
[model_providers.probe]
name = "Local retry probe"
base_url = "http://127.0.0.1:{server.server_port}/v1"
wire_api = "responses"
env_key = "RETRY_PROBE_DUMMY_KEY"
requires_openai_auth = false
request_max_retries = {args.request_retries}
stream_max_retries = {args.stream_retries}
stream_idle_timeout_ms = 1000
supports_websockets = false
'''
            (home / "config.toml").write_text(config)
            (case_dir / "config.toml").write_text(config)
            env = dict(os.environ, CODEX_HOME=str(home), RETRY_PROBE_DUMMY_KEY="local-dummy-key")
            # Prevent inherited proxy/base-url settings from redirecting the test.
            for key in list(env):
                if key.lower().endswith("_proxy") or key in (
                    "OPENAI_BASE_URL", "OPENAI_API_KEY", "CODEX_API_KEY"):
                    env.pop(key)
            env["NO_PROXY"] = "127.0.0.1,localhost"
            command = [args.codex, "exec", "--skip-git-repo-check", "--color", "never",
                       "--sandbox", "read-only", "Reply with probe complete. Do not use tools."]
            timed_out = False
            with (case_dir / "codex.log").open("w") as log:
                proc = subprocess.Popen(command, cwd=work, env=env,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                try:
                    proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.kill()
                    proc.wait()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    result = {"scenario": scenario, "codex_version": version,
              **summarize(records, proc.returncode, timed_out, args.fail_first)}
    (case_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", default="codex", help="Codex executable or compiled OSS binary")
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--fail-first", type=int, default=2)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--stream-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60, help="Seconds per scenario")
    parser.add_argument("--output", type=Path,
                        default=Path("captures") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if args.fail_first < 1 or min(args.request_retries, args.stream_retries) < 0 or args.timeout <= 0:
        parser.error("fail-first must be >= 1, retry budgets >= 0, and timeout > 0")
    args.codex = shutil.which(args.codex)
    if not args.codex:
        parser.error("Codex executable not found; pass --codex /path/to/codex")
    version = subprocess.check_output([args.codex, "--version"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run(args, s, args.output, version)
               for s in (SCENARIOS if args.scenario == "all" else (args.scenario,))]
    print(f"\nCaptures: {args.output.resolve()}")
    # Absence of retries or successful completion makes the experiment inconclusive.
    return 0 if all(r["retry_observed"] and r["success_response_sent"]
                    and r["codex_exit_code"] == 0 and not r["timed_out"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
