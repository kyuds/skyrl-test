"""Codex over a task set through skycap, in eval mode: record every rollout, grade it, train nothing.

Starts a skycap server in token mode (``--length-as-completed``) in front of the running vLLM engine, recording to
``records/<run>/``. Then, ``--concurrency`` at a time, for every task and repeat:

1. creates a trajectory, with ``task_id`` / ``instance_id`` / ``step`` in its meta, so skycap-viewer groups the
   repeats of one task the way it groups GRPO samples;
2. runs ``codex exec`` on the task in its own scratch directory, pointed at the trajectory (the same Codex settings
   as skycap-responses/04_codex.sh);
3. grades ``solution.py`` (tasks.py) and finishes the trajectory with ``reward``, ``stop_reason`` and the verdict
   as annotations, which writes its record.

Every rollout is one line in ``runs/<run>/results.jsonl``. The server stops at the end. Then run 02_analyze.sh.

    bash skyrl-test/skycap-codex-eval/01_eval.sh --tasks smoke --repeats 2
    bash skyrl-test/skycap-codex-eval/01_eval.sh --tasks codecontests --n 16 --repeats 4 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

import tasks as tasksets

EVAL_DIR = Path(os.environ["EVAL_DIR"])
RESPONSES_KIT = Path(os.environ["KIT_DIR"])
MODEL = os.environ["SKYCAP_MODEL"]
CONTEXT = int(os.environ["SKYCAP_MAX_MODEL_LEN"])

PROMPT = """Solve this programming problem in Python 3.

Write `solution.py` in the current directory. It reads the input from standard input and prints the answer to standard output. The examples are in `examples/` (`1.in` with its expected output `1.out`, and so on); run your program on them before you finish. It is judged on hidden tests with a time limit of {limit:g} seconds per test, so keep it efficient.

{statement}
"""

CODEX_CONFIG = """model = "{model}"
model_provider = "skycap"
model_context_window = {context}
web_search = "disabled"

[model_providers.skycap]
name = "skycap"
base_url = "{base_url}"
wire_api = "responses"
env_key = "SKYCAP_TRAJECTORY_KEY"
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 3600000
"""


def up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def start_skycap(run: str, port: int, log: Path) -> subprocess.Popen:
    """The eval's own skycap server, recording to records/<run>/."""
    base = f"http://127.0.0.1:{port}"
    if up(f"{base}/healthz"):
        raise SystemExit(f"something already answers on :{port}; stop it first: pkill -f 'skycap serve.*--port {port}'")
    env = {
        **os.environ,
        "SKYCAP_PORT": str(port),
        "SKYCAP_LENGTH_AS_COMPLETED": "1",
        "SKYCAP_RECORDS": os.environ["EVAL_RECORDS"],
        "SKYCAP_NAME": run,
    }
    server = subprocess.Popen(
        ["bash", str(RESPONSES_KIT / "02_skycap.sh")],
        env=env,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(120):
        if up(f"{base}/healthz"):
            return server
        if server.poll() is not None:
            raise SystemExit(f"skycap exited ({server.returncode}); see {log}")
        time.sleep(1)
    stop(server)
    raise SystemExit(f"skycap did not answer on :{port} in 2 minutes; see {log}")


def stop(process: subprocess.Popen | asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


async def rollout(
    http: httpx.AsyncClient, server: str, task: tasksets.Task, repeat: int, run: str, run_dir: Path, args, codex: str
) -> dict[str, Any]:
    name = f"{task.id}/r{repeat}"
    work = run_dir / "work" / task.id / f"r{repeat}"
    home = run_dir / "codex" / task.id / f"r{repeat}"
    home.mkdir(parents=True, exist_ok=True)
    tasksets.write_workspace(task, work)
    meta = {
        "task_id": task.id,
        "instance_id": task.id,
        "step": 0,
        "repetition_id": repeat,
        "run": run,
        "source": task.source,
        "rating": task.rating,
        "codex": codex,
    }
    created = (await http.post(f"{server}/trajectories", json={"meta": meta})).json()
    (home / "config.toml").write_text(CODEX_CONFIG.format(model=MODEL, context=CONTEXT, base_url=created["base_url"]))
    env = {**os.environ, "CODEX_HOME": str(home), "SKYCAP_TRAJECTORY_KEY": created.get("api_key") or "unused"}
    started = time.time()
    with (home / "codex.jsonl").open("w") as out, (home / "codex.log").open("w") as err:
        process = await asyncio.create_subprocess_exec(
            "codex", "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
            "--disable", "multi_agent", "--disable", "goals", "--json", "-C", str(work),
            PROMPT.format(limit=task.time_limit, statement=task.statement),
            env=env, stdout=out, stderr=err, stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )  # fmt: skip
        try:
            code = await asyncio.wait_for(process.wait(), timeout=args.timeout)
            stop_reason = "completed" if code == 0 else "error"
        except TimeoutError:
            stop(process)
            await asyncio.sleep(5)
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            code, stop_reason = None, "agent_timeout"
    seconds = round(time.time() - started, 1)
    graded = await asyncio.to_thread(tasksets.grade, task, work)
    annotations = {
        "reward": graded["reward"],
        "stop_reason": stop_reason,
        "verdict": graded["verdict"],
        "tests_passed": graded["passed"],
        "tests_total": graded["total"],
        "codex_exit": code,
        "seconds": seconds,
    }
    finished = (
        await http.post(f"{server}/trajectories/{created['id']}/finish", json={"annotations": annotations})
    ).json()
    document = (await http.get(f"{server}/trajectories/{created['id']}")).json()
    calls = [call for node in document["nodes"] for call in node["calls"]]
    result = {
        "id": created["id"],
        "task": task.id,
        "repeat": repeat,
        **annotations,
        "calls": len(calls),
        "paths": len((document.get("samples") or {}).get("rows") or []),
        "unbridged": sum(call.get("bridged") is False for call in calls),
        "truncated": sum(call.get("finish_reason") == "length" for call in calls),
        "record": (finished.get("record") or {}).get("path"),
    }
    print(
        f"{'PASS' if graded['reward'] else 'fail'} {name:<44} {stop_reason:<13} calls={result['calls']:<3} "
        f"paths={result['paths']} unbridged={result['unbridged']} {seconds:>6}s  {graded['verdict']}  {created['id']}",
        flush=True,
    )
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", choices=["codecontests", "smoke"], default="smoke")
    parser.add_argument("--n", type=int, default=16, help="codecontests: how many problems (lowest rated first)")
    parser.add_argument("--split", default="test", help="codecontests split: test (165), valid (117) or train")
    parser.add_argument("--max-rating", type=int, default=1600, help="codecontests: highest Codeforces rating")
    parser.add_argument("--max-tests", type=int, default=30, help="codecontests: tests judged per problem")
    parser.add_argument("--repeats", type=int, default=2, help="rollouts per task (a group in the viewer)")
    parser.add_argument("--concurrency", type=int, default=8, help="Codex processes at once")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per rollout before Codex is stopped")
    parser.add_argument("--run", help="run name: records/<run>/ and runs/<run>/ (default: <tasks>-<time>)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("EVAL_PORT", "8082")))
    args = parser.parse_args()

    run = args.run or f"{args.tasks}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = EVAL_DIR / "runs" / run
    if (Path(os.environ["EVAL_RECORDS"]) / run).exists():
        raise SystemExit(f"records/{run} exists already; pick another --run")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not up(f"{os.environ['SKYCAP_ENGINE_URL']}/v1/models"):
        raise SystemExit("vLLM is not up: skycap-responses/gcp/03_serve.sh (or 01_engine.sh on the node)")
    version = await asyncio.to_thread(
        subprocess.run, ["codex", "--version"], capture_output=True, text=True, check=False
    )
    codex = version.stdout.strip().splitlines()[-1]

    if args.tasks == "smoke":
        tasks = tasksets.smoke()
    else:
        tasks = tasksets.codecontests(args.split, args.n, args.max_rating, args.max_tests, EVAL_DIR / ".cache")
    planned = [(task, repeat) for task in tasks for repeat in range(args.repeats)]
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "run": run, "model": MODEL, "codex": codex, "tasks": [t.id for t in tasks]}, indent=2)
    )
    print(f"run {run}: {len(tasks)} tasks x {args.repeats} = {len(planned)} rollouts, {args.concurrency} at a time; "
          f"{MODEL} via {codex}")  # fmt: skip

    server = start_skycap(run, args.port, run_dir / "skycap.log")
    base = f"http://127.0.0.1:{args.port}"
    gate = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=120) as http:

            async def guarded(task: tasksets.Task, repeat: int) -> None:
                async with gate:
                    try:
                        result = await rollout(http, base, task, repeat, run, run_dir, args, codex)
                    except Exception as error:  # noqa: BLE001 - one rollout's failure is reported, not fatal
                        print(f"ERROR {task.id}/r{repeat}: {type(error).__name__}: {error}", flush=True)
                        result = {"task": task.id, "repeat": repeat, "error": f"{type(error).__name__}: {error}"}
                results.append(result)
                with (run_dir / "results.jsonl").open("a") as sink:
                    sink.write(json.dumps(result) + "\n")

            await asyncio.gather(*(guarded(task, repeat) for task, repeat in planned))
    finally:
        stop(server)
    solved = sum(1 for r in results if r.get("reward"))
    print(f"\n{solved}/{len(results)} rollouts accepted. Results: {run_dir / 'results.jsonl'}")
    print(f"Records: {Path(os.environ['EVAL_RECORDS']) / run}")
    print(f"Next: bash skyrl-test/skycap-codex-eval/02_analyze.sh {run}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
