"""Drive Responses conversations through a running skycap server and check what each one recorded.

One trajectory per scenario and parameter combination. Each is created, driven (scenarios.py), finished with
its scenario and parameters as annotations (so skycap-viewer can filter on them), and checked against what
the scenario expects: the number of paths, unbridged calls, finish reasons, and its own checks. Prints one
line per trajectory and writes them all to runs/<time>.jsonl.

    uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --config configs/smoke.json
    uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py \
        --scenario tools --set stream=false,true --set replay=verbatim,strip --repeats 2
    ... 03_drive.py --list

A config is JSON:

    {"defaults": {"temperature": 0.7},             parameters every run starts from
     "repeats": 1,
     "runs": [{"scenario": "tools",
               "params": {"max_output_tokens": 2048},
               "grid": {"stream": [false, true]},   every combination is its own trajectory
               "repeats": 2}]}

``--set key=v1,v2`` adds (or replaces) a grid axis on every run; values parse as JSON, else as strings.
``--server`` picks the skycap server; a ``server`` parameter in a grid runs the same scenarios on several
(e.g. one with --length-as-completed and one without).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import httpx
import openai

from scenarios import SCENARIOS, Session

KIT = Path(__file__).resolve().parent


def value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_sets(sets: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for item in sets:
        key, _, values = item.partition("=")
        if not key or not values:
            raise SystemExit(f"--set takes key=v1,v2 (got {item!r})")
        grid[key] = [value(v) for v in values.split(",")]
    return grid


def combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


def plan(config: dict[str, Any], sets: dict[str, list[Any]], only: list[str], repeats: int | None) -> list[tuple[str, dict]]:
    runs = config.get("runs") or [{"scenario": name} for name in only]
    planned = []
    for run in runs:
        name = run["scenario"]
        if only and name not in only:
            continue
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; --list shows them")
        grid = {**run.get("grid", {}), **sets}
        for combo in combinations(grid):
            for _ in range(repeats or run.get("repeats") or config.get("repeats") or 1):
                planned.append((name, {**config.get("defaults", {}), **run.get("params", {}), **combo}))
    return planned


async def run_one(http: httpx.AsyncClient, name: str, params: dict[str, Any], default_server: str) -> dict[str, Any]:
    server = str(params.get("server") or default_server).rstrip("/")
    capture = (await http.get(f"{server}/healthz")).json().get("capture", {})
    # The served model: this shell's SKYCAP_MODEL, else the tokenizer the server renders with (token mode's
    # model name), e.g. when driving a remote server from a laptop through a tunnel. Text mode names none; the
    # upstream's model list (through the trajectory) does, below.
    params = {"model": os.environ.get("SKYCAP_MODEL") or capture.get("tokenizer"), **params}
    created = (await http.post(f"{server}/trajectories", json={"meta": {"scenario": name, "params": params}})).json()
    llm = openai.AsyncOpenAI(
        base_url=created["base_url"],
        api_key=created.get("api_key") or "unused",
        max_retries=0,
        timeout=float(params.get("timeout", 1800)),
    )
    if not params["model"]:
        params["model"] = (await llm.models.list()).data[0].id
    session = Session(llm=llm, params=params, created=created, http=http, server=server, capture=capture)
    started, error = time.time(), None
    try:
        await SCENARIOS[name](session)
    except Exception as caught:  # noqa: BLE001 - reported per trajectory
        error = f"{type(caught).__name__}: {caught}"
        traceback.print_exc()
    finally:
        await llm.close()
    annotations = {
        "scenario": name,
        "ok": error is None and not session.failed_checks,
        **{key: v if isinstance(v, str | int | float | bool) else json.dumps(v) for key, v in params.items()},
    }
    # The scenario's notes (e.g. each refusal's 400 message), so the viewer shows what left no node behind.
    if session.notes:
        annotations["notes"] = " | ".join(session.notes)
    finished = (await http.post(f"{server}/trajectories/{created['id']}/finish", json={"annotations": annotations})).json()
    document = (await http.get(f"{server}/trajectories/{created['id']}")).json()
    calls = sorted((call for node in document.get("nodes", []) for call in node["calls"]), key=lambda c: c["t_start"])
    result = {
        "id": created["id"],
        "scenario": name,
        "params": params,
        "server": server,
        "seconds": round(time.time() - started, 1),
        "status": finished.get("status"),
        "paths": len(finished.get("samples") or []),
        "unbridged": finished.get("unbridged_calls"),
        "calls": len(calls),
        "finish_reasons": [call.get("finish_reason") for call in calls],
        "dialects": sorted({str(call.get("dialect")) for call in calls}),
        "failures": [failure.get("error", "")[:200] for failure in document.get("failures", [])],
        "record": (finished.get("record") or {}).get("path"),
        "error": error,
        "notes": session.notes,
    }
    problems = list(session.failed_checks)
    if error:
        problems.append(error)
    if session.expect_paths is not None and result["paths"] != session.expect_paths:
        problems.append(f"{result['paths']} paths, expected {session.expect_paths}")
    if session.expect_unbridged is not None and result["unbridged"] != session.expect_unbridged:
        problems.append(f"{result['unbridged']} unbridged calls, expected {session.expect_unbridged}")
    if session.expect_finish_reasons is not None:
        got = result["finish_reasons"]
        want = session.expect_finish_reasons
        if len(got) != len(want) or any(w is not None and g != w for g, w in zip(got, want, strict=False)):
            problems.append(f"finish reasons {got}, expected {want}")
    if result["calls"] and result["dialects"] != ["openai_responses"]:
        problems.append(f"calls recorded as {result['dialects']}")
    result["problems"] = problems
    return result


def line(result: dict[str, Any], keys: list[str]) -> str:
    shown = " ".join(f"{k}={json.dumps(result['params'].get(k))}" for k in keys if k in result["params"])
    verdict = "ok   " if not result["problems"] else "CHECK"
    return (
        f"{verdict} {result['scenario']:<11} {shown:<40} paths={result['paths']} unbridged={result['unbridged']} "
        f"calls={result['calls']} {result['seconds']:>6}s  {result['id']}"
        + "".join(f"\n        - {problem}" for problem in result["problems"])
        + "".join(f"\n        . {note}" for note in result["notes"])
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="a JSON config (configs/)")
    parser.add_argument("--scenario", action="append", default=[], help="run only these (repeatable)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=V1,V2", help="a grid axis on every run")
    parser.add_argument("--repeats", type=int, default=None, help="override every run's repeats")
    parser.add_argument("--server", default=os.environ.get("SKYCAP_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--concurrency", type=int, default=4, help="trajectories driven at once")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    args = parser.parse_args()
    if args.list:
        for name, run in SCENARIOS.items():
            print(f"{name:<12} {(run.__doc__ or '').strip().splitlines()[0]}")
        return 0
    config_path = args.config if args.config is None or args.config.exists() else KIT / args.config
    config = json.loads(config_path.read_text()) if config_path else {}
    if not config and not args.scenario:
        raise SystemExit("give --config or --scenario (--list shows the scenarios)")
    planned = plan(config, parse_sets(args.set), args.scenario, args.repeats)
    varied = sorted({k for _, params in planned for k in params if len({json.dumps(p.get(k)) for _, p in planned}) > 1})
    print(f"{len(planned)} trajectories against {args.server}")

    out = KIT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60) as http:
        health = (await http.get(f"{args.server.rstrip('/')}/healthz")).json()
        print(f"server capture: {json.dumps(health.get('capture'))}")

        async def guarded(name: str, params: dict[str, Any]) -> None:
            async with gate:
                result = await run_one(http, name, params, args.server)
            results.append(result)
            with out.open("a") as sink:
                sink.write(json.dumps(result) + "\n")
            print(line(result, varied), flush=True)

        await asyncio.gather(*(guarded(name, params) for name, params in planned))
    bad = [r for r in results if r["problems"]]
    records = sorted({str(Path(r["record"]).parent) for r in results if r.get("record")})
    print(f"\n{len(results) - len(bad)}/{len(results)} as expected. Results: {out}")
    if records:
        print("Records: " + ", ".join(records))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
