#!/usr/bin/env python
"""Probe W&B ``step`` semantics for async-eval logging (wandb 0.28.1, SkyRL's pin).

Question: must ``step=`` be monotonically increasing? What happens with same-step writes,
out-of-order writes, the ``commit`` variants, and a ``define_metric`` custom step axis?

One W&B run per scenario, project ``wandb-step-semantics``. Values encode the step a metric
*belongs to* (train/loss == step, eval/acc == step), so on the dashboard:
  - a correctly landed metric is the y = x line,
  - a dropped point is simply missing,
  - a misattributed point sits off the line.

After all runs finish, the script reads every run back through the public API and prints, per
scenario: the rows that landed, which writes were dropped or misattributed, the summary values,
and how many "less than the current step" warnings wandb-core wrote to its local logs.

Usage (from ~/dev/skyrl, needs WANDB_API_KEY):
    SkyRL/.venv/bin/python -u skyrl-test/wandb_step_semantics.py [--dir DIR] [--only 08,11]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import wandb

ENTITY = "sky-posttraining-uc-berkeley"
PROJECT = "wandb-step-semantics"
N = 8  # training steps per scenario
LAG = 2  # the async eval of step i arrives after training step i + LAG
GS = "trainer/global_step"  # the custom step metric the step-3 plan declares
DROP_MSG = "less than the current step"  # wandb-core's warning text (grep'd from the binary)


@dataclass
class Write:
    key: str
    value: float
    x: int  # the step this value belongs to
    step: int | None  # the `step=` kwarg, None if omitted
    commit: bool | None  # the `commit=` kwarg, None if omitted
    extra: dict[str, Any]
    note: str = ""

    def call(self) -> str:
        kw = [f"step={self.step}"] if self.step is not None else []
        if self.commit is not None:
            kw.append(f"commit={self.commit}")
        payload = {self.key: self.value, **self.extra}
        return f"log({payload}" + (", " + ", ".join(kw) if kw else "") + ")"


class Logger:
    """Thin wrapper over run.log that records every write for the read-back comparison."""

    def __init__(self, run):
        self.run = run
        self.writes: list[Write] = []

    def __call__(self, key, value, *, x, step=None, commit=None, note="", **extra):
        self.run.log({key: value, **extra}, step=step, commit=commit)
        self.writes.append(Write(key, value, x, step, commit, extra, note))


SCENARIOS: list[tuple[str, str, Callable]] = []  # (name, group, fn)


def scenario(name: str, group: str):
    def deco(fn):
        SCENARIOS.append((name, group, fn))
        return fn

    return deco


# --------------------------------------------------------------------------- basics


@scenario("01-monotonic", "basics")
def s01(run, L):
    """Control. train/loss at step=i for i in 0..N-1, nothing else."""
    for i in range(N):
        L("train/loss", i, x=i, step=i)


@scenario("02-same-step-two-keys", "basics")
def s02(run, L):
    """Non-decreasing steps. train/loss then eval/acc, both at step=i, no commit kwarg.
    Do the two writes merge into one row?"""
    for i in range(N):
        L("train/loss", i, x=i, step=i)
        L("eval/acc", i, x=i, step=i)


@scenario("03-same-step-same-key-twice", "basics")
def s03(run, L):
    """Same key written twice at the same step: 100+i then 200+i. Which value wins?
    (a 2xx value on the dashboard means the last write won)"""
    for i in range(N):
        L("train/loss", 100 + i, x=i, step=i, note="first write")
        L("train/loss", 200 + i, x=i, step=i, note="second write")


@scenario("04-same-step-after-commit", "basics")
def s04(run, L):
    """train/loss at step=i with commit=True, then eval/acc at the SAME step=i.
    Does a step that was already committed still accept writes?"""
    for i in range(N):
        L("train/loss", i, x=i, step=i, commit=True)
        L("eval/acc", i, x=i, step=i, note="same step, after commit=True")


@scenario("05-step-plus-commit-false", "basics")
def s05(run, L):
    """SkyRL's Tracking.log default (step=i, commit=False) for every write and never an explicit
    commit=True. Do rows merge, and does finish() flush the last open row?"""
    for i in range(N):
        L("train/loss", i, x=i, step=i, commit=False)
        L("eval/acc", i, x=i, step=i, commit=False)


@scenario("06-gaps", "basics")
def s06(run, L):
    """Monotonic with gaps: step = 0, 10, 20, ... Do rows land only at those steps?"""
    for i in range(N):
        L("train/loss", 10 * i, x=10 * i, step=10 * i)


@scenario("07-backwards-before-any-commit", "basics")
def s07(run, L):
    """step=10, then step=5 while NOTHING has been committed yet, then step=10 again.
    Is the floor the largest step ever *seen*, or the last *committed* row?"""
    L("train/loss", 10, x=10, step=10)
    L("eval/acc", 5, x=5, step=5, note="below the open step, nothing committed yet")
    L("eval/acc", 10, x=10, step=10, note="equal to the open step")


# --------------------------------------------------------------------------- async-eval patterns


@scenario("08-async-eval-step-kwarg", "async-eval")
def s08(run, L):
    """The async-eval pattern with step=: train/loss at step=i; the eval of step i-LAG arrives after
    training step i and is logged with step=i-LAG. After training ends the remaining evals arrive
    in order."""
    for i in range(N):
        L("train/loss", i, x=i, step=i)
        if i >= LAG:
            L("eval/acc", i - LAG, x=i - LAG, step=i - LAG, note=f"arrives after train step {i}")
    for k in range(N - LAG, N):
        L("eval/acc", k, x=k, step=k, note="tail eval, training already ended")


@scenario("09-async-eval-no-step-kwarg", "async-eval")
def s09(run, L):
    """Async eval logged with NO step= while the trainer uses step=. The trainer writes two keys per
    step at different times (train/loss, later train/grad_norm) to expose what the eval's stepless
    write does to the trainer's open row."""
    for i in range(N):
        L("train/loss", i, x=i, step=i)
        if i >= LAG:
            L("eval/acc", i - LAG, x=i - LAG, note=f"no step=, arrives after train step {i}")
        L("train/grad_norm", i, x=i, step=i, note="second trainer write for the same step")


@scenario("10-async-eval-step-latest", "async-eval")
def s10(run, L):
    """Workaround candidate: the eval of step i-LAG is logged with step=i (the latest training step).
    Does it land, and at which x?"""
    for i in range(N):
        L("train/loss", i, x=i, step=i)
        if i >= LAG:
            L("eval/acc", i - LAG, x=i - LAG, step=i, note=f"eval of step {i - LAG} logged with step={i}")


# --------------------------------------------------------------------------- define_metric axis


def define_axis(run):
    run.define_metric(GS)
    run.define_metric("*", step_metric=GS)


@scenario("11-define-metric-no-step-kwarg", "define-metric")
def s11(run, L):
    """The step-3 plan: define_metric('*', step_metric='trainer/global_step'), never pass step=,
    every write carries trainer/global_step and commit=True. Same arrival order as scenario 08."""
    define_axis(run)
    for i in range(N):
        L("train/loss", i, x=i, commit=True, **{GS: i})
        if i >= LAG:
            L("eval/acc", i - LAG, x=i - LAG, commit=True, note=f"arrives after train step {i}", **{GS: i - LAG})
    for k in range(N - LAG, N):
        L("eval/acc", k, x=k, commit=True, note="tail eval, training already ended", **{GS: k})


@scenario("12-define-metric-with-step-kwarg", "define-metric")
def s12(run, L):
    """Same define_metric axis as 11, but step= is still passed (out of order for the evals).
    Does the custom axis rescue an out-of-order step= kwarg?"""
    define_axis(run)
    for i in range(N):
        L("train/loss", i, x=i, step=i, **{GS: i})
        if i >= LAG:
            L("eval/acc", i - LAG, x=i - LAG, step=i - LAG, note=f"arrives after train step {i}", **{GS: i - LAG})


@scenario("13-define-metric-hybrid-open-row", "define-metric")
def s13(run, L):
    """define_metric axis; the trainer keeps SkyRL's step=i, commit=False; the async eval writes a
    stepless row carrying trainer/global_step while the trainer's row for step i is still open; the
    trainer then writes train/grad_norm for the same step."""
    define_axis(run)
    for i in range(N):
        L("train/loss", i, x=i, step=i, commit=False, **{GS: i})
        if i >= LAG:
            L("eval/acc", i - LAG, x=i - LAG, note=f"no step=, arrives while train row {i} is open", **{GS: i - LAG})
        L("train/grad_norm", i, x=i, step=i, commit=False, note="second trainer write for the same step", **{GS: i})


@scenario("14-summary-out-of-order", "define-metric")
def s14(run, L):
    """Summary semantics under out-of-order evals (define_metric axis, no step=). Each key is logged
    at global steps 0..N-1 in order except that step 5 arrives LAST. eval/acc: default summary.
    eval/acc_summary_max: define_metric(summary='max'). eval/acc_override: default summary, then
    run.summary[...] = value-at-highest-step set by hand after the last log."""
    define_axis(run)
    run.define_metric("eval/acc_summary_max", step_metric=GS, summary="max")
    order = [i for i in range(N) if i != 5] + [5]
    for k in order:
        for key in ("eval/acc", "eval/acc_summary_max", "eval/acc_override"):
            L(key, k, x=k, commit=True, **{GS: k})
    run.summary["eval/acc_override"] = max(order)


@scenario("15-same-key-both-commit", "basics")
def s15(run, L):
    """Scenario 03 with commit=True on BOTH writes: train/loss = 100+i then 200+i, both at step=i,
    both commit=True. Which value wins now?"""
    for i in range(N):
        L("train/loss", 100 + i, x=i, step=i, commit=True, note="first write, commit=True")
        L("train/loss", 200 + i, x=i, step=i, commit=True, note="second write, commit=True")


@scenario("16-same-key-commit-first-only", "basics")
def s16(run, L):
    """Scenario 03 with commit=True on the FIRST write only."""
    for i in range(N):
        L("train/loss", 100 + i, x=i, step=i, commit=True, note="first write, commit=True")
        L("train/loss", 200 + i, x=i, step=i, note="second write, no commit kwarg")


@scenario("17-same-key-commit-second-only", "basics")
def s17(run, L):
    """Scenario 03 with commit=True on the SECOND write only."""
    for i in range(N):
        L("train/loss", 100 + i, x=i, step=i, note="first write, no commit kwarg")
        L("train/loss", 200 + i, x=i, step=i, commit=True, note="second write, commit=True")


# --------------------------------------------------------------------------- driver


def run_all(args) -> list[dict]:
    results = []
    for name, group, fn in SCENARIOS:
        if args.only and not any(name.startswith(p) for p in args.only):
            continue
        notes = " ".join(fn.__doc__.split())
        print(f"\n{'=' * 88}\n>>> {name}\n    {notes}\n{'=' * 88}", flush=True)
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=name,
            group=group,
            job_type="probe",
            notes=notes,
            tags=["step-semantics", group],
            config={"scenario": name, "group": group, "N": N, "LAG": LAG, "wandb": wandb.__version__},
            dir=args.dir,
        )
        L = Logger(run)
        fn(run, L)
        info = {"name": name, "id": run.id, "url": run.url, "sync_dir": run.settings.sync_dir, "writes": L.writes}
        run.finish()
        results.append(info)
        print(f"<<< {name} finished  {info['url']}", flush=True)
    return results


def fetch_rows(api, path: str, tries: int = 8):
    api_run, rows = None, []
    for _ in range(tries):
        api_run = api.run(path)
        rows = list(api_run.scan_history())
        if rows:
            break
        time.sleep(5)
    rows.sort(key=lambda r: r["_step"])
    return api_run, rows


def metric_keys(rows) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(k for k, v in row.items() if not k.startswith("_") and v is not None)
    return sorted(keys)


def verify(results: list[dict], args) -> None:
    api = wandb.Api()
    for res in results:
        api_run, rows = fetch_rows(api, f"{args.entity}/{args.project}/{res['id']}")
        keys = metric_keys(rows)
        print(f"\n{'=' * 88}\n{res['name']}   {res['url']}\n{'-' * 88}")
        print(f"rows landed: {len(rows)}   writes issued: {len(res['writes'])}")
        for row in rows:
            cells = "  ".join(f"{k}={row[k]}" for k in keys if row.get(k) is not None)
            print(f"  _step={row['_step']:<3} {cells}")
        print("writes:")
        dropped = 0
        for w in res["writes"]:
            hits = [r for r in rows if r.get(w.key) == w.value]
            if not hits:
                verdict = "DROPPED"
                dropped += 1
            else:
                r = hits[-1]
                gs = r.get(GS)
                where = f"_step={r['_step']}" + (f" {GS}={gs}" if gs is not None else "")
                x_landed = gs if gs is not None else r["_step"]
                verdict = f"landed at {where}"
                if x_landed != w.x:
                    verdict += f"   <-- MISATTRIBUTED (belongs to x={w.x})"
            tail = f"   [{w.note}]" if w.note else ""
            print(f"  {w.call():<62} x={w.x:<3} -> {verdict}{tail}")
        print(f"dropped: {dropped}/{len(res['writes'])}")
        summary = {k: v for k, v in dict(api_run.summary).items() if not k.startswith("_")}
        print(f"summary: {summary}")
        n, first = 0, None
        for path in glob.glob(os.path.join(res["sync_dir"], "logs", "*.log")):
            with open(path) as f:
                for line in f:
                    if DROP_MSG in line:
                        n += 1
                        first = first or f"{os.path.basename(path)}: {line.strip()[:220]}"
        print(f"wandb-core '{DROP_MSG}' warnings in local logs: {n}" + (f"\n  e.g. {first}" if first else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--dir", default=None, help="where wandb writes its local run dirs (default: ./wandb)")
    ap.add_argument("--only", default=None, help="comma-separated scenario prefixes, e.g. 08,11")
    ap.add_argument("--settle", type=float, default=10.0, help="seconds to wait before reading runs back")
    args = ap.parse_args()
    args.only = args.only.split(",") if args.only else None
    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY is not set")
    print(f"wandb {wandb.__version__}   entity={args.entity}   project={args.project}   N={N} LAG={LAG}", flush=True)
    results = run_all(args)
    print(f"\nall runs finished; waiting {args.settle:.0f}s for the backend to settle before reading back", flush=True)
    time.sleep(args.settle)
    verify(results, args)
    print(f"\nproject: https://wandb.ai/{args.entity}/{args.project}")


if __name__ == "__main__":
    main()
