"""Verify PR 2198's W&B layout on a finished smoke run, from the run's full history.

    uv run --isolated --extra fsdp skyrl-test/anyscale-pr2198/check_wandb_run.py \
        --project pr2198_smoke --run_name "$(cat /tmp/pr2198_last_gsm8k_run)" \
        --expect_key eval/all/pass_at_2

Checks:
  1. every row carries `global_step` (the adapter injects it; `_step` is only a call counter);
  2. eval rows (any `eval/*` key) sit exactly at --expect_eval_steps, each with `eval/all/avg_score`;
  3. `timing/eval_generate` and every --expect_key are on every eval row;
  4. eval keys never share a row with training keys (they left the per-step payload);
  5. the summary still exposes `eval/all/avg_score` (tests/train/gpu_e2e_test/get_summary.py compatibility);
  6. reports the W&B tables logged (the eval trajectory table goes through the new log_table path).
"""

import argparse
import sys

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--run_name", required=True)
ap.add_argument("--entity", default=None, help="defaults to the API key's entity, as get_summary.py does")
ap.add_argument("--expect_eval_steps", default="0,1,2,3")
ap.add_argument("--expect_key", action="append", default=[], help="a key every eval row must have; repeatable")
a = ap.parse_args()

api = wandb.Api()
path = f"{a.entity}/{a.project}" if a.entity else a.project
run = next(iter(api.runs(path, filters={"display_name": a.run_name}, order="-created_at")), None)
if run is None:
    sys.exit(f"run {a.run_name!r} not found in {path}")
rows = list(run.scan_history())
failures = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


check(all("global_step" in r for r in rows), f"all {len(rows)} rows carry global_step")
steps = [r["_step"] for r in rows if "_step" in r]
check(len(set(steps)) == len(steps), "_step is unique per row (a call counter, one row per log call)")

eval_rows = [r for r in rows if any(k.startswith("eval/") for k in r)]
expected = sorted(int(s) for s in a.expect_eval_steps.split(","))
got = sorted({int(r["global_step"]) for r in eval_rows if "global_step" in r})
check(got == expected, f"eval rows at global_step {got} (expected {expected})")
check(bool(eval_rows) and all("eval/all/avg_score" in r for r in eval_rows), "every eval row has eval/all/avg_score")
for key in ["timing/eval_generate", *a.expect_key]:
    check(bool(eval_rows) and all(key in r for r in eval_rows), f"every eval row has {key}")

TRAIN_PREFIXES = ("trainer/", "loss/", "policy/", "critic/", "generate/", "vllm/")
check(
    all(not [k for k in r if k.startswith(TRAIN_PREFIXES)] for r in eval_rows),
    "eval keys never share a row with training keys",
)
check("eval/all/avg_score" in run.summary, "summary exposes eval/all/avg_score (get_summary.py still works)")

tables = sorted({k for r in rows for k, v in r.items() if isinstance(v, dict) and v.get("_type") == "table-file"})
print(f"info  tables logged: {tables or 'none'}")
if eval_rows:
    print("info  keys on the first eval row: " + ", ".join(sorted(k for k in eval_rows[0] if not k.startswith("_"))))
    print("\nstep  avg_score" + "".join(f"  {k}" for k in a.expect_key))
    for r in sorted(eval_rows, key=lambda r: r["global_step"]):
        extras = "".join(f"  {r.get(k, float('nan')):.4f}" for k in a.expect_key)
        print(f"{int(r['global_step']):>4}  {r['eval/all/avg_score']:.4f}{extras}")

sys.exit(1 if failures else 0)
