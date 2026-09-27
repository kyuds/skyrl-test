"""Side-by-side eval curves for OPD and GRPO runs on the same student: the comparison Charlie asked for.

    uv run --isolated --extra fsdp skyrl-test/opd-4xh100/compare_runs.py --project opd_4xh100 \
        --run opd=opd_base_0p8b_from_2b_2026... --run grpo=grpo_base_0p8b_2026...

Prints, per eval step, every eval set's avg_score for each run, plus the rollouts consumed so far
(step x prompts x samples, from the run config), so the two can be compared per step and per sample.
Also prints each run's mean timing/step and, for OPD, the mean exposed teacher time, for the wall-clock view.
"""

import argparse
import sys
from collections import defaultdict

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--entity", default=None)
ap.add_argument("--run", action="append", required=True, help="label=run_name; repeatable")
a = ap.parse_args()

api = wandb.Api()
path = f"{a.entity}/{a.project}" if a.entity else a.project
runs = {}
for spec in a.run:
    label, _, name = spec.partition("=")
    if not name:
        sys.exit(f"--run needs label=run_name, got {spec!r}")
    run = next(iter(api.runs(path, filters={"display_name": name}, order="-created_at")), None)
    if run is None:
        sys.exit(f"run {name!r} not found in {path}")
    runs[label] = run

table = defaultdict(dict)  # step -> {(label, key): value}
score_keys = set()
per_step = {}
for label, run in runs.items():
    rows = list(run.scan_history())
    trainer_cfg = run.config.get("trainer", {}); gen_cfg = run.config.get("generator", {})
    rollouts_per_step = trainer_cfg.get("train_batch_size", 0) * gen_cfg.get("n_samples_per_prompt", 0)
    step_times = [r["timing/step"] for r in rows if r.get("timing/step") is not None]
    exposed = [r["opd/teacher_time_exposed"] for r in rows if r.get("opd/teacher_time_exposed") is not None]
    per_step[label] = (rollouts_per_step, sum(step_times) / len(step_times) if step_times else float("nan"),
                       sum(exposed) / len(exposed) if exposed else None)
    for r in rows:
        keys = [k for k in r if k.startswith("eval/") and k.endswith("/avg_score")]
        if not keys:
            continue
        step = r.get("global_step")
        for k in keys:
            short = k[len("eval/"):-len("/avg_score")]
            score_keys.add(short)
            table[step][(label, short)] = r[k]

labels = list(runs)
keys = sorted(score_keys)
print("run        rollouts/step  mean step_s  mean teacher_exposed_s")
for label, (rps, st, ex) in per_step.items():
    print(f"{label:<10} {rps:>13}  {st:>11.0f}  {'-' if ex is None else f'{ex:.1f}':>22}")

header = "step  " + "  ".join(f"{label}:{k}" for label in labels for k in keys) + "  " + "  ".join(f"{label}:rollouts" for label in labels)
print("\n" + header)
for step in sorted(s for s in table if s is not None):
    cells = []
    for label in labels:
        for k in keys:
            v = table[step].get((label, k))
            cells.append(f"{v:.4f}" if v is not None else "   -  ")
    consumed = "  ".join(f"{step * per_step[label][0]:>12}" for label in labels)
    print(f"{step:>4}  " + "  ".join(f"{c:>{len(label) + len(k) + 1}}" for c, (label, k) in zip(cells, [(l, k) for l in labels for k in keys])) + "  " + consumed)
