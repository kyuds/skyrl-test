"""Summarize an OPD run from its W&B history and check it against the blog's targets.

    uv run --isolated --extra fsdp skyrl-test/opd-4xh100/check_opd_run.py \
        --project opd_4xh100 --run_name "$(cat $HOME/logs/last_opd_run)"

Prints, per training step: opd/reverse_kl, the teacher timing metrics next to timing/generate, and the
policy entropy; per eval step: eval/all/avg_score and pass@N. Then checks:
  1. the opd/* and timing/generate keys exist on every training row (the teacher path ran);
  2. opd/reverse_kl fell from its first to its last value (the student moves toward the teacher);
  3. the exposed teacher time is a small fraction of generation time (the overlap works);
  4. eval/all/avg_score improved from the step-0 eval to the last one.
Blog targets, for the eye (different models, so shapes not numbers): reverse KL falls fast and flattens
(near 0.09 for its 1.7B <- 4B pair, 0.01 for 4B <- 4B) while eval rises within ~20 steps. Those curves live in
the blog's W&B report (anyscale-llm-forge/aime_on_policy_distillation), which the lab key cannot read.
"""

import argparse
import sys

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--run_name", required=True)
ap.add_argument("--entity", default=None, help="defaults to the API key's entity")
ap.add_argument("--max_exposed_fraction", type=float, default=0.25, help="check 3 threshold, teacher_time_exposed / generate")
ap.add_argument("--grpo", action="store_true", help="a 04_run_grpo.sh baseline: no teacher, so skip the opd/* checks and print the reward metrics")
a = ap.parse_args()

api = wandb.Api()
path = f"{a.entity}/{a.project}" if a.entity else a.project
run = next(iter(api.runs(path, filters={"display_name": a.run_name}, order="-created_at")), None)
if run is None:
    sys.exit(f"run {a.run_name!r} not found in {path}")
rows = list(run.scan_history())
print(f"run {run.name} ({run.state}), {len(rows)} history rows, config train_batch_size={run.config.get('trainer', {}).get('train_batch_size')}")

failures = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


train_rows = [r for r in rows if r.get("opd/reverse_kl") is not None or r.get("timing/generate") is not None]
eval_rows = [r for r in rows if any(k.startswith("eval/") for k in r)]

if a.grpo:
    reward_keys = sorted({k for r in train_rows for k in r if k.startswith("reward/")})
    print("\nstep  " + "  ".join(k[len("reward/"):] for k in reward_keys) + "  generate_s  step_s  policy_entropy")
    for r in train_rows:
        vals = "  ".join(f"{r.get(k, float('nan')):.4f}" for k in reward_keys)
        print(f"{str(r.get('global_step', '?')):>4}  {vals}  {r.get('timing/generate', float('nan')):>10.1f}  "
              f"{r.get('timing/step', float('nan')):>6.0f}  {r.get('policy/policy_entropy', float('nan')):>14.4f}")
else:
  print("\nstep  reverse_kl  kl_abs_max  adv_opd_abs  teacher_exposed_s  teacher_group_mean_s  generate_s  step_s  policy_entropy")
  for r in train_rows:
      g = r.get("timing/generate")
      print(
          f"{str(r.get('global_step', '?')):>4}  "
          f"{r.get('opd/reverse_kl', float('nan')):>10.4f}  {r.get('opd/reverse_kl_abs_max', float('nan')):>10.3f}  "
          f"{r.get('opd/adv_opd_abs_mean', float('nan')):>11.4f}  {r.get('opd/teacher_time_exposed', float('nan')):>17.1f}  "
          f"{r.get('opd/teacher_time_per_group_mean', float('nan')):>20.2f}  {g if g is not None else float('nan'):>10.1f}  "
          f"{r.get('timing/step', float('nan')):>6.0f}  {r.get('policy/policy_entropy', float('nan')):>14.4f}"
      )

print("\neval rows (eval/all plus every per-dataset avg_score; the GSM8K test split is the only eval set):")
score_keys = sorted({k for r in eval_rows for k in r if k.startswith("eval/") and k.endswith("/avg_score")})
pass_keys = sorted({k for r in eval_rows for k in r if k.startswith("eval/all/pass_at_")})
for r in eval_rows:
    scores = "  ".join(f"{k[len('eval/'):-len('/avg_score')]}={r.get(k):.4f}" for k in score_keys if r.get(k) is not None)
    extras = "  ".join(f"{k.split('/')[-1]}={r.get(k):.3f}" for k in pass_keys if r.get(k) is not None)
    print(f"  step {str(r.get('global_step', '?')):>4}  {scores}  {extras}")

# 1. keys present
opd_keys = () if a.grpo else ("opd/reverse_kl", "opd/teacher_time_exposed", "opd/teacher_time_per_group_mean")
for key in opd_keys + ("timing/generate",):
    have = [r for r in train_rows if r.get(key) is not None]
    check(len(have) == len(train_rows) and train_rows, f"{key} on every training row ({len(have)}/{len(train_rows)})")

# 2. KL decreases (OPD only)
kls = [r["opd/reverse_kl"] for r in train_rows if r.get("opd/reverse_kl") is not None]
if not a.grpo:
    if len(kls) >= 2:
        check(kls[-1] < kls[0], f"opd/reverse_kl fell: first {kls[0]:.4f} -> last {kls[-1]:.4f} (min {min(kls):.4f})")
    else:
        check(False, "fewer than two reverse_kl values")

# 3. exposed teacher time small next to generation (OPD only)
fracs = [
    r["opd/teacher_time_exposed"] / r["timing/generate"]
    for r in train_rows
    if r.get("opd/teacher_time_exposed") is not None and r.get("timing/generate")
]
if fracs and not a.grpo:
    worst = max(fracs)
    check(worst <= a.max_exposed_fraction, f"teacher_time_exposed / generate <= {a.max_exposed_fraction} on every step (worst {worst:.3f})")

# 4. eval improves
scores = [(r.get("global_step"), r["eval/all/avg_score"]) for r in eval_rows if r.get("eval/all/avg_score") is not None]
if len(scores) >= 2:
    (s0, v0), (s1, v1) = scores[0], scores[-1]
    check(v1 > v0, f"eval/all/avg_score improved: step {s0} {v0:.4f} -> step {s1} {v1:.4f}")
else:
    print("skip  fewer than two eval rows; eval improvement not checked")

print("\nsummary keys:", sorted(k for k in run.summary.keys() if k.startswith(("eval/all/", "opd/", "reward/"))))
sys.exit(1 if failures else 0)
