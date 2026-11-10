# Phase 3: MOPD, one student from three experts

One run, `p3_mopd_m2rl_sft_from_math_code_if`, on `h100-2`. The student is phase 1.1's M2RL-SFT, distilled from three
M2RL experts at once. Each row is scored by the expert of its domain, picked by the row's `skyrl_mopd_domain` label.
This is how most people combine experts in practice, and the paper's own multi-teacher OPD beat each expert in its
own domain on math and instruction following.

| Teacher name | Model | Rows | Share | Engines |
|---|---|---|---|---|
| `math` | `Jackwang111/M2RL-RL_Math` | 22,056 | 38% | 2 × 1 GPU |
| `code` | `Jackwang111/M2RL-RL_Coding` | 19,169 | 33% | 1 × 1 GPU |
| `if` | `Jackwang111/M2RL-RL_IF` | 16,575 | 29% | 1 × 1 GPU |

The rows are `mopd_train.parquet`: the math set of phases 1 and 2, plus the blend's competitive-coding and
instruction-following prompts (`../data/README.md`). The paper used five experts, with science and agent too; this run
uses three.

OPD needs no reward, so code and IF train with no sandbox and no IF checker. Their rows get the `opd_v2_noreward`
environment.

## Run it

On your Mac, from `~/dev/skyrl`:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh h100-2
```

On the node:

```bash
cd ~/SkyRL
```

Smoke test, 2 tiny steps into `opd-blog-v2-smoke`; the first time it also prepares the data. It ends
with `=== smoke_p3_mopd_m2rl_sft_from_math_code_if passed`:

```bash
bash skyrl-test/opd-blog-v2/phase3-mopd/run.sh --smoke
```

The run:

```bash
nohup bash skyrl-test/opd-blog-v2/phase3-mopd/run.sh >> ~/opd-store/logs/p3_mopd_m2rl_sft_from_math_code_if.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/p3_mopd_m2rl_sft_from_math_code_if.log
```

After a preemption, run the same command again. It resumes from the last checkpoint.

## Settings

Everything is phase 1.1's recipe (`../phase1-opd/README.md`), so the math rows train exactly as in 1.1, except:

| Setting | Value |
|---|---|
| Data | `mopd_train.parquet`, 57,800 prompts, shuffled each pass |
| Teachers | `trainer.teachers.{math,code,if}` as above, each launched by the job on its own GPU(s) |
| Routing | `trainer.algorithm.opd.route_key=skyrl_mopd_domain`, the pinned SkyRL's default |
| Teacher concurrency | 32 requests in flight per engine |

Every teacher's term enters the same advantage and the same token-mean loss. Nothing reweights the teachers yet:
token-share balancing is the next PR of the OPD stack, not in the pinned commit.

## What to watch

Each teacher logs its own metrics under `opd/teacher/<name>/`:

| Metric | What it tells you |
|---|---|
| `opd/teacher/<name>/reverse_kl` | Falls as the student learns that domain. Code and IF have no in-training eval, so this is their progress signal |
| `opd/teacher/<name>/token_share` | The domain's share of the loss tokens, so of the gradient. Long answers take more |
| `opd/teacher/<name>/prompt_share` | The domain's share of the rows, about 0.38, 0.33 and 0.29 |
| `opd/teacher/<name>/rows` | Rows routed to it this step |
| `opd/teacher/<name>/time_per_row_mean` | Seconds from a row's rollout to its scores. A slow teacher shows here |
| `eval/all/avg_score`, `eval/aime-2024/avg_score`, `eval/aime-2025/avg_score` | Math, as in phases 1 and 2 |

`reward/avg_raw_reward` averages over every row, and code and IF rows score 0, so here it is about 0.38 times the math
accuracy.

## What is not evaluated yet

Only math is evaluated during the run. Code and instruction following need evaluating on the exports after the run,
and that is not set up:

- **LiveCodeBench v5 and v6:** SkyRL's `lcb` environment runs tests locally, but its data prep downloads an
  unversioned file from Google Drive, not v5 or v6.
- **IFEval and IFBench:** SkyRL has no checker for them.

Exports are written every 50 steps and stay on the node's disk, so the evals can be added after the run starts without
retraining. They are needed before this phase's results go in the blog.

Reference points, the paper's Table 4, for five teachers (average of 64 samples on AIME, scores in percent):

| | AIME 2024 | AIME 2025 | LCB v5 | LCB v6 | IFEval | IFBench |
|---|---|---|---|---|---|---|
| M2RL-SFT, the student | 56.0 | 51.3 | 55.9 | 52.0 | 79.5 | 38.4 |
| Best single expert | 77.7 (math) | 70.4 (math) | 65.0 (code) | 58.9 (code) | 90.9 (IF) | 59.5 (IF) |
| Multi-teacher OPD, 5 teachers | 80.6 | 74.5 | 63.3 | 58.9 | 93.9 | 59.2 |

## Tradeoffs and what could go wrong

- **Unbalanced gradient:** under a token-mean loss, domains with longer answers pull harder. Code and math answers
  are longer than IF's, so IF may learn least. `token_share` shows it; the next PR's balancing is the fix.
- **Not the paper's run:** three teachers instead of five, and 300 steps over 57,800 prompts is 1.3 passes, where the
  paper's 300 steps over all five domains were under one.
- **Teacher GPUs:** math gets two engines for its larger share. If one teacher's `time_per_row_mean` is far above the
  others, it is the bottleneck; the split can only change in whole GPUs, and the student keeps 4.
- **No code or IF eval yet:** see above.
