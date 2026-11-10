# Section 3: MOPD, one student from three experts

One run, `mopd_3_m2rl_sft_from_math_code_if`, on `h100-5` in stage 2, then `eval_mopd_3_final` on every benchmark.
The student is M2RL-SFT, distilled from three M2RL experts at once, each row scored by the expert of its domain
(the row's `skyrl_mopd_domain` label). No teacher is trained here: the experts are M2RL's own, and `../0-evals`
checks them against their published numbers first.

| Teacher name | Model | Rows | Share | Engines |
|---|---|---|---|---|
| `math` | `Jackwang111/M2RL-RL_Math` | 22,056 | 38% | 2 × 1 GPU |
| `code` | `Jackwang111/M2RL-RL_Coding` | 19,169 | 33% | 1 × 1 GPU |
| `if` | `Jackwang111/M2RL-RL_IF` | 16,575 | 29% | 1 × 1 GPU |

The rows are `mopd_train.parquet` (`../data/README.md`). OPD needs no reward, so code and IF train with no sandbox
and no IF checker; their rows get the `opd_v2_noreward` environment.

## Settings

Script: [`run.sh`](run.sh), block `HYPERPARAMETERS`. M2RL's MT-OPD as published, with three of its five teachers;
the same block as section 2's.

| Setting | Value | Source |
|---|---|---|
| Student | `Jackwang111/M2RL-SFT` | [paper] |
| Teachers | RL_Math, RL_Coding, RL_IF, each row routed to its domain's | [paper] (five teachers: these three plus science and agent); [route] |
| Prompts | 57,800: math 22,056, code 19,169, IF 16,575 | [paper] section 3.2: the multi-task RLVR prompts; the proposal keeps three domains |
| Batch | 256 prompts × 4 samples, one update per step | [paper] appendix A; [script] |
| Steps | 300 | [paper]. [script] says `--num-rollout 400`; the paper's number is kept |
| Learning rate | 1e-6, constant, no warmup | [paper]; [script] `--lr 1e-6 --lr-decay-style constant` |
| Optimizer | Adam, betas 0.9 and 0.98, weight decay 0.1 | [script] |
| Grad clip | 1.0 | [slime] `--clip-grad` default |
| Rollouts | 16,384 tokens, prompts up to 2,048, temperature 1.0, top-p 1.0, no top-k, thinking on | [script]; [slime] defaults |
| Advantage | the routed teacher's log-prob minus the student's, per token | [loss] |
| Loss | importance sampling; each rollout's token mean, averaged over the batch | [slime] `--calculate-per-token-loss` off; [mean] |
| In-training eval | AIME 24 and 25, 8 samples, 32,768 tokens, temperature 0.6, top-p 0.95, every 5 steps | this kit's |

Infrastructure: 4 GPUs for the student, 4 for the teachers; 32 requests in flight per teacher engine;
`trainer.algorithm.opd.route_key=skyrl_mopd_domain`.

Every teacher's term enters the same advantage and the same loss. Nothing reweights the teachers: token
share balancing is not in the pinned commit.

## Targets

Each expert's own score in its domain (M2RL Table 3), from M2RL-SFT's:

| Benchmark | M2RL-SFT (start) | Expert (target) | Paper's MT-OPD, 5 teachers (reference only) |
|---|---|---|---|
| AIME 2024 / 2025 | 56.04 / 51.30 | RL_Math 77.66 / 70.42 | 80.57 / 74.48 |
| LiveCodeBench v5 / v6 | 55.92 / 52.00 | RL_Coding 65.00 / 58.86 | 63.31 / 58.86 |
| IFEval / IFBench | 79.48 / 38.44 | RL_IF 90.94 / 59.52 | 93.90 / 59.18 |

Success is recovering most of each expert's gain over M2RL-SFT in its own domain. The paper's MT-OPD row used five
teachers (with science and agent), so it is not matched exactly here.

## Run by hand

On the node, from `~/SkyRL`:

```bash
nohup bash skyrl-test/opd-blog-v2/common/job.sh mopd_3_m2rl_sft_from_math_code_if > /dev/null 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/mopd_3_m2rl_sft_from_math_code_if.log
```

Smoke test, 2 tiny steps (it passed on h100-2 on 2026-11-09, as phase 3 of the previous layout):

```bash
bash skyrl-test/opd-blog-v2/3-mopd/run.sh --smoke
```

## What to watch

| Metric | What it tells you |
|---|---|
| `opd/teacher/<name>/reverse_kl` | Falls as the student learns that domain. Code and IF have no in-training eval, so this is their progress signal |
| `opd/teacher/<name>/token_share` | The domain's share of the loss tokens, so of the gradient |
| `opd/teacher/<name>/rows` | Rows routed to it this step |
| `opd/teacher/<name>/time_per_row_mean` | Seconds from a row's rollout to its scores; a slow teacher shows here |
| `eval/aime-2024/avg_score`, `eval/aime-2025/avg_score` | Math |

`eval_mopd_3_final` scores the last export on all six benchmarks.

[paper]: https://arxiv.org/abs/2602.12566
[script]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L66-L127
[route]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/on_policy_distillation.py#L20-L25
[slime]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/utils/arguments.py#L730-L731
[loss]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/loss.py#L318-L337
[mean]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/cp_utils.py#L70-L76
