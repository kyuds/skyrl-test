# Phase 2: a better teacher fails to teach

One run, `p2_opd_m2rl_sft_from_qwen3_32b`, on `b200-2`. It is phase 1.1 with one change: the teacher is
`Qwen/Qwen3-32B` in thinking mode instead of M2RL-RL_Math. Same student, same prompts, same recipe, same steps. The
expected result is a crash: a teacher that is better at everything still fails to teach a student whose distribution is
far from its own. The fixes, SFT warm-up and clipping, are not run here; this phase only shows the crash.

## Why this teacher

Qwen3-32B is 8 times the student's size and scores higher than phase 1.1's teacher on every benchmark both report:

| Benchmark | M2RL-RL_Math (phase 1.1) | Qwen3-32B, thinking |
|---|---|---|
| AIME 2024 | 77.7 | 81.4 |
| AIME 2025 | 70.4 | 72.9 |
| GPQA-Diamond | 58.5 | 68.4 |
| LiveCodeBench v5 | 59.6 | 65.7 |
| IFEval, strict prompt | 80.6 | 85.0 |
| BFCL v3 | 50.3 | 70.3 |

M2RL-RL_Math's numbers are from the M2RL paper, Table 3; Qwen3-32B's from the Qwen3 technical report, Table 13. The
harnesses differ, but the M2RL paper quotes the same report for its own Qwen baselines.

The gap on math is modest, and that is the point: the claim is not that a far stronger teacher breaks OPD, but that a
better one is no guarantee. Its post-training differs from M2RL-SFT's (Qwen's own pipeline against NVIDIA's Nemotron SFT
data), which is the misalignment this phase is about.

It also needs nothing changed. OPD teachers score the student's exact token ids, and Qwen3-32B's chat template is
M2RL's, byte for byte, so the student is M2RL-SFT as published, the same as in 1.1. Qwen3-30B-A3B-Thinking-2507 was
considered first, 85.0 on AIME 2025, but its template puts `<think>` in the prompt where M2RL's has the model write it,
so a crash could have been blamed on the template.

## Run it

On your Mac, from `~/dev/skyrl`:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh b200-2
```

On the node:

```bash
cd ~/SkyRL
```

Smoke test, 2 tiny steps into `opd-blog-v2-smoke`. The first time it also prepares the data and downloads the teacher
(66 GB), so allow about 40 minutes. It ends with `=== smoke_p2_opd_m2rl_sft_from_qwen3_32b passed`:

```bash
bash skyrl-test/opd-blog-v2/phase2-opd/run.sh --smoke
```

The run:

```bash
nohup bash skyrl-test/opd-blog-v2/phase2-opd/run.sh >> ~/opd-store/logs/p2_opd_m2rl_sft_from_qwen3_32b.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/p2_opd_m2rl_sft_from_qwen3_32b.log
```

On a B200, the 32B model (66 GB in bf16) and its scoring fit on one GPU, so the default is 4 engines of 1 GPU. On an
H100 node instead, one GPU leaves too little room to score 18k-token sequences: put `TEACHER_ENGINES=2 TEACHER_TP=2`
in front of both commands.

## Settings

Everything is phase 1.1's (see `../phase1-opd/README.md`), except:

| Setting | Phase 1.1 | Phase 2 |
|---|---|---|
| Teacher | `Jackwang111/M2RL-RL_Math` | `Qwen/Qwen3-32B`, thinking |
| Teacher engines | 4 × 1 GPU | 4 × 1 GPU on a B200; 2 × 2 GPUs on an H100 |

## What a crash looks like

Put this run next to 1.1 in W&B, on the same axes:

| Metric | Phase 1.1, healthy | Phase 2, if it breaks |
|---|---|---|
| `eval/all/avg_score` | Rises toward the teacher | Falls below the student's step-0 score |
| `opd/teacher/_/reverse_kl` at step 1 | The starting distance | Much larger: the teacher disagrees with much of what the student writes |
| `opd/teacher/_/log_ratio_p01` | Moderate | Far more negative: tokens the teacher all but rules out |
| `generate/avg_num_tokens` | Near its start | Drifts: answers balloon to the 16k limit, or shrink |
| `opd/truncation_rate`, `opd/repetition_rate` | Low | Rise |

The step-0 eval is the same student in both runs, so it is the shared baseline. A crash is expected, not certain. If it
does not break, that is a finding too, and the next step is a teacher further from the student.

## Fixes, for later

Not run here. For a follow-up, the pinned SkyRL already has the clipping knobs, set on the same command line:

- `trainer.algorithm.opd.divergence_clip=<nats>`: drops the teacher signal on tokens whose reverse KL is above the
  ceiling.
- `trainer.algorithm.opd.advantage_clip_min=<-nats>` and `advantage_clip_max=<nats>`: bound each token's
  `log p_teacher - log p_student`.
- `trainer.algorithm.opd.advantage_log_compress=true`: compresses it to `sign(a) log(1 + |a|)` instead.

SFT warm-up, a short SFT on the teacher's own answers before OPD, has no script in this kit yet.

## Tradeoffs and what could go wrong

- **More than quality differs:** Qwen3-32B differs from M2RL-RL_Math in size, base model (Qwen's 32B base, not
  Qwen3-4B-Base) and post-training. A crash shows that this teacher fails to teach, not which difference causes it.
- **Modest math gap:** 2.5 to 3.7 points on AIME. If the blog wants "much stronger", this teacher does not show it;
  "better on every benchmark" it does.
- **Teacher's own length:** Qwen3-32B in thinking mode may reason longer than 16k tokens on hard problems, while the
  student is cut at 16k. The teacher only scores the student's answers, so this does not change what it scores.
- **Eval is AIME only:** enough for a crash, which shows on math first.
