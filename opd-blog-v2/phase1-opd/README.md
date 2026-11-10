# Phase 1: OPD works

Two runs, one claim: distilling a student from an RL-trained teacher of its own lineage, on the teacher's RL prompts,
recovers the teacher's gains, as Thinking Machines described. In both runs the student is the teacher's own starting
point, and both use the same data; only the models differ.

| Run | Student | Teacher | Thinking | Node |
|---|---|---|---|---|
| 1.1 `p1_opd_m2rl_sft_from_rl_math` | `Jackwang111/M2RL-SFT` (4B) | `Jackwang111/M2RL-RL_Math` | on | `h100-1` |
| 1.2 `p1_opd_qwen3_8b_from_grpo` | `Qwen/Qwen3-8B` | the GRPO run's export, see `../grpo-qwen3-8b/` | off | `b200-1`, after the GRPO run |

## Run 1.1

On your Mac, from `~/dev/skyrl`:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh h100-1
```

On the node:

```bash
cd ~/SkyRL
```

Smoke test, 2 tiny steps into `opd-blog-v2-smoke`. It also prepares the data, so the first time takes about 30
minutes. It ends with `=== smoke_p1_opd_m2rl_sft_from_rl_math passed`:

```bash
bash skyrl-test/opd-blog-v2/phase1-opd/run_m2rl.sh --smoke
```

The run:

```bash
nohup bash skyrl-test/opd-blog-v2/phase1-opd/run_m2rl.sh >> ~/opd-store/logs/p1_opd_m2rl_sft_from_rl_math.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/p1_opd_m2rl_sft_from_rl_math.log
```

## Run 1.2

Wait for the GRPO run on `b200-1` to finish (`status.sh` shows it `DONE`), and pick its teacher step as
`../grpo-qwen3-8b/README.md` says. On your Mac:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh b200-1
```

On the node:

```bash
cd ~/SkyRL
```

Smoke test, with your step in place of 250:

```bash
TEACHER_STEP=250 bash skyrl-test/opd-blog-v2/phase1-opd/run_qwen3_8b.sh --smoke
```

The run:

```bash
TEACHER_STEP=250 nohup bash skyrl-test/opd-blog-v2/phase1-opd/run_qwen3_8b.sh >> ~/opd-store/logs/p1_opd_qwen3_8b_from_grpo.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/p1_opd_qwen3_8b_from_grpo.log
```

After a preemption, run the same command again with the same `TEACHER_STEP`. It resumes from the last checkpoint.

## The recipe, shared by every OPD run in this blog

Phases 1, 2 and 3 share all of these, from `v2_opd_recipe` in `../common/run_lib.sh`. Most are M2RL's multi-teacher
OPD (paper appendix A and its `slime/scripts/multi_teacher_opd.sh`). The rest are SkyRL's OPD defaults.

| Setting | Value | Source |
|---|---|---|
| Data | `math_train.parquet`, 22,056 prompts (phase 3: `mopd_train.parquet`) | The teachers' RL prompts |
| Batch | 256 prompts × 4 samples, 1 optimizer step per training step | M2RL |
| Steps | 300 | M2RL |
| Learning rate | 1e-6 constant, no warmup | M2RL |
| Weight decay, grad clip | 0.1, 1.0 | M2RL, SkyRL |
| Teacher term | `advantages -= 1.0 × (log p_student - log p_teacher)` per token, pure distillation (no task reward) | Thinking Machines / SkyRL default |
| Loss | `importance_sampling`, token mean, no KL to a reference | SkyRL's OPD defaults |
| Sampling | temperature 1.0, top-p 1.0, untruncated | Required for an unbiased reverse KL |
| Answer length | 16,384 tokens with thinking, 8,192 without | M2RL's OPD used 16k |
| Eval | AIME 2024 and 2025, 16 samples, temperature 0.6, top-p 0.95, 32,768 tokens (8,192 without thinking), every 25 steps and before step 1 | M2RL's eval, at 16 samples instead of 64 |
| Checkpoints / exports | every 10 steps (last 2 kept) / every 50 | |
| Placement | student on 4 GPUs (FSDP and 4 colocated vLLM engines), teacher on the other 4 | The teacher needs GPUs of its own |
| Teacher engines | 4 × 1 GPU; context = prompt + answer + 1 | |

Between the two runs only the models, thinking, answer length, and the micro-batch differ: 1 per GPU for 1.1's 16k
answers, 2 for 1.2's 8k. `bash skyrl-test/opd-blog-v2/common/check_configs.sh` on your Mac prints both as SkyRL reads
them.

## What to watch

W&B project `opd-blog-v2`. Scores are accuracies from 0 to 1. `reward/avg_raw_reward` is the training accuracy; it is
logged but does not train.

| Metric | Healthy |
|---|---|
| `eval/all/avg_score`, `eval/aime-2024/avg_score`, `eval/aime-2025/avg_score` | Rise from the student's toward the teacher's |
| `opd/teacher/_/reverse_kl` | Falls: the student moves toward the teacher |
| `opd/teacher/_/log_ratio_p01` | The worst tokens' disagreement. Wildly negative means the teacher rejects what the student writes |
| `generate/avg_num_tokens` | Stays near the start; a sharp fall means answers are collapsing |
| `opd/truncation_rate`, `opd/repetition_rate` | Low and flat |

Reference points for 1.1, from the paper's Table 3 (AIME 2024 / 2025, average of 64 samples at 32k tokens). Ours use
16 samples, so expect them close, not equal:

| Model | AIME 2024 | AIME 2025 |
|---|---|---|
| M2RL-SFT, the student | 56.0 | 51.3 |
| M2RL-RL_Math, the teacher | 77.7 | 70.4 |
| M2RL's multi-teacher OPD, 5 teachers | 80.6 | 74.5 |

For 1.2, the reference is the GRPO run's own eval curve: same eval, same prompts.

Time: not measured. Our estimate is 2 to 4 days for 1.1, mostly generating 16k-token answers on 4 GPUs. The paper's
300-step multi-teacher run took 1,452 GPU-hours, which would be 7.5 days on one node, but its cluster and teacher
placement differ. 1.2's answers are about a fifth as long, so it should take under a day. `status.sh` shows the real
ETA after the first steps.

## Tradeoffs and what could go wrong

- **Prompts seen by the teacher:** both teachers were RL-trained on these exact prompts, which is the point: it is
  where they are strongest. It also means the student learns on problems the teacher has seen several times.
- **Teacher step for 1.2:** chosen by eval on AIME, which is also the test. Pick it once and report the student at
  every step, not just its best.
- **Throughput:** the student's 4 GPUs generate all 1,024 answers a step, while the teacher's 4 only score them, so
  the teacher side may sit partly idle. A 5 and 3 split is not possible: the rollouts must split evenly over the
  student's GPUs.
