# GRPO teacher: Qwen3-8B, thinking off, on M2RL's math set

Trains the teacher for phase 1.2. It is Qwen3-8B plus GRPO on the same math prompts M2RL-RL_Math was trained on, so
phase 1's two experiments share their data and differ only in their models. Phase 1.2 then distills it back into
Qwen3-8B.

Node: `b200-1`, all 8 GPUs. Phase 1.2 runs on the same node afterwards, because its teacher is one of this run's
exports.

## Run it

On your Mac, from `~/dev/skyrl`, log into the node:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh b200-1
```

On the node, everything from `~/SkyRL`:

```bash
cd ~/SkyRL
```

A smoke test first: 2 tiny steps, about 15 minutes, into the W&B project `opd-blog-v2-smoke`. It also prepares the
data on the node. It ends with `=== smoke_grpo_qwen3_8b_nothink passed`:

```bash
bash skyrl-test/opd-blog-v2/grpo-qwen3-8b/run.sh --smoke
```

The run:

```bash
nohup bash skyrl-test/opd-blog-v2/grpo-qwen3-8b/run.sh >> ~/opd-store/logs/grpo_qwen3_8b_nothink.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/grpo_qwen3_8b_nothink.log
```

The ssh session can close; the run keeps going. Preempted or stopped? Bring the node back as `gcp/README.md` says,
then run the same `nohup` command. It resumes from the last checkpoint.

## Settings

| Setting | Value | Why |
|---|---|---|
| Model | `Qwen/Qwen3-8B`, `enable_thinking: false` | Thinking on starts at 76.0 / 67.3 on AIME 2024 / 2025 (Qwen3 report), about where M2RL-RL_Math ends, and would cost about 25 times more. Thinking off starts at 29.1 / 20.9 and leaves OPD a gap to close |
| Data | `math_train.parquet`, 22,056 prompts | M2RL-RL_Math's own RL prompts, in M2RL's format |
| Reward | 1 or 0 from `opd_v2_math` (math-verify), no length penalty | M2RL's. SkyRL's `aime` env would mark correct fractions wrong. Without the length penalty, the mechanism behind the first launch's Base collapse is gone |
| Batch | 128 prompts × 16 samples | M2RL's math RL |
| Optimizer steps per training step | 1 (mini-batch = batch) | M2RL's. The 2026-11 teachers took 8 per step and became unstable after their peaks |
| Learning rate | 2e-6, constant after 20 warmup steps | M2RL's, for its 4B model; untested at 8B. If entropy climbs, restart with `trainer.policy.optimizer_config.lr=1e-6` |
| Loss | PPO clip 0.2 / 0.28, token mean, no KL | M2RL's clip-higher; its KL coefficient was 0 |
| Train/inference mismatch | token-level TIS, cap 2.0 | M2RL ran slime with `--use-tis` |
| Answer length | 8,192 tokens, temperature 1.0 | Thinking-off answers ran 2,000 to 3,000 tokens in the 2026-11 runs |
| Steps | 400, about 2.3 passes over the data | M2RL's |
| Eval | AIME 2024 and 2025, 16 samples each, temperature 0.6, top-p 0.95, every 25 steps and before step 1 | Temperature and top-p as M2RL evaluated |
| Checkpoints / exports | every 10 steps (last 2 kept) / every 50 steps | An export is at most 25 steps from any eval |
| Placement | 8 GPUs, colocated FSDP and 8 vLLM engines, eager mode | The 2026-11 8B runs' settings, which held for 600 steps on a B200 |

Every value is in `run.sh`. On your Mac, `bash skyrl-test/opd-blog-v2/common/check_configs.sh` prints how SkyRL reads
them.

## What to watch

W&B project `opd-blog-v2`, run `grpo_qwen3_8b_nothink`. Rewards and scores are accuracies from 0 to 1 here, unlike
the 2026-11 runs, which scored -1 to 1.

| Metric | Healthy |
|---|---|
| `eval/all/avg_score` | Rises from step 0's value, then levels off |
| `reward/avg_raw_reward` | Rises from its first value. On DAPO-17k, the 2026-11 Qwen3-8B run started near 0.5 accuracy |
| `generate/avg_num_tokens` | Stays in the low thousands. A fall below a few hundred means answers are collapsing |
| `policy/policy_entropy` | Roughly flat. A steady climb was the warning sign before the 2026-11 4B collapse |
| `policy/loss_metrics/clip_ratio` | Small and flat. It rose about tenfold just before that collapse |

`bash skyrl-test/opd-blog-v2/status.sh` shows the step and ETA. Time: about 2 to 3 days, an estimate. The 2026-11 8B
run took 255 seconds a step on a B200 node with half the samples per prompt.

## Pick the teacher step

The 2026-11 teachers peaked between steps 225 and 550 and then declined, so the last step is not necessarily the best.
When the run is done, take the export nearest its best `eval/all/avg_score`; exports are at every multiple of 50.
Phase 1.2 takes it as `TEACHER_STEP`, and the exports stay on the node's disk.

To keep it on the Hub too, on the node:

```bash
bash skyrl-test/opd-blog-v2/common/upload_export.sh --run grpo_qwen3_8b_nothink --step 250
```

That uploads to `kyuds/opd-blog-v2/grpo_qwen3_8b_nothink_step250`, public. Replace 250 with your step.

## Tradeoffs and what could go wrong

- **Learning rate:** 2e-6 is M2RL's value for a 4B model with thinking on. At 8B it may move too fast; the entropy and
  clip ratio above show it within the first 100 steps.
- **Not M2RL's exact recipe:** thinking off and 8k answers where M2RL used thinking and 32k. The data and update shape
  match; the mode does not.
- **Grader limits:** `0.2` against a `20\%` answer scores 0 (see `data/README.md`). A pathological expression could stall
  math-verify, because its timeouts cannot run off the main thread; a stalled step would show as `STALLED` in
  `status.sh`.
- **Memory:** 128 × 16 samples is twice the 2026-11 runs' per step. The smoke test does not reach full batch size, so
  an out-of-memory error would show in the first real steps; lower `trainer.micro_train_batch_size_per_gpu` to 2.
