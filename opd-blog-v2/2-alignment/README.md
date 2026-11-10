# Section 2: OPD breaks when the student is not aligned with the teacher

One student, M2RL-SFT (Qwen3-4B-Base SFT'd on NVIDIA's Nemotron 3 Nano data), on M2RL's math prompts, with M2RL's own
MT-OPD recipe. Two teachers of about the same AIME score but different lineage:

- **2a, aligned:** M2RL-RL_Math, RL'd from this very student. AIME 2024/2025: 77.66/70.42 (M2RL Table 3).
- **2b, misaligned:** Qwen3-32B in thinking mode: stronger (81.4/72.9, Qwen3 report Table 13), but trained along a
  different line. Expected to break.

Then two fixes for 2b, with the same teacher:

- **2c, SFT warm-up:** M2RL-SFT first SFT'd on Qwen3-32B's correct answers to the same math prompts, then 2b's OPD.
- **2d, clipping:** 2b with each token's teacher signal clipped from below.

The student starts at 56.04/51.30 (M2RL Table 3); both teachers leave about 20 points to learn. The smoke tests of
2026-11-09 already point the expected way: at step 1 (512-token answers), M2RL-SFT's reverse KL to Qwen3-32B was 0.81,
against 0.11 to RL_Math. Qwen3-32B's chat template renders the student's prompts exactly as M2RL-SFT's does (checked
2026-11-09), so the teacher scores the same tokens the student wrote.

| Job | Stage | Node | What |
|---|---|---|---|
| `traces_2c_qwen3_32b` | 1 | b200-2 | 2c's warm-up data: Qwen3-32B's answers |
| `sft_2c_m2rl_sft_warmup` | 2 | b200-2 | 2c's warm-up SFT |
| `opd_2c_m2rl_warmup_from_qwen3_32b` | 2 | b200-2 | 2c |
| `opd_2b_m2rl_sft_from_qwen3_32b` | 2 | h100-1 | 2b |
| `opd_2d_m2rl_sft_from_qwen3_32b_clip` | 2 | h100-3 | 2d |
| `opd_2a_m2rl_sft_from_m2rl_rl_math` | 2 | h100-2 | 2a |
| `eval_opd_2*_final` | | | each run's last export at 64 samples, M2RL's settings |

Each OPD run waits for its teacher's and its student's evals (`../0-evals`): Qwen3-32B or RL_Math, and M2RL-SFT.

## The OPD recipe, the same for 2a to 2d

Script: [`opd.sh`](opd.sh), block `HYPERPARAMETERS`. M2RL's multi-teacher OPD with one teacher; section 3 uses the
same block with three.

| Setting | Value | Source |
|---|---|---|
| Student | `Jackwang111/M2RL-SFT`; for 2c, its warm-up output | the proposal |
| Teacher | 2a: `Jackwang111/M2RL-RL_Math`, 4 engines × 1 GPU. 2b to 2d: `Qwen/Qwen3-32B`, thinking, 2 engines × 2 GPUs on H100s (4 × 1 on B200s) | the proposal |
| Prompts | M2RL's math set, 22,056 (`../data/README.md`), up to 2,048 tokens | [paper] appendix A; [script] `--rollout-max-prompt-len 2048` |
| Batch | 256 prompts × 4 samples, one update per step | [paper]: batch size 256, group size 4, "gradient update is conducted per 1024 rollouts"; [script] |
| Steps | 400 | [script] `--num-rollout 400`. The paper's appendix A says 300; the code's number is used |
| Learning rate | 1e-6, constant, no warmup | [paper]; [script] `--lr 1e-6 --lr-decay-style constant` |
| Optimizer | Adam, betas 0.9 and 0.98, weight decay 0.1 | [script] `--adam-beta1 0.9 --adam-beta2 0.98 --weight-decay 0.1` |
| Grad clip | 1.0 | [slime] `--clip-grad` default, which the script leaves alone |
| Rollouts | 16,384 tokens, temperature 1.0, top-p 1.0, no top-k | [script] `--rollout-max-response-len 16384 --rollout-temperature 1.`; [slime] top-p and top-k defaults (#L225-L228) |
| Advantage | teacher log-prob minus student log-prob per token, no coefficient, no task reward | [loss]; [script] `--kl-loss-coef 0.00 --entropy-coef 0.00` |
| Loss | importance sampling; each rollout's token mean, averaged over the batch | [slime] `--calculate-per-token-loss` off; [mean]. slime's PPO loss has ratio 1 at one update a step, so its clip never acts: the same gradient |
| 2d's clip | `advantage_clip_min=-2.3` nats | this kit's (below) |
| In-training eval | AIME 2024 and 2025, 8 samples, 32,768 tokens, temperature 0.6, top-p 0.95, every 5 steps | this kit's, at M2RL's eval settings (`../0-evals`) |

Infrastructure: 4 GPUs for the student (training and generation colocated), 4 for the teacher; micro-batch 1;
checkpoints every 10 steps (the last 2 kept), exports every 50, and the last export goes to the Hub.

## 2c's warm-up

**The data** ([`traces.sh`](traces.sh)): Qwen3-32B answers every one of the 22,056 math prompts twice. Per prompt, the
first answer that is correct (the kit's math grader) and finished (it closed its thinking and stopped at `<|im_end|>`)
is kept, then tokenized with M2RL-SFT's tokenizer.

| Setting | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen3-32B`, thinking | the proposal |
| Prompts | all 22,056 of M2RL's math set, the ones 2c's OPD then trains on | the proposal ("Nemotron3 math"); the count is this kit's |
| Samples | 2 per prompt, first correct and finished one kept | this kit's: a second try keeps more of the hard prompts |
| Sampling | temperature 0.6, top-p 0.95, top-k 20 | [qwen3] section 4.6, thinking mode |
| Length | 16,384 tokens | this kit's: 2c's OPD rollout length, so the warm-up teaches answers that fit it |

**The SFT** ([`sft_2c.sh`](sft_2c.sh)). No paper gives a warm-up recipe. The closest published one is M2RL's own SFT of
M2RL-SFT on teacher answers, the multi-teacher off-policy distillation (MT-OFF) baseline, so this uses its settings:

| Setting | Value | Source |
|---|---|---|
| Base | `Jackwang111/M2RL-SFT` | |
| Data | the kept answers, one per prompt (about 17,000 expected, not measured) | above |
| Learning rate | 5e-5 | [paper] appendix A, MT-OFF |
| Batch | 512, 1 epoch (about 33 steps) | [paper] MT-OFF; the epoch count is ASSUMED |
| Warmup | 10% of the steps, then constant | [paper] appendix A, SFT ("10% training steps ... for learning rate warmup"); no decay is given, so constant is ASSUMED |
| Optimizer | Adam, betas 0.9 and 0.98 (ASSUMED: not given for SFT, taken from [script]), weight decay 0.1 | [paper] appendix A, SFT |
| Grad clip | 1.0 | [slime] default |
| Length | 18,432 tokens (a 2,048 prompt + a 16,384 answer) | this kit's |

A warm-up this strong may do most of the work itself. 2c's step-0 eval, against M2RL-SFT's, says how much: if the
warm-up alone closes most of the gap, the story is "SFT on the teacher" rather than "OPD after a warm-up". Lower the
learning rate or the rows in that case.

## 2d's clip

`trainer.algorithm.opd.advantage_clip_min=-2.3`: no token's `log q - log p` goes below -2.3 nats. That is about the
1st percentile of 2a's own pair (M2RL-SFT under RL_Math) in the smoke tests of 2026-11-09, where Qwen3-32B's was
-7.9. So the misaligned teacher's signal is kept to the aligned teacher's range. Change it with `CLIP_MIN` in
`opd.sh`. The pinned SkyRL also has `divergence_clip` and `advantage_log_compress` (`skyrl/train/opd/config.py`).

## Run by hand

The orchestrator runs these (`../README.md`). By hand on a node, from `~/SkyRL`:

```bash
nohup bash skyrl-test/opd-blog-v2/common/job.sh opd_2b_m2rl_sft_from_qwen3_32b > /dev/null 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/opd_2b_m2rl_sft_from_qwen3_32b.log
```

Smoke tests (tiny; nothing uploaded). The warm-up's smoke test needs the traces' smoke test first:

```bash
bash skyrl-test/opd-blog-v2/2-alignment/opd.sh 2b --smoke
```

```bash
bash skyrl-test/opd-blog-v2/2-alignment/traces.sh --smoke
```

```bash
bash skyrl-test/opd-blog-v2/2-alignment/sft_2c.sh --smoke
```

## What to watch

On W&B `opd-blog-v2`, for 2a to 2d side by side:

| Metric | Aligned (2a) | Broken (2b), as expected |
|---|---|---|
| `eval/aime-2024/avg_score` | rises toward the teacher | flat or falls |
| `opd/teacher/_/reverse_kl` | falls from about 0.1 | starts near 0.8, stays high or rises |
| `opd/teacher/_/log_ratio_p01` | about -2.3 | far below (about -7.9) |
| `opd/truncation_rate`, `opd/repetition_rate` | stable | rise: answers run to the limit or loop |

2c and 2d should look like 2a. If 2b does not break, that is a finding too.

## Risks

- **The teachers differ in size as well as lineage** (4B against 32B). A reader can ask whether size, not alignment,
  breaks 2b. The smoke tests' reverse KL is the direct evidence that it is alignment; 2c and 2d fixing 2b with the
  same 32B teacher is the stronger evidence.
- **Qwen3-32B is a slow teacher on H100s**: two engines of two GPUs each score every rollout. 2b and 2d may take
  noticeably longer per step than 2a; 2c runs on a B200.
- **2c's warm-up recipe is borrowed, not tuned** (above).

[paper]: https://arxiv.org/abs/2602.12566
[script]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L66-L127
[slime]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/utils/arguments.py#L730-L731
[loss]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/loss.py#L318-L337
[mean]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/cp_utils.py#L70-L76
[qwen3]: https://arxiv.org/abs/2505.09388
