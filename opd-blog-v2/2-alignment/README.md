# Section 2: OPD breaks when the student is not aligned with the teacher

One student, M2RL-SFT (Qwen3-4B-Base SFT'd on NVIDIA's Nemotron 3 Nano data), on M2RL's math prompts, with M2RL's own
MT-OPD recipe. Two teachers of about the same AIME score but different lineage (AIME 2024/2025, measured on
2026-11-10, `../0-evals/README.md`):

- **2a, aligned:** M2RL-RL_Math, RL'd from this very student: 82.29/75.83 (M2RL Table 3 says 77.66/70.42).
- **2b, misaligned:** Qwen3-32B in thinking mode, trained along a different line: 80.68/72.24 at Qwen's settings
  (Qwen3 report Table 13: 81.4/72.9). Expected to break.

Then fixes for 2b, with the same teacher:

- **2c, SFT warm-up:** M2RL-SFT first SFT'd on Qwen3-32B's correct answers to the same math prompts, then 2b's OPD.
- **2d, clipping:** 2b with each token's teacher signal clipped from below. Stopped at step 56: it collapsed as 2b
  did ("2d's clip").
- **2e, light warm-up:** 2c with its warm-up at a fifth of the learning rate, 1e-5 instead of 5e-5 ("2e's light
  warm-up").

The student starts at 56.51/43.54 measured (M2RL Table 3: 56.04/51.30); both teachers leave 24 to 32 points to
learn. Its AIME 2025 miss is its own behaviour: 21% of its answers there are a bare Python tool call, which scores
0 (`../0-evals/README.md`). The smoke tests of
2026-11-09 already point the expected way: at step 1 (512-token answers), M2RL-SFT's reverse KL to Qwen3-32B was 0.81,
against 0.11 to RL_Math. Qwen3-32B's chat template renders the student's prompts exactly as M2RL-SFT's does (checked
2026-11-09), so the teacher scores the same tokens the student wrote.

| Job | Stage | Node | What |
|---|---|---|---|
| `traces_2c_qwen3_32b` | 1 | b200-2 | 2c's warm-up data: Qwen3-32B's answers |
| `sft_2c_m2rl_sft_warmup` | 2 | b200-2 | 2c's warm-up SFT |
| `opd_2c_m2rl_warmup_from_qwen3_32b` | 2 | b200-2 | 2c |
| `opd_2b_m2rl_sft_from_qwen3_32b` | 2 | b200-2 | 2b, after 2c |
| `opd_2d_m2rl_sft_from_qwen3_32b_clip` | 2 | h100-2 | 2d, stopped at step 56 on 2026-11-23; no longer in `config.sh` |
| `sft_2e_m2rl_sft_light_warmup` | 2 | h100-2 | 2e's light warm-up SFT, on 2c's warm-up data (copied from b200-2) |
| `opd_2e_m2rl_light_warmup_from_qwen3_32b` | 2 | h100-2 | 2e |
| `opd_2a_m2rl_sft_from_m2rl_rl_math` | 2 | h100-4 | 2a |
| `eval_opd_2*_final` | | | each run's last export at 64 samples, M2RL's settings |

Each OPD run waits for its teacher's and its student's evals (`../0-evals`): Qwen3-32B or RL_Math, and M2RL-SFT.

## The OPD recipe, the same for 2a to 2e

Script: [`opd.sh`](opd.sh), block `HYPERPARAMETERS`. M2RL's multi-teacher OPD with one teacher; section 3 uses the
same block with three.

| Setting | Value | Source |
|---|---|---|
| Student | `Jackwang111/M2RL-SFT`; for 2c and 2e, the output of its warm-up | the proposal |
| Teacher | 2a: `Jackwang111/M2RL-RL_Math`, 4 engines × 1 GPU. 2b to 2e: `Qwen/Qwen3-32B`, thinking, 2 engines × 2 GPUs on H100s (4 × 1 on B200s) | the proposal |
| Prompts | M2RL's math set, 22,056 (`../data/README.md`), up to 2,048 tokens | [paper] appendix A; [script] `--rollout-max-prompt-len 2048` |
| Batch | 256 prompts × 4 samples, one update per step | [paper]: batch size 256, group size 4, "gradient update is conducted per 1024 rollouts"; [script] |
| Steps | 250 | fewer than [script] `--num-rollout 400` and the paper's appendix A (300), to save GPU time: 2c's in-training AIME was flat from about step 110. 2c was launched with a limit of 400 and resumed with 250 at step 210; the learning rate is constant, so it trains as a 250-step run would |
| Learning rate | 1e-6, constant, no warmup | [paper]; [script] `--lr 1e-6 --lr-decay-style constant` |
| Optimizer | Adam, betas 0.9 and 0.98, weight decay 0.1 | [script] `--adam-beta1 0.9 --adam-beta2 0.98 --weight-decay 0.1` |
| Grad clip | 1.0 | [slime] `--clip-grad` default, which the script leaves alone |
| Rollouts | 16,384 tokens, temperature 1.0, top-p 1.0, no top-k | [script] `--rollout-max-response-len 16384 --rollout-temperature 1.`; [slime] top-p and top-k defaults (#L225-L228) |
| Advantage | teacher log-prob minus student log-prob per token, no coefficient, no task reward | [loss]; [script] `--kl-loss-coef 0.00 --entropy-coef 0.00` |
| Loss | importance sampling; each rollout's token mean, averaged over the batch | [slime] `--calculate-per-token-loss` off; [mean]. slime's PPO loss has ratio 1 at one update a step, so its clip never acts: the same gradient |
| 2d's clip | `advantage_clip_min=-2.3` nats | this kit's (below) |
| In-training eval | AIME 2024 and 2025, 8 samples, 32,768 tokens, temperature 0.6, top-p 0.95, every 5 steps | this kit's, at M2RL's eval settings (`../0-evals`) |

Infrastructure: one node per run, 4 GPUs for the student (training and generation colocated) and 4 for its teacher:
RL_Math on 4 engines of 1 GPU for 2a (h100-4); Qwen3-32B on 2 engines of 2 GPUs for 2e (h100-2, after 2d), and on 4
engines of 1 GPU on b200-2 for 2c and then 2b. That puts 2a, 2b and 2e on the three nodes left after MOPD at once; with
served teachers, the two H100 nodes would fit one run at a time. Micro-batch 1; checkpoints every 5 steps (the last 2 kept),
exports every 50, and the last export goes to the Hub.

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
| Data | the kept answers, one per prompt: 12,535 of the 22,056 prompts, 103M tokens, 8,252 per answer on average (measured 2026-11-11; Qwen3-32B was right on 50.2% of its 44,112 answers, and 25.4% hit the 16,384-token limit) | above |
| Learning rate | 2c: 5e-5. 2e: 1e-5 | 2c: [paper] appendix A, MT-OFF. 2e: this kit's ("2e's light warm-up") |
| Batch | 512, 1 epoch (25 steps) | [paper] MT-OFF; the epoch count is ASSUMED |
| Warmup | 10% of the steps, then constant | [paper] appendix A, SFT ("10% training steps ... for learning rate warmup"); no decay is given, so constant is ASSUMED |
| Optimizer | Adam, betas 0.9 and 0.98 (ASSUMED: not given for SFT, taken from [script]), weight decay 0.1 | [paper] appendix A, SFT |
| Grad clip | 1.0 | [slime] default |
| Length | 18,432 tokens (a 2,048 prompt + a 16,384 answer) | this kit's |

## 2e's light warm-up

2c's warm-up at 5e-5 aligned its student but damaged it. At OPD step 1 its reverse KL to Qwen3-32B was 0.19, against
2b's 0.54 without a warm-up, and M2RL-SFT's bare tool calls were gone. But at step 0 (the in-training eval: 480
answers, temperature 0.6, 32,768 tokens) 45% were right against M2RL-SFT's 52%, 34% ran to the token limit and 26%
ended in a loop (the last 8,000 characters compress to under a tenth), against 19% and 3%. The kept answers themselves
do not loop: none of the 12,535 ends in one. 2c's OPD then spent its first 75 steps repairing this (loops down to 5%,
60% right) and stayed near 60% to step 250 (measured 2026-11-22).

M2RL's 5e-5 is for answers from teachers RL'd from this very student, which are close to what it already writes;
Qwen3-32B's are not, so the same rate moves the student much further. As a rough measure of how far each moved it,
learning rate × steps is 25 × 5e-5 ≈ 1.3e-3 for the warm-up against 250 × 1e-6 = 2.5e-4 for all of 2c's OPD. 2e's
1e-5 brings the warm-up to about the OPD's 2.5e-4. Only the learning rate changes, so 2e against 2c measures how hard
the warm-up pushes and nothing else; dropping the longest answers (the 20% over 12,288 tokens) is the next change if
2e's warm-up still loops.

Checks in 2e's first day, against which it is stopped rather than run for 6 days: reverse KL at step 1 between 2c's
0.19 and 2b's 0.54; at the step-0 in-training eval at most about 5% looping answers and at least M2RL-SFT's 52% right;
and no collapse at steps 20 to 30, where 2b's began. 2e's OPD runs on h100-2 (2 Qwen3-32B engines of 2 GPUs), 2c's ran
on b200-2 (4 of 1 GPU): the same recipe, a different layout.

## 2d's clip

Stopped at step 56 on 2026-11-23, after it collapsed as 2b did; its step-50 export is on the Hub
(`opd_2d_m2rl_sft_from_qwen3_32b_clip_step50`). 2b and 2d both rose to about 65 on the in-training AIME 2024 by step
15 and fell from step 20, within a few points of each other through step 45: 2d stayed near 20 from step 30 to 50, 2b
went between 8 and 37 through step 85. By step 30 about half of either run's eval answers ended in a loop at the
32,768-token limit, against 2a's 1 to 2%. The clip acted on 9% of tokens at step 1 and about 1% from step 35. The
setting:

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
bash skyrl-test/opd-blog-v2/2-alignment/sft_2c.sh 2e --smoke
```

## What to watch

On W&B `opd-blog-v2`, for 2a to 2e side by side:

| Metric | Aligned (2a) | Broken (2b), as expected |
|---|---|---|
| `eval/aime-2024/avg_score` | rises toward the teacher | flat or falls |
| `opd/teacher/_/reverse_kl` | falls from about 0.1 | starts near 0.8, stays high or rises |
| `opd/teacher/_/log_ratio_p01` | about -2.3 | far below (about -7.9) |
| `opd/truncation_rate`, `opd/repetition_rate` | stable | rise: answers run to the limit or loop |

2b broke as expected, and 2d with it ("2d's clip"); 2c and 2e should look like 2a.

## Risks

- **The teachers differ in size as well as lineage** (4B against 32B). A reader can ask whether size, not alignment,
  breaks 2b. The smoke tests' reverse KL is the direct evidence that it is alignment; 2c and 2e fixing 2b with the
  same 32B teacher is the stronger evidence.
- **The aligned teacher measured slightly stronger** (82.29/75.83 against Qwen3-32B's 80.68/72.24), so part of any
  2a-over-2b gap could be teacher strength rather than alignment. 2c and 2e, with the same teacher as 2b, are the
  clean comparison.
- **The student's tool calls.** Neither teacher writes `<tool_call>` here, so OPD should remove M2RL-SFT's bare tool
  calls early in every run, 2b included. Part of any early AIME gain is that alone; count the tool-call rate in the
  final evals' `responses.jsonl` before reading a gain as learned math.
- **Qwen3-32B is the slowest teacher**: even a node of its own (4 engines of 2 GPUs) keeps 2b's and 2e's steps
  longer than 2a's (`../teachers/README.md`). 2a shares RL_Math with MOPD (h100-2). If a teacher node is preempted, its runs fail and resume from their last checkpoint
  (`../teachers/README.md`).
- **The warm-up recipe is borrowed, not tuned**: 2e changes only its learning rate (above).

[paper]: https://arxiv.org/abs/2602.12566
[script]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L66-L127
[slime]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/utils/arguments.py#L730-L731
[loss]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/loss.py#L318-L337
[mean]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/cp_utils.py#L70-L76
[qwen3]: https://arxiv.org/abs/2505.09388
