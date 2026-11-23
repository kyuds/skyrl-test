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
- **2e, light warm-up:** 2c with its warm-up at a fifth of the learning rate. Stopped at OPD step 2: its student was
  2c's again ("Repairing 2c's warm-up").
- **2f, short-answer warm-up:** 2c with its warm-up on only the answers of at most 8,192 tokens. Stopped at OPD
  step 1: its student looped less than 2c's but was right less often (same section).
- **2g, warm-up on temperature-1.0 answers:** 2c with its warm-up on Qwen3-32B answers sampled as OPD samples
  (temperature 1.0, top-p 1.0, no top-k), not at Qwen's thinking settings (same section).

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
| `sft_2e_m2rl_sft_light_warmup`, `opd_2e_m2rl_light_warmup_from_qwen3_32b` | 2 | h100-2 | 2e, stopped at OPD step 2 on 2026-11-23; no longer in `config.sh` |
| `sft_2f_m2rl_sft_warmup_short_answers`, `opd_2f_m2rl_warmup_short_answers_from_qwen3_32b` | 2 | h100-2 | 2f, stopped at OPD step 1 on 2026-11-23; no longer in `config.sh` |
| `eval_sft_2c_warmup_math_temperature_1` | 2 | h100-2 | 2c's warm-up student at temperature 1.0 (`../0-evals`) |
| `traces_2g_qwen3_32b_temperature_1` | 2 | h100-2 | 2g's warm-up data: Qwen3-32B's answers at temperature 1.0 |
| `sft_2g_m2rl_sft_warmup_temperature_1` | 2 | h100-2 | 2g's warm-up SFT |
| `opd_2g_m2rl_warmup_temperature_1_from_qwen3_32b` | 2 | h100-2 | 2g |
| `opd_2a_m2rl_sft_from_m2rl_rl_math` | 2 | h100-4 | 2a |
| `eval_opd_2*_final` | | | each run's last export at 64 samples, M2RL's settings |

Each OPD run waits for its teacher's and its student's evals (`../0-evals`): Qwen3-32B or RL_Math, and M2RL-SFT.

## The OPD recipe, the same for 2a to 2g

Script: [`opd.sh`](opd.sh), block `HYPERPARAMETERS`. M2RL's multi-teacher OPD with one teacher; section 3 uses the
same block with three.

| Setting | Value | Source |
|---|---|---|
| Student | `Jackwang111/M2RL-SFT`; for 2c, 2e, 2f and 2g, the output of its warm-up | the proposal |
| Teacher | 2a: `Jackwang111/M2RL-RL_Math`, 4 engines × 1 GPU. 2b to 2g: `Qwen/Qwen3-32B`, thinking, 2 engines × 2 GPUs on H100s (4 × 1 on B200s) | the proposal |
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
RL_Math on 4 engines of 1 GPU for 2a (h100-4); Qwen3-32B on 2 engines of 2 GPUs for 2d, 2e, 2f and 2g (h100-2), and on
4 engines of 1 GPU on b200-2 for 2c and then 2b. That puts three runs on the three nodes left after MOPD at once; with
served teachers, the two H100 nodes would fit one run at a time. Micro-batch 1; checkpoints every 5 steps (the last 2 kept),
exports every 50, and the last export goes to the Hub.

## 2c's warm-up

**The data** ([`traces.sh`](traces.sh)): Qwen3-32B answers every one of the 22,056 math prompts twice. Per prompt, the
first answer that is correct (the kit's math grader) and finished (it closed its thinking and stopped at `<|im_end|>`)
is kept, then tokenized with M2RL-SFT's tokenizer. 2g draws its own answers the same way, sampled differently
(`traces.sh 2g`).

| Setting | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen3-32B`, thinking | the proposal |
| Prompts | all 22,056 of M2RL's math set, the ones 2c's OPD then trains on | the proposal ("Nemotron3 math"); the count is this kit's |
| Samples | 2 per prompt, first correct and finished one kept | this kit's: a second try keeps more of the hard prompts |
| Sampling | temperature 0.6, top-p 0.95, top-k 20. 2g: temperature 1.0, top-p 1.0, no top-k | [qwen3] section 4.6, thinking mode. 2g: OPD's rollouts (above) |
| Length | 16,384 tokens | this kit's: 2c's OPD rollout length, so the warm-up teaches answers that fit it |

**The SFT** ([`sft_2c.sh`](sft_2c.sh)). No paper gives a warm-up recipe. The closest published one is M2RL's own SFT of
M2RL-SFT on teacher answers, the multi-teacher off-policy distillation (MT-OFF) baseline, so this uses its settings:

| Setting | Value | Source |
|---|---|---|
| Base | `Jackwang111/M2RL-SFT` | |
| Data | the kept answers, one per prompt: 12,535 of the 22,056 prompts, 103M tokens, 8,252 per answer on average (measured 2026-11-11; Qwen3-32B was right on 50.2% of its 44,112 answers, and 25.4% hit the 16,384-token limit). 2f: only those of at most 8,192 tokens. 2g: its own | above; 2f and 2g: "Repairing 2c's warm-up" |
| Learning rate | 2c, 2f and 2g: 5e-5. 2e: 1e-5 | [paper] appendix A, MT-OFF. 2e: this kit's ("Repairing 2c's warm-up") |
| Batch | 512, 1 epoch (25 steps for 2c and 2e, 14 for 2f; 2g's depend on how many of its answers are kept) | [paper] MT-OFF; the epoch count is ASSUMED |
| Warmup | 10% of the steps, then constant | [paper] appendix A, SFT ("10% training steps ... for learning rate warmup"); no decay is given, so constant is ASSUMED |
| Optimizer | Adam, betas 0.9 and 0.98 (ASSUMED: not given for SFT, taken from [script]), weight decay 0.1 | [paper] appendix A, SFT |
| Grad clip | 1.0 | [slime] default |
| Length | 18,432 tokens (a 2,048 prompt + a 16,384 answer) | this kit's |

## Repairing 2c's warm-up: 2e, 2f and 2g

2c's warm-up at 5e-5 aligned its student but damaged it. At OPD step 1 its reverse KL to Qwen3-32B was 0.19, against
2b's 0.54 without a warm-up, and M2RL-SFT's bare tool calls were gone. But at step 0 (the in-training eval: 480
answers, temperature 0.6, 32,768 tokens) 45% were right against M2RL-SFT's 52%, 34% ran to the token limit and 26%
ended in a loop (the last 8,000 characters compress to under a tenth), against 19% and 3%. The kept answers themselves
do not loop: none of the 12,535 ends in one. 2c's OPD then spent its first 75 steps repairing this (loops down to 5%,
60% right) and stayed near 60% to step 250 (measured 2026-11-22).

Each repair changes one thing in 2c's warm-up and is screened before its 6-day OPD: it runs until OPD step 1 and is
stopped there unless its student passes three checks against M2RL-SFT and 2c's: reverse KL at step 1 between 2c's
0.19 and 2b's 0.54, at most about 5% looping answers at the step-0 eval, and at least M2RL-SFT's 52% right. A run that
passes continues, and must not collapse at steps 20 to 30, where 2b's began. All three run on h100-2 (2 Qwen3-32B
engines of 2 GPUs; 2c ran on b200-2 with 4 of 1 GPU): the same recipe, a different layout. 2e and 2f did not pass:

| At the start of OPD | M2RL-SFT | 2c (5e-5) | 2e (1e-5) | 2f (answers up to 8,192 tokens) |
|---|---|---|---|---|
| Reverse KL to Qwen3-32B, step 1 | 0.54 | 0.192 | 0.193 | 0.229 |
| Step-0 eval: right / looping / at the token limit | 52% / 3% / 19% | 45% / 26% / 34% | 47% / 23% / 31% | 42% / 18% / 30% |
| Policy entropy, step 1 | 0.67 | 0.39 | 0.39 | 0.41 |
| Warm-up's SFT loss, step 3 / 6 / last | | 0.419 / 0.414 / 0.345 (25) | 0.554 / 0.407 / 0.356 (25) | 0.432 / 0.477 / 0.374 (14) |

Both keep 2c's answers, sampled from Qwen3-32B at Qwen's thinking settings (temperature 0.6, top-p 0.95, top-k 20,
`traces.sh`); 2g changes that sampling ("2g" below).

**2e, the light warm-up: the learning rate, 1e-5 instead of 5e-5.** M2RL's 5e-5 is for answers from teachers RL'd from
this very student; Qwen3-32B's are not, so the same rate might move the student much further. It did not matter.
Stopped at OPD step 2 on 2026-11-23, its student was 2c's again. At a fifth of the rate the loss fell more slowly for
3 steps and was level with 2c's by step 6: the student takes on the answers' style within about 2,500 of them either
way. What the warm-up teaches, not how hard it pushes, does the
damage. 2e's warm-up is on the Hub (`sft_2e_m2rl_sft_light_warmup`).

**2f, the short-answer warm-up: only the answers of at most 8,192 tokens.** 2c's recipe at 5e-5, on the kept answers
whose loss-masked tokens number at most 8,192: 6,756 of 12,535 (54%), 4,863 tokens each on average, so 14 steps
instead of 25. The 10% of answers over 14,000 tokens end within 2,400 tokens of the 16,384-token limit; a
student that copies their length runs long, and 2c's warm-up student ran to the limit on 34% of its eval answers.
`filter_traces.py` built the store from 2c's on the node (`sft_2c_qwen3_32b_traces_answers_le8192`). Length was not
the cause. Stopped at OPD step 1 on 2026-11-23, 2f's student had never
seen an answer over 8,192 tokens, yet 30% of its eval answers ran to the 32,768-token limit; fewer looped than 2c's
(18% against 26%), and fewer were right (42% against 45%). 2f's warm-up is on the Hub
(`sft_2f_m2rl_sft_warmup_short_answers`).

**2g, the warm-up on temperature-1.0 answers: Qwen3-32B's answers sampled as OPD samples (temperature 1.0, top-p 1.0,
no top-k), not at Qwen's thinking settings.** 2c, 2e and 2f learn from answers drawn at temperature 0.6, top-p 0.95 and
top-k 20: a sharpened copy of Qwen3-32B. A student that learns the copy is sharpened by its training, and the eval at
temperature 0.6 sharpens it again; Qwen's model card warns that its models repeat endlessly under greedy decoding
([qwen3-card]). Sampled at temperature 1.0, 2c's warm-up student hardly loops (`eval_sft_2c_warmup_math_temperature_1`,
`../0-evals`: 64 samples per AIME problem, 32,768 tokens, measured 2026-11-23):

| 2c's warm-up student, AIME 2024 and 2025 | Temperature 0.6, top-p 0.95 (step-0 eval) | Temperature 1.0, top-p 1.0 |
|---|---|---|
| Right | 45% | 53% (AIME 2024 53.8, 2025 51.9) |
| Looping | 26% | 0.3% |
| At the 32,768-token limit | 34% | 9% |

The two come from different harnesses (the in-training eval at 8 samples; `../0-evals` at 64) at the same length. At 1.0
it is right as often as M2RL-SFT is at 0.6 (53% against 52%) and loops less (0.3% against 3%). So its loops come from
sampling it sharply, not from what its answers teach. 2g's student learns Qwen3-32B's own distribution, so at the eval's
temperature 0.6 it should sample as 2c's does at 1.0 (ASSUMED: the hypothesis 2g tests), and its OPD scores the
distribution it was warmed up toward. Not done instead: evaluating at temperature 1.0, since M2RL's numbers and every
other run's are at 0.6. 2g's answers are its own (`sft_2g_qwen3_32b_traces_temperature_1`), drawn on h100-2's 4 engines
of 2 GPUs: 2c's took 8.5 hours on b200-2's 8 engines of 1 GPU, so about a day is expected (an estimate). `traces.sh`
saves each engine's answers in 8 parts, so a preemption loses at most one. Its OPD is screened as 2e's and 2f's were.

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
bash skyrl-test/opd-blog-v2/2-alignment/traces.sh 2c --smoke
```

```bash
bash skyrl-test/opd-blog-v2/2-alignment/sft_2c.sh 2c --smoke
```

## What to watch

On W&B `opd-blog-v2`, for 2a to 2d side by side:

| Metric | Aligned (2a) | Broken (2b), as expected |
|---|---|---|
| `eval/aime-2024/avg_score` | rises toward the teacher | flat or falls |
| `opd/teacher/_/reverse_kl` | falls from about 0.1 | starts near 0.8, stays high or rises |
| `opd/teacher/_/log_ratio_p01` | about -2.3 | far below (about -7.9) |
| `opd/truncation_rate`, `opd/repetition_rate` | stable | rise: answers run to the limit or loop |

2b broke as expected, and 2d with it ("2d's clip"); 2c should look like 2a.

## Risks

- **The teachers differ in size as well as lineage** (4B against 32B). A reader can ask whether size, not alignment,
  breaks 2b. The smoke tests' reverse KL is the direct evidence that it is alignment; 2c fixing 2b with the same 32B
  teacher is the stronger evidence.
- **The aligned teacher measured slightly stronger** (82.29/75.83 against Qwen3-32B's 80.68/72.24), so part of any
  2a-over-2b gap could be teacher strength rather than alignment. 2c, with the same teacher as 2b, is the clean
  comparison.
- **The student's tool calls.** Neither teacher writes `<tool_call>` here, so OPD should remove M2RL-SFT's bare tool
  calls early in every run, 2b included. Part of any early AIME gain is that alone; count the tool-call rate in the
  final evals' `responses.jsonl` before reading a gain as learned math.
- **Qwen3-32B is the slowest teacher**: even a node of its own (4 engines of 2 GPUs) keeps 2b's steps
  longer than 2a's (`../teachers/README.md`). 2a shares RL_Math with MOPD (h100-2). If a teacher node is preempted, its runs fail and resume from their last checkpoint
  (`../teachers/README.md`).
- **The warm-up recipe is borrowed, not tuned**: 2e and 2f each changed one thing in it, and neither repaired it;
  2g changes the answers' sampling (above).
- **2g's answers may be fewer and rougher.** At temperature 1.0 Qwen3-32B is likely right less often and runs to the
  16,384-token limit more, so fewer prompts keep an answer (2c kept 12,535 of 22,056), and a smaller warm-up moves
  the student less. `traces.sh` prints how many it kept.

[paper]: https://arxiv.org/abs/2602.12566
[script]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/scripts/multi_teacher_opd.sh#L66-L127
[slime]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/utils/arguments.py#L730-L731
[loss]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/loss.py#L318-L337
[mean]: https://github.com/Mosi-AI/M2RL/blob/c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4/M2RL/slime/slime/backends/training_utils/cp_utils.py#L70-L76
[qwen3]: https://arxiv.org/abs/2505.09388
[qwen3-card]: https://huggingface.co/Qwen/Qwen3-32B
