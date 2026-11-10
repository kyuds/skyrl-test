# Section 0: reproduce the published numbers first

Before any model here is used as a teacher or a starting point, it is evaluated with its authors' settings and has
to land near their published numbers. The same harness then scores every SFT output and every final OPD export.

## The jobs

Stage 1, before anything trains against these models:

| Job | Node | Model | Benchmarks | Published target |
|---|---|---|---|---|
| `eval_qwen3_8b_math` | h100-1 | `Qwen/Qwen3-8B` | AIME 24, 25 | 76.0, 67.3 ([qwen3] Table 17, thinking) |
| `eval_qwen3_32b_math` | h100-2 | `Qwen/Qwen3-32B` | AIME 24, 25 | 81.4, 72.9 ([qwen3] Table 13, thinking) |
| `eval_m2rl_sft_math` | h100-3 | `Jackwang111/M2RL-SFT` | AIME 24, 25 | 56.04, 51.30 ([m2rl] Table 3) |
| `eval_m2rl_rl_math_math` | h100-4 | `Jackwang111/M2RL-RL_Math` | AIME 24, 25 | 77.66, 70.42 |
| `eval_m2rl_sft_code_if` | h100-5 | `Jackwang111/M2RL-SFT` | LCB v5, v6, IFEval, IFBench | 55.92, 52.00, 79.48, 38.44 |
| `eval_m2rl_rl_coding_code` | h100-6 | `Jackwang111/M2RL-RL_Coding` | LCB v5, v6 | 65.00, 58.86 |
| `eval_m2rl_rl_if_if` | h100-6 | `Jackwang111/M2RL-RL_IF` | IFEval, IFBench | 90.94, 59.52 |

Later stages: `eval_sft_1a` (target: Thinking Machines' 60% on AIME 2024), and one `eval_<run>_final` after every
OPD run, on its last export.

**Pass mark:** within about 3 points of the published number. AIME's 30 problems at 64 samples each give a standard
error near 1 point, so 3 points is a real difference. A model that misses is not used until the gap is understood:
first check the cut rate (answers that hit the length limit) and the grader, then the settings below.

## Settings

Script: [`run.sh`](run.sh), block `HYPERPARAMETERS`; the samples per problem are `BENCH_SAMPLES` in
[`benchmarks.py`](benchmarks.py).

| | Qwen preset (`qwen`) | M2RL preset (`m2rl`) |
|---|---|---|
| Source | [qwen3] section 4.6, thinking mode | [m2rl] appendix A, "Evaluation" |
| Temperature | 0.6 | 0.6 |
| Top-p | 0.95 | 0.95, ASSUMED (given only for the paper's GenRM study, same appendix) |
| Top-k | 20 | none |
| Answer length | 38,912 tokens for AIME, 32,768 otherwise | 32,768 tokens ("32K") |
| Samples per problem | 64 for AIME ("For each question, we sample 64 times") | 64 AIME, 8 LiveCodeBench v5, 1 v6, 4 IFEval, 1 IFBench |
| Thinking | on (the chat template's default) | on |
| Used for | Qwen3-8B, Qwen3-32B, and section 1's models | M2RL's models and every section 2 and 3 model |

The score is the mean over all samples (Avg@K), in percent. A model's own context caps its answers: `generate.py`
shortens a request that would not fit. Section 2's runs all use the M2RL preset, so they compare on one setting; the
teachers' own scores are their authors' settings. Qwen3-32B runs two GPUs per vLLM engine on an H100 (its 64 GB of
weights leave one GPU no room for a 38,912-token context), one on a B200.

## Benchmarks and graders

- **AIME 2024, 2025:** the kit's own files (`../data/README.md`), M2RL's prompt. Graded by `common/kit_main.py`'s
  `is_correct`, the same function the OPD runs log: the answer after the last `</think>`, compared with math-verify.
  M2RL grades with QwQ-32B's evaluator, so small differences are expected.
- **LiveCodeBench v5, v6:** `livecodebench/code_generation_lite` at `0fe84c39`, problems dated
  [2024-08-01, 2025-02-01) and [2025-02-01, 2025-05-01). **ASSUMED:** M2RL names the versions, not the dates; these are
  the windows most 2025 reports use. LiveCodeBench's own `release_v6` file instead adds 175 problems from January to
  April 2025, of which 131 fall in the v6 window here. If M2RL-SFT's LCB numbers miss, the window is the first suspect.
  The prompt is LiveCodeBench's own generic one, sent as one user turn; the grader is LiveCodeBench's (`lcb_runner` at
  `28fef95e`) on the public and private tests, on the last code block after `</think>`.
- **IFEval, IFBench:** `google/IFEval` at `966cd895` (541 prompts) and IFBench's test set (300). Both graded by
  IFBench's strict checker (`allenai/IFBench` at `cd189a9c`), which includes the classic IFEval checks, on the text
  after `</think>`; prompt-level strict accuracy. M2RL uses open-instruct's IFEvalG verifier, built from the same
  official code.

The graders are cloned at those commits into `~/opd-store/tools/` on first use. LiveCodeBench runs model-written code
on the node, in subprocesses with its own guards and timeouts, as its official harness does.

## Run one by hand

On a node, from `~/SkyRL`. Through the job runner, as the orchestrator does:

```bash
nohup bash skyrl-test/opd-blog-v2/common/job.sh eval_qwen3_8b_math > /dev/null 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/eval_qwen3_8b_math.log
```

Any model, outside the job table:

```bash
bash skyrl-test/opd-blog-v2/0-evals/run.sh eval_my_model Qwen/Qwen3-4B qwen math
```

A smoke run (2 samples, 512 tokens; checks the plumbing, scores mean nothing):

```bash
bash skyrl-test/opd-blog-v2/0-evals/run.sh eval_qwen3_8b_math Qwen/Qwen3-8B qwen all --smoke
```

The model can be a Hub id, `kit:<SFT job>` (its output; fetched from the Hub when not on this node) or `run:<job>`
(that run's last export on this node).

## Where results go

- `~/opd-store/evals/<job>/`: `requests.jsonl`, `responses.jsonl`, `graded.jsonl` (one line per sample) and
  `scores.json`.
- W&B project `opd-blog-v2-evals`, one run per job, the scores in its summary.
- `orchestrate.sh` shows the scores in its PROGRESS column once a job is graded.

A rerun reuses what already finished: the requests, any GPU's finished share of the sampling, and the responses.

## Time

Rough, not measured. AIME at 64 samples is 3,840 answers of up to 32,768 tokens: about an hour for an 8B model on
8 H100s, and several hours for Qwen3-32B at 38,912 tokens on four engines of two GPUs. LiveCodeBench v5 at 8 samples
is about 2,200 answers, plus grading.

## Verified and assumed

Verified on 2026-11-09, on the Mac: `benchmarks.py` builds AIME, LiveCodeBench v6 (131 problems), IFEval (541) and
IFBench (300). `grade.py` on hand-written responses: a correct boxed answer passes and a wrong or unfinished one
fails; LiveCodeBench passes a correct stdin program and a correct `class Solution` method and fails a wrong program
and an unfinished answer; the IF checker runs every IFEval and IFBench instruction without error and gives the same
score twice.

Not verified: `generate.py`, which needs GPUs and vLLM; the v5 window's count; and every published number, which is
what these jobs are for.

[qwen3]: https://arxiv.org/abs/2505.09388
[m2rl]: https://arxiv.org/abs/2602.12566
