# OPD versus naive GRPO on one 4xH100 node: Qwen3.5-0.8B student, Qwen3.5-9B teacher

Written 2026-09-25 for PR [#2256](https://github.com/NovaSky-AI/SkyRL/pull/2256) (branch
`kyuds/opd-entrypoint`). Methodology from the NovaSky post
[On-Policy Distillation in SkyRL](https://novasky-ai.notion.site/On-Policy-Distillation-in-SkyRL-2a38f0016b9d8095ae7cc5660b18debb):
DAPO-Math-17k prompts, pure reverse-KL OPD, batch 512 × 16 with no mini-batching, LR 1e-5,
temperature 1.0, prompt 2048 / response 8192, AIME24 avg@32 every 5 steps; and its RL baseline
(LR 1e-6, mini-batch 32), here as naive GRPO. Its W&B report is public by link but not readable with
the lab key, so the comparison to the post is by eye; the comparison that matters is the one you run.

## The configuration to test first

| | Model | GPUs | Script |
|---|---|---|---|
| Teacher | `Qwen/Qwen3.5-9B` (post-trained), two stock `vllm serve` processes, round-robin | 0, 1 | `02_serve_teacher.sh --pair post` |
| Student, OPD | `Qwen/Qwen3.5-0.8B`, `colocate_all`, FSDP over 2 ranks + 2 vLLM engines | 2, 3 | `04_run_opd.sh` |
| Student, GRPO | `Qwen/Qwen3.5-0.8B`, same, no teacher | 0–3 (after the teacher is stopped) | `05_run_grpo.sh` |

The pair shares one tokenizer (verified from the HF file hashes: `Qwen3.5-0.8B` and `Qwen3.5-9B` both
carry `tokenizer.json` 5f9e4d49…). The Base pair (`OPD_PAIR=base`: 0.8B-Base ← 9B-Base, tokenizer
fe000e3e…) is the same kit with one variable changed, for later; never mix a Base teacher with a
post-trained student.

Both models are `Qwen3_5ForConditionalGeneration`: multimodal, hybrid full-attention / Gated DeltaNet,
248K vocabulary. SkyRL trains the student text-only on FSDP with the flags of
`examples/train/models/run_qwen3.5_0.8b.sh` (`language_model_only=true` on policy, ref and engines;
`remove_microbatch_padding=false`; NCCL weight sync). The 9B teacher fits one H100 easily (19 GB of
weights; only 8 of 32 layers keep a KV cache), but a 9B prefill is several times slower than the 0.8B
student's generation, so the teacher gets two GPUs and still sets the pace of an OPD step.

## Step by step

All commands run from the SkyRL checkout root on the node, on the PR branch.

**0. One-time setup.**
```
git clone <SkyRL remote> && cd SkyRL && git checkout kyuds/opd-entrypoint
git clone https://github.com/kyuds/skyrl-test.git                       # these scripts; later: git -C skyrl-test pull
export WANDB_API_KEY=...                                                # never printed by the scripts
```

**1. Environment.** Checks the branch (refuses on `main`), `uv`, the four GPUs and the key; exports the
`OPD_*` knobs (`post` pair, 9B teacher, GPUs 0,1 teacher / 2,3 student, thinking off).
```
OPD_PAIR=post source skyrl-test/opd-4xh100/00_env.sh
```

**2. Data and model caches.** DAPO-17k + AIME24 (cleaned, as the post), a 256-row GSM8K eval slice,
and the HF cache for `Qwen3.5-0.8B` and `Qwen3.5-9B` (19 GB). Run once.
```
bash skyrl-test/opd-4xh100/01_prepare_data.sh
```

**3. Serve the teacher.** One `vllm serve` per GPU on 0 and 1 (ports 8100, 8101), waits until both
answer `/v1/models`, checks the served id and that `max_model_len` ≥ 2048 + 8192 + 1 (the branch has
no preflight checks yet), and writes the manifest the run scripts read. Logs in `~/logs/teacher_post_gpu*.log`.
```
bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair post
bash skyrl-test/opd-4xh100/02_serve_teacher.sh --status          # UP/DOWN per URL, any time
```

**4. Smoke the OPD path.** Three tiny steps (16 prompts × 8, 2k responses, 4-sample evals) on GPUs
2,3 against the live teacher. Well under an hour once the caches are warm. Then read the run back from
W&B: every `opd/*` key present, reverse KL fell, exposed teacher time a small fraction of generation,
eval improved (the last two can legitimately fail on three tiny steps; the first two must pass).
```
bash skyrl-test/opd-4xh100/03_run_opd_smoke.sh
uv run --isolated --extra fsdp skyrl-test/opd-4xh100/check_opd_run.py --project opd_4xh100 --run_name "$(cat ~/logs/last_opd_run)"
```
This is the first live exercise of the vLLM teacher path (the `prompt_logprobs` wire format was
verified against SkyRL's own parser, not a live server), so a failure here is most likely there; the
teacher logs and the trainer's error name the request.

**5. The OPD run.** 40 steps at the post's shape, evals at 0, 5, 10, …; checkpoints every 10 steps.
The run name is printed and saved to `~/logs/last_opd_run`. If it is interrupted, re-run with
`OPD_RUN_NAME=<that name>` and it resumes from the last checkpoint.
```
bash skyrl-test/opd-4xh100/04_run_opd.sh                          # OPD_MAX_STEPS=N to change the cap
uv run --isolated --extra fsdp skyrl-test/opd-4xh100/check_opd_run.py --project opd_4xh100 --run_name "$(cat ~/logs/last_opd_run)"
```

**6. Stop the teacher, then the GRPO baseline.** Same student, prompts, evals, batch shape and lengths;
plain GRPO with SkyRL's defaults spelled out in the script (group-normalized advantages, `regular`
PPO-clip at 0.2, token-mean, KL loss 0.001 to the reference, none of DAPO's additions), LR 1e-6,
mini-batch 32, 200 steps. It runs on all four GPUs and refuses to start while the teacher holds 0,1.
Smoke it first the same way.
```
bash skyrl-test/opd-4xh100/02_serve_teacher.sh --pair post --stop
bash skyrl-test/opd-4xh100/05_run_grpo.sh --smoke
uv run --isolated --extra fsdp skyrl-test/opd-4xh100/check_opd_run.py --grpo --project opd_4xh100 --run_name "$(cat ~/logs/last_grpo_run)"
bash skyrl-test/opd-4xh100/05_run_grpo.sh                         # GRPO_MAX_STEPS=N to change the cap
```
`GRPO_GPUS=2,3 GRPO_NUM_GPUS=2` runs it on the OPD student's GPUs instead, for a like-for-like
`timing/step`; the default trades that for roughly half the wall clock.

**7. Compare.** Both runs' eval scores per eval step, side by side, with the rollouts consumed at each
step (the post's axis: OPD reached in ~20 steps what RL reached in ~200) and mean step times.
```
uv run --isolated --extra fsdp skyrl-test/opd-4xh100/compare_runs.py --project opd_4xh100 \
    --run opd="$(cat ~/logs/last_opd_run)" --run grpo="$(cat ~/logs/last_grpo_run)"
```

**8. Later, the Base pair.** Same steps with `OPD_PAIR=base` in step 1 (teacher `Qwen3.5-9B-Base` on
ports 8000–8001, student `Qwen3.5-0.8B-Base`).

## What to look at on W&B (project `opd_4xh100`)

- OPD: `opd/reverse_kl` should fall fast and flatten (the post's 1.7B ← 4B pair flattened near 0.09;
  different models here, so shape not number), `opd/reverse_kl_abs_max`, and the two advantage scales.
- OPD timing: `timing/generate` versus `opd/teacher_time_exposed`; with a 9B teacher on two GPUs expect
  the exposed time to be a large share, the honest cost of this teacher. `opd/teacher_time_per_group_mean`
  is the per-prompt-group scoring time.
- Both: `eval/all/avg_score`, `eval/<data_source>/avg_score` for the AIME and GSM8K sets,
  `eval/all/pass_at_32`, `policy/policy_entropy`, and the GRPO run's `reward/*`.
- Cost, all ASSUMED until the first `timing/step`: OPD steps of 20–25 minutes dominated by the teacher
  prefill, ~15–17 hours for 40 steps; GRPO steps of ~10 minutes on four GPUs, ~1.5–2 days for 200
  steps plus 40 avg@32 evals. Eval every 5 steps means either run can be stopped early once flat.

## Knobs (exported by `00_env.sh`; override in the shell before sourcing or on the command line)

| Variable | Default | Meaning |
|---|---|---|
| `OPD_PAIR` | `post` | `post` or `base`; selects student and teacher |
| `OPD_TEACHER_MODEL` | `Qwen/Qwen3.5-9B` (`-Base` for `base`) | HF path or local dir; the served name and `trainer.teacher.model` |
| `OPD_THINKING` | `false` | `enable_thinking` for the chat template, both runs |
| `OPD_TEACHER_GPUS` / `OPD_STUDENT_GPUS` / `OPD_NUM_STUDENT_GPUS` | `0,1` / `2,3` / `2` | the split |
| `OPD_TEACHER_PORT_BASE` | `8100` post / `8000` base | teacher server i listens on base + i |
| `OPD_TEACHER_MAX_MODEL_LEN` | `10496` | must be ≥ 2048 + 8192 + 1; checked against the served value |
| `OPD_TEACHER_MAX_CONCURRENCY` | `256` | teacher requests in flight across the two servers (128 each) |
| `OPD_MAX_STEPS` / `GRPO_MAX_STEPS` | `40` / `200` | step caps |
| `GRPO_GPUS` / `GRPO_NUM_GPUS` | `0,1,2,3` / `4` | GRPO's GPUs |
| `OPD_PROJECT` / `OPD_DATA` / `OPD_LOGS` | `opd_4xh100` / `~/data` / `~/logs` | W&B project, data root, logs and manifests |

Every run script forwards extra `key=value` overrides to the entrypoint, e.g.
`bash 04_run_opd.sh trainer.algorithm.opd.use_task_reward=true` for the mixed variant. Dict overrides
need a space after the colon (`"{enable_thinking: false}"`); the parser reads `{k:v}` as a key named `k:v`.

## Decisions, and what is not yet verified

- **Thinking off** for both runs so a 0.8B thinker does not truncate at 8k every sample and both
  methods see the same prompt format. `OPD_THINKING=true` flips it (long responses, slower steps).
- **Two eval sets.** AIME24 is the post's; 0.8B models sit near its floor, so the GSM8K slice is there
  to give the curves signal. Per-dataset keys are `eval/<data_source>/avg_score`.
- **GRPO on four GPUs, OPD student on two.** Steps and rollouts are the comparison axis, not wall
  clock; set `GRPO_GPUS=2,3 GRPO_NUM_GPUS=2` if wall clock is the question.
- **Not verified on a GPU:** anything. What has been verified: every script's generated argument list
  parses and passes `validate_cfg` (and `validate_opd_cfg`) on the branch for this exact configuration,
  the launcher writes the manifests the run scripts read, and the GPU-collision guard refuses a
  student on a GPU a running teacher holds. The 9B checkpoints and tokenizer hashes were checked on HF.
- **If vLLM 0.28 rejects `prompt_logprobs` with prefix caching on**, add `--no-enable-prefix-caching`
  to the `vllm serve` line in `02_serve_teacher.sh`; not expected on this version.
- **If the teacher OOMs at startup**, lower `--max-num-seqs` in `02_serve_teacher.sh` (128 per server).

## Files

`00_env.sh` env and knobs · `01_prepare_data.sh` data and caches · `02_serve_teacher.sh` teacher launcher
(`--pair`, `--model`, `--gpus`, `--port-base`, `--status`, `--stop`, `--stop-all`) · `03_run_opd_smoke.sh` ·
`04_run_opd.sh` · `05_run_grpo.sh` (`--smoke`) · `_common.sh` shared flags and guards · `check_opd_run.py`
(`--grpo`) · `compare_runs.py` · `slice_parquet.py`.

## Next, after these runs

The token-overlap and overlap-advantage diagnostics from arXiv 2604.13016 §2.3 (design to follow; see
`research/readings/rethinking-opd-dynamics-metrics.md`) would make the OPD side mechanistic: the
paper's claim is that successful OPD is progressive alignment on the shared top-k tokens.
