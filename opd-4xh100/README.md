# OPD versus naive GRPO on one 4xH100 node: Qwen3.5-0.8B student, Qwen3.5-9B teacher

Written 2026-09-25 for PR [#2256](https://github.com/NovaSky-AI/SkyRL/pull/2256); runs on branch
`kyuds/opd-teacher-launching`, which is the PR's entrypoint plus the teacher launched by the job
(`trainer.teacher.backend=skyrl`; design in `plan/opd-entrypoint/teacher-launcher.md`). Methodology
from the NovaSky post
[On-Policy Distillation in SkyRL](https://novasky-ai.notion.site/On-Policy-Distillation-in-SkyRL-2a38f0016b9d8095ae7cc5660b18debb):
DAPO-Math-17k prompts, pure reverse-KL OPD, batch 512 × 16 with no mini-batching, LR 1e-5,
temperature 1.0, prompt 2048 / response 8192, AIME24 avg@32 every 5 steps; and its RL baseline
(LR 1e-6, mini-batch 32), here as naive GRPO. Its W&B report is public by link but not readable with
the lab key, so the comparison to the post is by eye; the comparison that matters is the one you run.

## The configuration to test first

| | Model | GPUs | Script |
|---|---|---|---|
| Teacher | `Qwen/Qwen3.5-9B` (post-trained), launched **by the OPD run** as SkyRL inference servers behind a `RemoteInferenceClient` | 2 (two servers, TP 1) | part of `03_run_opd.sh` |
| Student, OPD | `Qwen/Qwen3.5-0.8B`, `colocate_all`, FSDP over 2 ranks + 2 vLLM engines | the other 2 | `03_run_opd.sh` |
| Student, GRPO | `Qwen/Qwen3.5-0.8B`, same, no teacher | all 4 | `04_run_grpo.sh` |

Ray picks the devices. The teacher servers are Ray actors that hold their GPUs in Ray's ledger, so the
student's placement group lands on whatever is left; no script names a device id. The run joins the
node's Ray cluster (an Anyscale workspace starts one; elsewhere `ray start --head` first), as SkyRL's
own Anyscale CI does.

**Status (2026-09-26).** The launched teacher is implemented: the OPD run creates the teacher's vLLM
servers in a placement group of their own, before the student's engines and workers exist, drives them
through a `RemoteInferenceClient` like the student's engines, and tears them down when the run ends.
Everything teacher-related in this kit is one array, `TEACHER_OPTS` in `_common.sh`: two TP-1 servers
(`inference_engine.num_engines=2`), text-only loading, and the in-flight cap. Defaults the launched block
brings are not repeated: memory fraction 0.9, prefix caching off, `max_model_len` = longest input +
longest response + 1 (4097 for the smoke, 10241 for the run). Nothing has run on a GPU yet.

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

Every script `cd`s into the SkyRL checkout before calling `uv`, so the commands work from any directory.

**0. One-time setup.** Two checkouts under one root, side by side (the node's `~/default/kyuds/`):
the SkyRL repo on the PR branch, and the public `kyuds/skyrl-test` repo that carries this kit. The
scripts find SkyRL next to the kit (or around it, if `skyrl-test` is cloned inside `SkyRL/` as the
workspace rules say); `SKYRL_DIR=/path/to/SkyRL` overrides the search.
```
cd ~/default/kyuds
git clone <SkyRL remote> && (cd SkyRL && git checkout kyuds/opd-teacher-launching)
git clone https://github.com/kyuds/skyrl-test.git                       # these scripts; later: git -C skyrl-test pull
export WANDB_API_KEY=...                                                # never printed by the scripts
```
If stand-alone `vllm serve` teachers from an earlier version of this kit are still running on the
node, stop them first, or the run's own teacher servers land on the same GPUs and run out of memory:
```
pkill -TERM -f "vllm serve" ; sleep 10 ; nvidia-smi
```

**1. Environment.** Finds the SkyRL checkout and refuses unless it has the OPD entrypoint and the
launched teacher (i.e. is on the branch above); checks `uv`, the GPUs, the W&B key, that a Ray
cluster is up and which Ray version it runs (the run scripts override SkyRL's pin to it at run time
when they differ); exports the `OPD_*`
knobs (`post` pair, 9B teacher on 2 GPUs, student on 2, thinking off) plus `SKYRL_DIR` and
`OPD_KIT_DIR`. Source it from bash, in the shell you will run everything else from.
```
OPD_PAIR=post source skyrl-test/opd-4xh100/00_env.sh
```

**Detached runs.** Every run script below can also be started detached, so a dropped terminal or a
sleeping laptop does not kill it: `bg.sh` gives the run its own session with hangups ignored, sends
its output to a log under `~/logs`, and appends a last line with the exit status. Start it from the
shell where you sourced `00_env.sh`, because the run inherits those exports. For example, instead of
`bash skyrl-test/opd-4xh100/04_run_grpo.sh --smoke`:
```
bash skyrl-test/opd-4xh100/bg.sh 04_run_grpo.sh --smoke
```
Follow it (`~/logs/latest.log` always points at the newest detached run; the run is done when a line
starting with `===` reports its exit status):
```
tail -f ~/logs/latest.log
```
From any shell, even after reconnecting:
```
bash skyrl-test/opd-4xh100/bg.sh --status
```
Stop it, which ends its Ray job and frees the GPUs:
```
bash skyrl-test/opd-4xh100/bg.sh --stop
```
After reconnecting, source `00_env.sh` again before the check commands, which need its exports.

**2. Data and model caches.** DAPO-17k + AIME24 (cleaned, as the post), the GSM8K validation split,
and the HF cache for `Qwen3.5-0.8B` and `Qwen3.5-9B` (19 GB). Run once. It ends by building the eval set
(AIME24 + the first 256 GSM8K rows, under `$OPD_DATA/opd-eval`), which every run script also rebuilds.
```
bash skyrl-test/opd-4xh100/01_prepare_data.sh
```

**3. The GRPO baseline (runs today).** Same student, prompts, evals, batch shape and lengths as the
OPD run; plain GRPO with SkyRL's defaults spelled out in the script (group-normalized advantages,
`regular` PPO-clip at 0.2, token-mean, KL loss 0.001 to the reference, none of DAPO's additions),
LR 1e-6, mini-batch 32, 200 steps, all four GPUs. Smoke it first: three tiny steps. Each command
below runs after the previous one has exited.
```
bash skyrl-test/opd-4xh100/04_run_grpo.sh --smoke
```
Then read the smoke back from W&B (in the same shell: the check needs the `SKYRL_DIR` and
`OPD_KIT_DIR` exports, and `uv run` has to execute inside the SkyRL project because its environment
holds `wandb`, hence the `cd`):
```
(cd "$SKYRL_DIR" && uv run --isolated --extra fsdp "$OPD_KIT_DIR/check_opd_run.py" --grpo --project opd_4xh100 --run_name "$(cat ~/logs/last_grpo_run)")
```
Then the full run (`GRPO_MAX_STEPS=N` changes the cap; interrupted? re-run with
`GRPO_RUN_NAME=<the printed name>` and it resumes from the last checkpoint):
```
bash skyrl-test/opd-4xh100/04_run_grpo.sh
```

**4. Smoke the OPD path.** Three tiny steps (16 prompts × 8, 2k responses, 4-sample evals); the run
brings up its two teacher servers first, so add the 9B model load to the time. Well under an hour once
the caches are warm.
```
bash skyrl-test/opd-4xh100/02_run_opd_smoke.sh
```
Then read it back: every `opd/*` key present, reverse KL fell, exposed teacher time a small fraction
of generation, eval improved (the last two can legitimately fail on three tiny steps; the first two
must pass).
```
(cd "$SKYRL_DIR" && uv run --isolated --extra fsdp "$OPD_KIT_DIR/check_opd_run.py" --project opd_4xh100 --run_name "$(cat ~/logs/last_opd_run)")
```
This is the first live exercise of the launched teacher path (scoring goes through the deployment's
`RemoteInferenceClient`; unit-tested against fakes, not a live server), so a failure here is most likely
there; the trainer's error names the request, and the teacher servers' logs sit next to the student's
under the run's log path.

**5. The OPD run.** 40 steps at the post's shape, evals at 0, 5, 10, …; checkpoints every 10 steps.
The run name is printed and saved to `~/logs/last_opd_run`. Needs the node to itself (the GRPO run
must have finished): with the teacher on two GPUs and the student on two, all four are in use, and a
run that cannot get its GPUs fails on a placement-group timeout after 180 s (`SKYRL_RAY_PG_TIMEOUT_IN_S`).
```
bash skyrl-test/opd-4xh100/03_run_opd.sh
```
`OPD_MAX_STEPS=N` changes the cap; if interrupted, re-run with `OPD_RUN_NAME=<that name>` and it
resumes from the last checkpoint. After it finishes, the same check as in step 4:
```
(cd "$SKYRL_DIR" && uv run --isolated --extra fsdp "$OPD_KIT_DIR/check_opd_run.py" --project opd_4xh100 --run_name "$(cat ~/logs/last_opd_run)")
```

**6. Compare.** Both runs' eval scores per eval step, side by side, with the rollouts consumed at each
step (the post's axis: OPD reached in ~20 steps what RL reached in ~200) and mean step times.
```
(cd "$SKYRL_DIR" && uv run --isolated --extra fsdp "$OPD_KIT_DIR/compare_runs.py" --project opd_4xh100 \
    --run opd="$(cat ~/logs/last_opd_run)" --run grpo="$(cat ~/logs/last_grpo_run)")
```

**7. Later, the Base pair.** Same steps with `OPD_PAIR=base` in step 1 (teacher `Qwen3.5-9B-Base`,
student `Qwen3.5-0.8B-Base`); each run launches its own teacher, so the pairs simply run one after
the other.

## What to look at on W&B (project `opd_4xh100`)

- OPD: `opd/reverse_kl` should fall fast and flatten (the post's 1.7B ← 4B pair flattened near 0.09;
  different models here, so shape not number), `opd/reverse_kl_abs_max`, and the two advantage scales.
- OPD timing: `timing/generate` versus `opd/teacher_time_exposed`; with a 9B teacher on two GPUs expect
  the exposed time to be a large share, the honest cost of this teacher. `opd/teacher_time_per_group_mean`
  is the per-prompt-group scoring time.
- Both: `eval/all/avg_score`, `eval/<data_source>/avg_score` for the AIME and GSM8K sets,
  `eval/all/pass_at_32`, `policy/policy_entropy`, and the GRPO run's `reward/*`. `vllm/train/*` is the
  student's engines only: the launched teacher block turns Ray Prometheus stats off.
- Cost, all ASSUMED until the first `timing/step`: OPD steps of 20–25 minutes dominated by the teacher
  prefill, ~15–17 hours for 40 steps; GRPO steps of ~10 minutes on four GPUs, ~1.5–2 days for 200
  steps plus 40 avg@32 evals. Eval every 5 steps means either run can be stopped early once flat.

## Knobs (exported by `00_env.sh`; override in the shell before sourcing or on the command line)

| Variable | Default | Meaning |
|---|---|---|
| `OPD_PAIR` | `post` | `post` or `base`; selects student and teacher |
| `OPD_TEACHER_MODEL` | `Qwen/Qwen3.5-9B` (`-Base` for `base`) | HF path or local dir; `trainer.teacher.model`, the model the run's teacher servers load |
| `OPD_THINKING` | `false` | `enable_thinking` for the chat template, both runs |
| `OPD_TEACHER_NUM_GPUS` / `OPD_NUM_STUDENT_GPUS` | `2` / `2` | teacher servers (`trainer.teacher.inference_engine.num_engines`, TP 1 each) and student GPUs; Ray picks the devices |
| `OPD_TEACHER_MAX_CONCURRENCY` | `256` | teacher scoring requests in flight (`trainer.teacher.max_concurrency`); each server also caps at `SKYRL_GENERATE_CONCURRENCY_PER_ENGINE` (512) |
| `OPD_MAX_STEPS` / `GRPO_MAX_STEPS` | `40` / `200` | step caps |
| `GRPO_NUM_GPUS` | `4` | GRPO's GPU count (`2` for a like-for-like `timing/step` with the OPD student) |
| `OPD_RAY_VERSION` | read from the base environment's `ray` | the cluster's Ray version; when it differs from SkyRL's pin the run scripts add `--with ray==<version>` (SkyRL's install doc); `pin` disables the override |
| `OPD_PROJECT` / `OPD_DATA` / `OPD_LOGS` | `opd_4xh100` / `~/data` / `~/logs` | W&B project, data root, logs and the last-run-name files |
| `SKYRL_DIR` | found next to (or around) the kit | the SkyRL checkout; set it if the layout differs |

Every run script forwards extra `key=value` overrides to the entrypoint, e.g.
`bash 03_run_opd.sh trainer.algorithm.opd.use_task_reward=true` for the mixed variant. Dict overrides
need a space after the colon (`"{enable_thinking: false}"`); the parser reads `{k:v}` as a key named `k:v`.

## Decisions, and what is not yet verified

- **The teacher is launched by the run, not served on the side.** Ray places SkyRL's workers from its
  own ledger and knows nothing about processes it did not launch. Two stand-alone versions of this kit
  lost to that: a driver-side `CUDA_VISIBLE_DEVICES` is ignored by a cluster that is already up (verified
  2026-09-25 against a fake 4-GPU `ray start` cluster: the student got GPU 0, under the teacher), and a
  private Ray instance per run (`RAY_ADDRESS=local`) is refused on an Anyscale node, whose exported
  `RAY_OVERRIDE_RESOURCES` pins any new raylet to all four GPUs (`Attempting to start raylet with 4 GPU,
  but CUDA_VISIBLE_DEVICES contains ['2', '3']`). Teacher servers that are SkyRL inference-server actors
  are in the ledger by construction.
- **Ray version override, not a pyproject edit.** The workspace's cluster runs its image's Ray (2.51.1)
  while SkyRL's lockfile pins 2.57.0, and a driver on another version is refused at connect
  (`Version mismatch`, 2026-09-27). SkyRL's install doc ("Running on an existing Ray cluster") says to
  override at run time: `uv run … --with ray==<cluster version>`. `00_env.sh` reads the cluster's version
  from the base environment's `ray` and the run scripts add that flag; the Ray uv hook copies every
  `uv run` option into the workers' `py_executable`, so they import the same Ray (read in the hook's
  source). SkyRL states compatibility with Ray ≥ 2.48 on this path; 2.51.1 itself is not exercised in
  SkyRL's CI, which runs the pin. The clean alternative is a workspace image whose Ray matches the pin,
  e.g. SkyRL's CI image `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0`; that needs a workspace restart.
- **Thinking off** for both runs so a 0.8B thinker does not truncate at 8k every sample and both
  methods see the same prompt format. `OPD_THINKING=true` flips it (long responses, slower steps).
- **Two eval sets.** AIME24 is the post's; 0.8B models sit near its floor, so the GSM8K slice is there
  to give the curves signal. Per-dataset keys are `eval/<data_source>/avg_score`. The runs read the kit's
  copies under `$OPD_DATA/opd-eval`, which `make_eval_set.py` rebuilds from the sources at the start of
  every run: one Arrow type convention, AIME rows in a fixed order, the first 256 GSM8K rows, then the
  same load-and-concatenate SkyRL does. Without it the run dies in `get_eval_dataset` (seen 2026-09-27):
  SkyRL's DAPO prep writes AIME through pandas 3 (`large_string`), the GSM8K script writes through
  `datasets` (`string`), and `datasets.concatenate_datasets` refuses the mix.
- **GRPO on four GPUs, OPD student on two.** Steps and rollouts are the comparison axis, not wall
  clock; set `GRPO_NUM_GPUS=2` if wall clock is the question.
- **Not verified on a GPU:** anything. What has been verified offline against
  `kyuds/opd-teacher-launching` @ `a57a03a9`: every script's generated argument list (GRPO smoke and
  full, OPD smoke and full) parses and passes `validate_cfg`, and the OPD lists pass `validate_opd_cfg`
  with the launched backend, resolving to two TP-1 text-only servers at memory fraction 0.9, prefix
  caching and Ray Prometheus stats off, context 4097 (smoke) / 10241 (run), first port 8200 (past the
  student's two windows), 256 requests in flight; sleep mode is off by the frozen-role argument builder
  (read in source, not exercised here). The 9B checkpoints and tokenizer hashes were checked on HF.
- **If a teacher server OOMs at startup**, lower `trainer.teacher.inference_engine.gpu_memory_utilization`
  (0.9 on the launched block) or `trainer.teacher.inference_engine.max_num_seqs` (1024) as extra
  overrides on the run script. Prefix caching is already off on the launched block, so vLLM 0.28's
  `prompt_logprobs` restriction does not apply.

## Files

`00_env.sh` env and knobs · `01_prepare_data.sh` data and caches · `02_run_opd_smoke.sh` ·
`03_run_opd.sh` · `04_run_grpo.sh` (`--smoke`) · `_common.sh` shared flags and the `TEACHER_OPTS` array ·
`_locate.sh` finds SkyRL · `bg.sh` detached runs (`--status`, `--stop`) · `check_opd_run.py` (`--grpo`) · `compare_runs.py` · `make_eval_set.py` the eval set.

## Next, after these runs

The token-overlap and overlap-advantage diagnostics from arXiv 2604.13016 §2.3 (design to follow; see
`research/readings/rethinking-opd-dynamics-metrics.md`) would make the OPD side mechanistic: the
paper's claim is that successful OPD is progressive alignment on the shared top-k tokens.
