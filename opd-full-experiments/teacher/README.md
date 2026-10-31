# Teacher training

Trains the math teachers for the OPD experiments, one model per node, all at the same time. One script on your Mac
starts everything, and another shows what has been preempted. Written 2026-10-31.

| Node | GPUs | Base model | Run name, also the W&B name |
|---|---|---|---|
| `h100-1` | 8 × H100 | `Qwen/Qwen3-1.7B-Base` | `dapo_nemotron_math_qwen3_1p7b_base` |
| `h100-2` | 8 × H100 | `Qwen/Qwen3-1.7B` | `dapo_nemotron_math_qwen3_1p7b` |
| `h100-3` | 8 × H100 | `Qwen/Qwen3-4B-Base` | `dapo_nemotron_math_qwen3_4b_base` |
| `h100-4` | 8 × H100 | `Qwen/Qwen3-4B` | `dapo_nemotron_math_qwen3_4b` |
| `b200-1` | 8 × B200 | `Qwen/Qwen3-8B-Base` | `dapo_nemotron_math_qwen3_8b_base` |
| `b200-2` | 8 × B200 | `Qwen/Qwen3-8B` | `dapo_nemotron_math_qwen3_8b` |

The first run doubles as the student's GRPO baseline. To change the list, edit `TEACHER_RUNS` in `config.sh`.

The code teacher is left out for now: SkyRL's `lcb` environment runs model-written code with `exec` inside the training
process, on nodes that hold your Hub write token, so it needs a real sandbox.

## The recipe

| | |
|---|---|
| Method | SkyRL's DAPO recipe, as in `opd-gcp-spot-full-repro`: GRPO advantages, dual-clip loss, overlong filtering and punishment, LR 1e-6 |
| Data | Nemotron-Cascade-RL-Math, 14,461 of its 14,476 problems |
| Steps | 1,000, about 9 passes over the data |
| Batch | 128 prompts × 8 rollouts, 2 optimizer steps per training step |
| Length | 2,048 prompt tokens, 8,192 response tokens |
| Thinking | Off, on Base and instruct models alike |
| Eval | AIME 2024 and 2025, 32 samples each, every 25 steps |
| Checkpoints | Every 10 steps, last 2 kept, on the boot disk |
| HF exports | Every 100 steps and at each epoch end, on the boot disk |

**Expect about 2 to 3.5 days per run, plus preemptions.** The 1.7B runs should finish first. That is an estimate from
earlier runs at a larger batch, so compare it with the first steps' `ETA` in `status.sh`.

## Results

- **W&B:** project `opd_full_experiments`, named as in the table and tagged `teacher`, `dapo`,
  `nemotron-cascade-rl-math` and the base model. Smoke tests go to `opd_full_experiments_smoke`.
- **Hugging Face:** `kyuds/opd_full_experiments/<run>_step1000`, public. A Hub repo id cannot have a second slash, so
  this is one repo, `kyuds/opd_full_experiments`, with one folder per model.

vLLM loads a repo or a local path, not a folder inside a repo, so download a model before using it as a teacher. From
`~/SkyRL` on a VM:

```bash
uv run --isolated --extra fsdp hf download kyuds/opd_full_experiments --include 'dapo_nemotron_math_qwen3_8b_base_step1000/*' --local-dir ~/opd-store/teachers
```

## Start it

Run these from `~/dev/skyrl`, on your Mac.

**1. Push the kit.** The VMs read it from GitHub, and `launch_all.sh` refuses to start if your copy differs.

**2. Put your keys in the shell, and refresh the gcloud login.** `HF_TOKEN` must be able to write to `kyuds`.

```bash
source ~/.zprofile
```

```bash
gcloud auth login
```

**3. Check the plan.** This changes nothing:

```bash
bash skyrl-test/opd-full-experiments/teacher/launch_all.sh
```

**4. Smoke test every node.** This sets each node up and runs 2 tiny steps with the real recipe, so a broken node shows
up now and not in three days. It takes about 30 to 40 minutes. Plug the Mac in and keep the lid open until it ends.

```bash
nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --smoke --yes > ~/opd_teachers.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd_teachers.log
```

The last lines list each node's result. Fix any failure before going on.

**5. Start the real runs.** This takes a few minutes now that the nodes are set up.

```bash
nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --yes > ~/opd_teachers.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd_teachers.log
```

Once it prints the result, the runs live on the VMs under `nohup`, so the Mac can sleep.

## Check on it

```bash
bash skyrl-test/opd-full-experiments/teacher/status.sh
```

`PREEMPTED` counts every preemption of that VM, even after it came back. The end of a run's log:

```bash
bash skyrl-test/opd-full-experiments/teacher/status.sh h100-1
```

| State | Meaning | What to do |
|---|---|---|
| `TRAINING` | Going normally | Nothing |
| `PREEMPTED` | The VM was taken back | Run `launch_all.sh --yes` |
| `STOPPED` | The VM was stopped, not preempted | Run `launch_all.sh --yes` if you want it back |
| `RUN-DIED` | The VM is up but the run stopped | Read the log, then run `launch_all.sh --yes` |
| `STALLED` | No log output for 30 minutes | Read the log |
| `NO-ANSWER` | The VM is up but ssh failed | Try again, then read the log |
| `NOT-STARTED` | The run never started | Run `launch_all.sh --yes` |
| `UPLOADING` | Trained, uploading | Nothing |
| `UPLOAD-FAILED` | Trained, but not uploaded. The export is safe | Run `launch_all.sh --yes`, which retries only the upload |
| `DONE` | Trained and uploaded | Stop the node |

Each run's W&B page is another way to see whether it is alive.

## After a preemption

Run the same command again, from any state:

```bash
nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --yes > ~/opd_teachers.log 2>&1 < /dev/null &
```

It starts the stopped VM, rebuilds the node's caches and resumes from the last checkpoint, which loses at most about 30
minutes. Active and finished runs are left alone. Nothing does this on its own, so check `status.sh` a few times a day.

If a VM will not start, its zone has no spot capacity right now. This keeps asking for up to 4 hours, then run
`launch_all.sh` again. Its `*_NODES` values matter: it starts every node up to those counts, so lower them if you have
stopped nodes on purpose.

```bash
nohup caffeinate -i bash skyrl-test/opd-full-experiments/gcp/01_create_nodes.sh --yes > ~/opd_exp_nodes.log 2>&1 < /dev/null &
```

A resumed run shows up on W&B as a second run with the same name, continuing at the step it resumed from.

## When a run is done

Each node bills until you stop it, so stop it once `status.sh` says `DONE`. The boot disk stays, with its checkpoints
and every export:

```bash
bash skyrl-test/opd-full-experiments/gcp/down.sh stop h100-1
```

Only the final export is uploaded. Any other one, from the VM:

```bash
bash skyrl-test/opd-full-experiments/teacher/upload_export.sh --run dapo_nemotron_math_qwen3_8b_base --step 500
```

## Knobs

Edit `config.sh` and push: the VMs read it from GitHub, so a variable set on your Mac does not reach them. It holds the
run table, steps, batch shape and intervals. Changing the batch shape or steps of a run in progress mixes two recipes
in one run, so make those changes before step 5.

`launch_all.sh --only h100-1,b200-2` limits it to those nodes, with or without `--smoke`.

The VMs check out one pinned SkyRL commit, `SKYRL_COMMIT`, the head of `kyuds/opd-entrypoint` on 2026-10-31, so a
resume uses the same code as the start even if the branch moves. Pushing to the kit branch does reach a node when it
is next set up.

## Verified and assumed

Verified on 2026-10-31, on the Mac:

- The data conversion on the real dataset: 14,461 rows, same columns and prompt template as the AIME eval. 15 rows were
  dropped because the grader cannot match their answers, which contain `<`. SkyRL drops 7 more when it loads the file,
  for prompts over 2,048 tokens.
- The exact overrides for every model and for the smoke test, through SkyRL's real `DAPOConfig` and `validate_cfg`.
- `launch_all.sh` and `status.sh` end to end on the Mac's bash 3.2, against a stand-in for `gcloud` that ran the real
  run, upload and report scripts and the real `prepare_dapo_data.sh`. Covered: fresh launch, rerun, preemption, no
  capacity, a crashed run, a failed or conflicting upload, smoke pass and failure, preemption during setup, an expired
  login, and `nohup` surviving a hangup. No key reached a command line or a log.
- On real nodes, read-only: 8 GPUs with driver 580 and working CUDA on an H100 and a B200, `nvidia-fabricmanager` failing
  harmlessly with the GPU fabric reporting Success, the expected NVMe drives, and 1,000 GB boot disks. `status.sh`, the
  plan and the preemption query against real `gcloud`, reaching all six nodes, and the `report` phase on real Ubuntu.
  GitHub serves the pinned commit, and the Hub accepts `HF_TOKEN`.
- The 8B models share the student's vocabulary and special-token IDs.

Not verified:

- Nothing has trained on a real node yet, which is what the smoke test is for. It proves the setup, but not 8K-token
  memory use.
- **Memory:** the settings are the earlier 4B runs'. That is untested for 8B, and for thinking-off generation at these
  lengths. If a run dies of memory, lower `micro_train_batch_size_per_gpu`.
- **Run time:** an estimate, as above. 1,000 steps and 128 × 8 are a choice, not a tested recipe.
- **Teacher strength:** 1,000 steps may overshoot the best step. Eval curves on W&B will show it, and every export
  stays on the boot disk until the VM is deleted.
- **Upload path:** the `hf upload` call has not reached the real Hub, and a failure only costs a retry.
- **Non-integer answers:** 18% of the data is graded by string match, so some correct rollouts on those problems score
  as wrong. **Assumed** to be noise and not harmful.

## Files

On your Mac: `launch_all.sh`, `status.sh`. On the VM: `node_setup.sh`, `00_env.sh`, `01_prepare_data.sh`,
`prepare_nemotron_math.py`, `02_train_teacher.sh`, `upload_export.sh`. On both: `config.sh`. Your Mac keeps per-node logs
in `~/opd-full-experiments-teacher/logs/`, and each VM keeps run logs in `~/opd-store/logs/`.
