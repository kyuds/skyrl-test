# Teacher training

Trains the math teachers for the OPD experiments, one model per node, all at the same time. One script on your Mac
starts everything, and another shows what has been preempted. Written 2026-10-31. The recipe changed on 2026-11-01
after the first launch failed: see [First launch](#first-launch-2026-11-01).

| Node | GPUs | Base model | Run name, also the W&B name |
|---|---|---|---|
| `h100-1` | 8 × H100 | `Qwen/Qwen3-1.7B-Base` | `dapo_dapo17k_qwen3_1p7b_base` |
| `h100-2` | 8 × H100 | `Qwen/Qwen3-1.7B` | `dapo_dapo17k_qwen3_1p7b` |
| `h100-3` | 8 × H100 | `Qwen/Qwen3-4B-Base` | `dapo_dapo17k_qwen3_4b_base` |
| `h100-4` | 8 × H100 | `Qwen/Qwen3-4B` | `dapo_dapo17k_qwen3_4b` |
| `b200-1` | 8 × B200 | `Qwen/Qwen3-8B-Base` | `dapo_dapo17k_qwen3_8b_base` |
| `b200-2` | 8 × B200 | `Qwen/Qwen3-8B` | `dapo_dapo17k_qwen3_8b` |

The first run doubles as the student's GRPO baseline. To change the list, edit `TEACHER_RUNS` in `config.sh`.

The code teacher is left out for now: SkyRL's `lcb` environment runs model-written code with `exec` inside the training
process, on nodes that hold your Hub write token, so it needs a real sandbox.

## The recipe

| | |
|---|---|
| Method | SkyRL's DAPO recipe, as in `opd-gcp-spot-full-repro`: GRPO advantages, dual-clip loss, overlong filtering and punishment, LR 1e-6 with 160 warmup optimizer steps |
| Data | DAPO-Math-17k, 17,391 problems |
| Steps | 1,000, about 7 passes over the data |
| Batch | 128 prompts × 8 rollouts, 8 optimizer steps per training step on 128 rollouts each |
| Length | 2,048 prompt tokens, 8,192 response tokens |
| Prompt | Base: the template's plain prompt. Instruct: the empty think block, which turns thinking off |
| Eval | AIME 2024 and 2025, 32 samples each, every 25 steps |
| Checkpoints | Every 10 steps, last 2 kept, on the boot disk |
| HF exports | At each epoch end, about every 135 steps, and at the last step, on the boot disk |

**Expect about 2 to 3 days per run, plus preemptions.** The 1.7B runs should finish first. On the first launch the
healthy runs took 150 to 240 seconds per step, and steps slow down as answers get longer. **Assumed:** 8 optimizer steps
instead of 2 add little to that. Compare it with the first steps' `ETA` in `status.sh`.

## Results

- **W&B:** project `opd_full_experiments`, named as in the table and tagged `teacher`, `dapo`,
  `dapo-math-17k` and the base model. Smoke tests go to `opd_full_experiments_smoke`.
- **Hugging Face:** `kyuds/opd_full_experiments/<run>_step1000`, public. A Hub repo id cannot have a second slash, so
  this is one repo, `kyuds/opd_full_experiments`, with one folder per model. Do not use the two
  `dapo_nemotron_math_*_step1000` folders in it: they are the failed first launch.

vLLM loads a repo or a local path, not a folder inside a repo, so download a model before using it as a teacher. From
`~/SkyRL` on a VM:

```bash
uv run --isolated --extra fsdp hf download kyuds/opd_full_experiments --include 'dapo_dapo17k_qwen3_8b_base_step1000/*' --local-dir ~/opd-store/teachers
```

## First launch, 2026-11-01

The first recipe used Nemotron-Cascade-RL-Math, 2 optimizer steps per training step, warmup 20, and the empty think
block on every model. `dapo_nemotron_math_qwen3_1p7b_base` and `dapo_nemotron_math_qwen3_4b_base` reached step 1,000 in
10 hours and are worthless. By about step 30 they answered every problem with a bare guess like `Answer: 5`, and AIME
stayed at -1.000. Train reward still rose, to about -0.55, on problems they had seen about 9 times.

A flat wrong guess scores -1, and those two started at -1.016 and -1.042. Every healthy run started above -1, including
8B-Base at -0.932 and the earlier Base runs at -0.970 and -0.924. A rollout cut off at 8,192 tokens scores -2, stays in
its group's mean and std, and has its own loss masked. So in a group of wrong answers, every rollout that finished gets
a positive advantage, and a bare guess always finishes.

**Most likely cause:** the think block on Base models. On the same AIME 2024 eval at step 0, it raised 4B-Base's mean
length from 2,149 to 4,063 tokens and 1.7B-Base's from 2,377 to 3,167. This is **not proven**: the data and update rate
changed in the same step, so the new recipe cannot separate them.

All six runs restarted, so the teachers differ only in the model. The four that were healthy stopped at about step
230 to 370. Their checkpoints and exports stay on disk under the old names.

## Restart on this recipe

Use this once, to move the six nodes off the first launch. From `~/dev/skyrl`, after pushing the kit:

```bash
source ~/.zprofile
```

```bash
gcloud auth login
```

```bash
bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --restart
```

```bash
nohup caffeinate -i bash skyrl-test/opd-full-experiments/teacher/launch_all.sh --restart --yes > ~/opd_teachers.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd_teachers.log
```

It stops each first-launch run and starts the new one. It takes a few minutes per node, and the Mac can sleep once it
prints its results. Running it again leaves the new runs alone.

Check each new run on W&B, because a collapse shows within about 2 hours:

- `reward/avg_raw_reward` at step 1 is clearly above -1.
- `generate/avg_num_tokens` stays above several hundred past step 40.
- `eval/all/avg_score` rises, and is not stuck at -1.

## Start it on new nodes

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
| `DONE` | Reached step 1,000 and uploaded. This says nothing about quality | Check its eval on W&B, then stop the node |

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

Each node bills until you stop it, so check its eval and stop it once `status.sh` says `DONE`. The boot disk stays, with its checkpoints
and every export:

```bash
bash skyrl-test/opd-full-experiments/gcp/down.sh stop h100-1
```

Only the final export is uploaded. Any other one, from the VM:

```bash
bash skyrl-test/opd-full-experiments/teacher/upload_export.sh --run dapo_dapo17k_qwen3_8b_base --step 405
```

## Knobs

Edit `config.sh` and push: the VMs read it from GitHub, so a variable set on your Mac does not reach them. It holds the
run table, steps, batch shape and intervals. Changing the batch shape or steps of a run in progress mixes two recipes
in one run. For a new recipe, change `TEACHER_DATA` too, which gives every run a new name, and use `--restart`. Under
the old names a finished run would be skipped and an active one left alone.

`launch_all.sh --only h100-1,b200-2` limits it to those nodes, with `--smoke`, `--restart` or neither.

Disk: an 8B export is 31 GB and a checkpoint 92 GB, with three on disk while one saves. A run needs about 525 GB on a
B200 and the first launch's files stay, which leaves about 90 GB free. Watch `DISK_FREE` in `status.sh`.

The VMs check out one pinned SkyRL commit, `SKYRL_COMMIT`, the head of `kyuds/opd-entrypoint` on 2026-10-31, so a
resume uses the same code as the start even if the branch moves. Pushing to the kit branch does reach a node when it
is next set up.

## Verified and assumed

Verified on 2026-11-01:

- The exact overrides for all six models and the smoke test, through SkyRL's real `DAPOConfig` and `validate_cfg`.
- `launch_all.sh` and `status.sh` end to end on the Mac's bash 3.2, against a stand-in for `gcloud` that ran the real
  run, upload and report scripts and the real `prepare_dapo_data.sh`: 105 scenarios, including `--restart`, a run that
  will not stop, preemption, a crash, failed uploads and smoke tests. Breaking the Base prompt, the data paths, the
  restart check or the stop check on purpose each made them fail.
- All six nodes hold the same 17,391 DAPO-Math-17k rows, and the plan with `--restart` reaches them through real `gcloud`.
- `stop-run`'s patterns match exactly the three run processes on each active node, and the GPU check reads 16 processes
  on an active node and 0 on an idle one. No real run has been stopped with it.
- The real Base template: the plain prompt ends in `<|im_start|>assistant\n`, with no think block.
- On the first launch, on real nodes: node setup, training, W&B logging and `hf upload` to the real Hub. The 8B settings
  held for about 250 steps without running out of memory.
- Between `b4b10f48`, the branch head when `73ca8v4j` ran, and the pinned commit: the penalty's arithmetic, GRPO, the
  dual-clip loss and the AIME environment match. The inference-server changes were not audited.

Verified on 2026-10-31: 8 GPUs with driver 580 and working CUDA on an H100 and a B200, `nvidia-fabricmanager` failing
harmlessly with the GPU fabric reporting Success, the expected NVMe drives, 1,000 GB boot disks, and the 8B models sharing
the student's vocabulary and special-token IDs.

Not verified:

- **The cause:** the think block is the main suspect, but data and update rate changed with it.
- **This recipe on a real node:** nothing has trained on it yet. Check the first hours as in
  [Restart on this recipe](#restart-on-this-recipe).
- **Gradient noise:** each optimizer step sees 128 rollouts, against 512 in the earlier runs and on the first launch.
- **Memory late in a run:** answers get longer as training goes on. If a run dies of memory, lower
  `micro_train_batch_size_per_gpu`.
- **Run time:** an estimate, as above. 1,000 steps and 128 × 8 are a choice, not a tested recipe.
- **Teacher strength:** 1,000 steps is 8,000 optimizer steps, against 1,440 for `73ca8v4j` at its step 90, so a run may
  overshoot its best step. Eval curves on W&B will show it, and every export stays on the boot disk.
- **Cross-lineage OPD:** the teacher scores the student's exact prompt (`SkyRL/skyrl/train/opd/trainer.py:72`). A Base
  student's plain prompt puts an instruct teacher in thinking mode, so Base-student runs with instruct teachers need a
  decision first.

## Files

On your Mac: `launch_all.sh`, `status.sh`. On the VM: `node_setup.sh`, `00_env.sh`, `01_prepare_data.sh`,
`02_train_teacher.sh`, `upload_export.sh`. On both: `config.sh`. Your Mac keeps per-node logs
in `~/opd-full-experiments-teacher/logs/`, and each VM keeps run logs in `~/opd-store/logs/`.
