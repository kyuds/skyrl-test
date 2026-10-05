# Reproducing the blog's OPD result on one spot 8×B200 node

Written 2026-10-05. The 4×H100 experiment ([kyuds/skyrl-test#1](https://github.com/kyuds/skyrl-test/pull/1))
had naive GRPO beat OPD for a Qwen3.5-0.8B student on GSM8K (best eval 0.85 against 0.72 and 0.71), and a
0.8B student may simply be too small to say anything about OPD. This kit instead reruns the exact setup
of the NovaSky post [On-Policy Distillation in SkyRL](https://novasky-ai.notion.site/on-policy-distillation)
through the new entrypoint (`skyrl.train.entrypoints.main_opd`,
[PR #2256](https://github.com/NovaSky-AI/SkyRL/pull/2256), branch `kyuds/opd-entrypoint`):
RL-train Qwen3-4B-Base with the DAPO recipe, then distil that model back into Qwen3-4B-Base and into
Qwen3-1.7B-Base.

It runs on one spot `a4-highgpu-8g` VM (8 × B200) on GCP, set up after Charlie's guide
[Running SkyRL on GCP Spot B200s](https://gist.github.com/CharlieFRuan/6eeae93d70ede0e81f12f91a4eb74d57)
cut down to a single node.

## The four runs

No 32B model is trained anywhere in the post: Qwen3-32B only appears in the snippet that shows how to
name a teacher. The results come from these four runs, all on DAPO-Math-17k with AIME 2024 as the eval.

| # | Run name | Method | Model | GPUs here | Steps | Script |
|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_4b_base` | DAPO (RL) | Qwen3-4B-Base | 8 | 90 | `02_run_dapo.sh 4b` |
| 2 | `opd_qwen3_4b_base_from_dapo4b_s90` | OPD | Qwen3-4B-Base ← run 1 | 4 student + 4 teacher | 30 | `03_run_opd.sh 4b` |
| 3 | `opd_qwen3_1p7b_base_from_dapo4b_s90` | OPD | Qwen3-1.7B-Base ← run 1 | 4 student + 4 teacher | 60 | `03_run_opd.sh 1.7b` |
| 4 | `dapo_qwen3_1p7b_base` | DAPO (RL) | Qwen3-1.7B-Base | 8 | 200 | `02_run_dapo.sh 1.7b` |

Run 1 is both the teacher and the 4B RL curve. Run 4 is only the curve that run 3 is compared against,
and it is the longest, so it goes last. Every run needs the whole node, so they go one after another;
`04_run_all.sh` does that.

What the post reports, read off its W&B report (small charts, so approximate; AIME scores are +1 / −1
per answer, so `eval/all/avg_score` runs from −1 to 1):

- Run 1 climbs from about −0.95 to about −0.5 by step 90. Run 2 reaches the same level in about 20 steps.
- Run 4 sits near −0.75 after 200 steps. Run 3 gets there in about 20 steps and keeps improving.
- `opd/reverse_kl` flattens near 0.01 for run 2 and near 0.09 for run 3.

## Where the post's recipe is ambiguous

The post's four "Full training script" bullets have no link behind them, so the settings below come from
the repo's scripts for these runs (`examples/train/algorithms/dapo/run_dapo_aime_qwen3_4b_aime.sh`,
`run_dapo_qwen3_1.7b_aime.sh`, and `examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_{4b,1.7b}.sh`
at the PR head) and from the post's W&B report.

| Question | What the sources say | What this kit does |
|---|---|---|
| Which checkpoint of run 1 is the teacher? | The post: "the resulting model". The pre-PR OPD scripts on `main` hardcode `~/ckpts/dapo_qwen3_4b_base/global_step_90/`, and the report's DAPO 4B curve ends near step 90. | Step 90 (`TEACHER_STEP`). An assumption from those two hints. |
| How long does each run go? | The scripts say `epochs=20` (about 680 steps); the post never says when a run was stopped. The report's curves end near 90 (run 1), 30 (run 2), 60–70 (run 3), 230 (run 4; the text says "~200"). | 90 / 30 / 60 / 200 (`DAPO_MAX_STEPS`, `OPD_MAX_STEPS`). Read off charts. |
| What hardware? | Not stated. The DAPO 4B script is written for 2 nodes × 8 GPUs, the other three for 1 × 8; no GPU model is named. The report's run table shows runtimes of 2 to 5 days (names truncated, so not mapped to runs). | 8 B200s for everything. Run 1 uses 8 ranks instead of 16 with the same mini-batch (32), optimizer steps per batch (16) and micro-batch (4). |
| Where does the teacher run? | In the post it is the reference-model slot: an FSDP forward on the student's own 8 GPUs. | The entrypoint under test serves it from vLLM engines on GPUs of its own: student 4, teacher 4. This is the thing being tested, not a free choice. |
| How is the OPD loss aggregated? | The post divides the sum of per-token losses by a fixed constant (`seq_mean_token_sum_norm` with `max_seq_len` = prompt + response). The entrypoint's default, `token_mean`, divides by the batch's token count, which is a different update. | The post's, passed explicitly by `03_run_opd.sh`: `loss_reduction=seq_mean_token_sum_norm`, `max_seq_len=10240` (2048 + 8192). The entrypoint's default stays `token_mean`. |
| Are today's scripts the post's scripts? | The post is from 2025-11; the repo scripts have been edited since (for example `loss_reduction=token_mean_legacy`, added to keep the old behaviour). | Today's scripts, flag for flag. |
| `enforce_eager` | On in the DAPO scripts ("due to instability with vLLM then"), off in the OPD scripts. | Kept as is. |
| LR | DAPO 1e-6 with 160 warmup optimizer steps (10 batches); OPD 1e-5, no warmup. The post says 1e-5 was unstable for DAPO. | Kept as is. |

One operational setting differs on purpose: checkpoints every 5 steps instead of 10 (`OPD_CKPT_INTERVAL`),
keeping two. It changes no result and halves what a preemption costs.

## How many B200s

Eight: one `a4-highgpu-8g`. The post's largest run was written for 16 GPUs, and 8 B200s hold more memory
than 16 80 GB cards, so nothing needs a second node. That removes the RDMA networks, multi-node NCCL and
a shared filesystem from the setup. The cost is wall clock: the four runs are sequential, and nothing
here measures how long a step takes on B200s. The post's own runs took days. Watch the first few steps
of run 1 before trusting any estimate.

## Part A. On your Mac: get the VM

These scripts drive GCP through `gcloud`. Run them from `~/dev/skyrl`.

**A0. Prerequisites (once).** You need access to the lab's GCP project (its id is in Charlie's guide) with
permission to create spot `a4-highgpu-8g` VMs in `us-west3-b`, and the `gcloud` CLI:

```bash
brew install --cask gcloud-cli
```

```bash
gcloud auth login
```

```bash
gcloud config set project <project id from Charlie's guide>
```

The project id is read from your `gcloud` config and is not written into this public repo. The scripts
assume the network `b200-vpc` (subnet `b200-vpc`) and its firewall rules for the tag `b200-train` already
exist, as the guide does; `gcp/config.sh` lists every setting and each can be overridden from the
environment (for example `GCP_VM`, default `kyuds-opd-b200`).

**A1. Push this branch.** The VM clones the kit from GitHub, on the branch this checkout is on.

**A2. Create the VM.** Spot B200 capacity comes and goes; when a request is refused for lack of capacity
the script retries every minute for up to four hours, so keep the laptop awake. Any other error (quota,
permissions, a retired image) stops it at once. It asks before it starts billing.

The image is `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`, not the guide's
`pytorch-2-7-cu128-ubuntu-2204-nvidia-570`: every image of the guide's family was deprecated by 2026-10,
so it no longer resolves. Google retires these families regularly; if this one goes too, the script
lists the current ones and `GCP_IMAGE_FAMILY=<family>` selects another.

```bash
caffeinate -i bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh
```

**A3. Set the node up.** A driver check (SkyRL's torch is a CUDA 13 build and needs driver 580, which this
image ships; an older image gets it installed) and one reboot for the open-file limit, the NVMe
array at `/mnt/local_storage` for caches, then `uv`, SkyRL on `kyuds/opd-entrypoint` with this kit inside
it (`~/SkyRL/skyrl-test`), and a warm environment. About 20 minutes on a fresh VM. The long part runs
detached on the VM, so if the connection drops, run the same command again and it re-attaches.

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/02_setup_node.sh
```

**A4. Start Ray.** Expect it to report 8 GPUs.

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/03_start_ray.sh
```

**A5. Send your keys.** Copies `WANDB_API_KEY` and `HF_TOKEN` from your Mac's shell into `~/.opd_secrets`
on the VM (mode 600) over ssh's stdin. `HF_TOKEN` must be a write token. Export both in the shell first.

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/04_push_secrets.sh
```

**A6. Log in.**

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/ssh.sh
```

From the Mac, at any time, this shows whether the VM is still there and what it is doing (GPUs, disks,
Ray, finished and uploaded runs, the tail of the newest log):

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/status.sh
```

## Part B. On the VM: run the experiment

Everything below runs in `~/SkyRL`, in bash. Long steps are started with `nohup`, so closing the ssh
session does not stop them.

**B1. Environment.** Loads the node settings and your keys, checks Ray, exports the knobs. Source it in
every new shell before anything else.

```bash
cd ~/SkyRL && source skyrl-test/opd-gcp-spot-full-repro/00_env.sh
```

**B2. Data and models.** DAPO-Math-17k and AIME 2024 through the repo's own preparation script, plus the
two base models into the HF cache.

```bash
nohup bash skyrl-test/opd-gcp-spot-full-repro/01_prepare_data.sh > ~/opd-store/logs/prepare.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/prepare.log
```

**B3. Smoke test.** Two tiny steps of DAPO, exported, then two tiny steps of OPD with that export as the
teacher. It exercises everything the real runs do except the upload: 8 FSDP ranks, the export, a teacher
served from an export, the 4 + 4 split. Do not skip it: none of this has run on a B200 yet.

```bash
nohup bash -c 'bash skyrl-test/opd-gcp-spot-full-repro/02_run_dapo.sh 4b --smoke && bash skyrl-test/opd-gcp-spot-full-repro/03_run_opd.sh 4b --smoke' > ~/opd-store/logs/smoke.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/smoke.log
```

It passed if the log ends with `=== smoke_opd_4b finished`.

**B4. The four runs, back to back.**

```bash
nohup bash skyrl-test/opd-gcp-spot-full-repro/04_run_all.sh > ~/opd-store/logs/run_all.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/run_all.log
```

Or one at a time, in this order (each waits for the node to be free; run 1 must finish before 2 and 3):

```bash
nohup bash skyrl-test/opd-gcp-spot-full-repro/02_run_dapo.sh 4b > ~/opd-store/logs/dapo_4b.log 2>&1 < /dev/null &
```

```bash
nohup bash skyrl-test/opd-gcp-spot-full-repro/03_run_opd.sh 4b > ~/opd-store/logs/opd_4b.log 2>&1 < /dev/null &
```

```bash
nohup bash skyrl-test/opd-gcp-spot-full-repro/03_run_opd.sh 1.7b > ~/opd-store/logs/opd_1p7b.log 2>&1 < /dev/null &
```

```bash
nohup bash skyrl-test/opd-gcp-spot-full-repro/02_run_dapo.sh 1.7b > ~/opd-store/logs/dapo_1p7b.log 2>&1 < /dev/null &
```

Curves are in W&B project `opd_gcp_spot_full_repro`. Engine and worker logs are under
`~/opd-store/logs/skyrl/<run>/`.

**B5. Uploads.** A run that finishes successfully uploads its final HF export to
`kyuds/opd-gcp-spot-full-repro-<run name>-step<N>` as a public repo, and records the URL in
`~/opd-store/logs/<run>.uploaded`. If the upload fails the run still counts as finished (the script exits
with status 3 and says so); running the same command again retries only the upload. By hand:

```bash
bash skyrl-test/opd-gcp-spot-full-repro/upload_export.sh --run dapo_qwen3_4b_base --dry-run
```

`--step N` uploads another export, `--repo` names the repo, `--private` creates the repo private, and
`OPD_UPLOAD=0` (set before sourcing `00_env.sh`) turns the automatic upload off.

## After a preemption

A preempted spot VM is stopped, not deleted. Its boot disk survives, and with it `~/opd-store` (data,
checkpoints, exports, logs, markers), the checkouts, the base environment and your keys. The NVMe array
is wiped, so the caches go. From the Mac:

```bash
caffeinate -i bash skyrl-test/opd-gcp-spot-full-repro/gcp/01_create_vm.sh
```

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/02_setup_node.sh
```

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/03_start_ray.sh
```

Then on the VM: B1, B2 (the data is still there; the models download again), and the same B4 command as
before. Run names are fixed, so a run resumes from its last checkpoint, and `04_run_all.sh` skips the
stages that already finished. At most 5 steps are lost. A resumed run may show up as a second W&B run
with the same name; that was not checked.

## Shutting down

Stop the VM and keep its disk (you pay for the disk only; A2 brings it back):

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/down.sh stop
```

Delete the VM and its disk, which removes every checkpoint and export not uploaded (asks for the VM name):

```bash
bash skyrl-test/opd-gcp-spot-full-repro/gcp/down.sh delete
```

## Knobs

All are environment variables. The first group is read by `00_env.sh`, so set them before sourcing it.

| Variable | Default | Meaning |
|---|---|---|
| `OPD_UPLOAD` | `1` | Upload each finished run's final export. `0` needs no `HF_TOKEN`. |
| `HF_USER` | `kyuds` | Hub namespace for the uploads. |
| `OPD_STORE` | `~/opd-store` | Data, checkpoints, exports, logs. On the boot disk. |
| `OPD_DAPO_NUM_GPUS` | `8` | GPUs of a DAPO run. |
| `OPD_NUM_STUDENT_GPUS`, `OPD_TEACHER_NUM_GPUS` | `4`, `4` | The OPD split. They must add up to at most 8. |
| `DAPO_MAX_STEPS` | 90 (4b), 200 (1.7b) | Where a DAPO run stops. |
| `OPD_MAX_STEPS` | 30 (4b), 60 (1.7b) | Where an OPD run stops. |
| `TEACHER_RUN`, `TEACHER_STEP` | `dapo_qwen3_4b_base`, `90` | Which export is the teacher. |
| `OPD_TEACHER_MODEL` | unset | A path or Hub id instead of that export. A private Hub repo also needs `OPD_FORWARD_HF_TOKEN=1`. |
| `OPD_TAG` | unset | Suffix for the run name: a fresh run of the same kind. It does not change the teacher. |
| `OPD_CKPT_INTERVAL` | `5` | Steps between checkpoints. |
| `OPD_CKPT_ROOT` | `$OPD_STORE/ckpts/<project>` | May be a `gs://` path; SkyRL then writes checkpoints to GCS, which survives losing the VM. |
| `OPD_STAGES` | `dapo:4b opd:4b opd:1.7b dapo:1.7b` | What `04_run_all.sh` runs, in order. |

Anything after the size on a run script's command line is passed to SkyRL as extra overrides.

## Things worth knowing

- **The OPD runs set the post's loss aggregation explicitly.** `03_run_opd.sh` passes
  `trainer.algorithm.loss_reduction=seq_mean_token_sum_norm` and `trainer.algorithm.max_seq_len=10240`
  (prompt 2048 + response 8192; 3072 in the smoke run), as the PR's two math example scripts do. The
  post's version divides the batch's summed token losses by a constant; the entrypoint's default,
  `token_mean`, divides by the batch's own token count, which rescales every update by that batch's
  mean response length. To run the default instead, append `trainer.algorithm.loss_reduction=token_mean`
  to the command (with `OPD_TAG` for a separate run name).
- **`HF_TOKEN` is kept out of the training process.** SkyRL copies `HF_TOKEN` into the Ray runtime
  environment and logs its value while doing so (`prepare_runtime_environment` in
  `skyrl/train/utils/utils.py`), so a run started with the token set writes it into its own log. The run
  scripts remove it from the training command's environment; only the upload sees it.
- **NCCL uses plain sockets** (`NCCL_NET=Socket`, `NCCL_NET_PLUGIN=none` in `~/.opd_cluster_env`), as the
  guide prescribes for a single node. Ray workers inherit them from the raylet, which is why
  `03_start_ray.sh` sets them before `ray start`.
- **Ray's version is the lockfile's.** The cluster is started from SkyRL's base environment
  (`~/venvs/skyrl`), so the runs need no `--with ray==...` override, unlike the Anyscale node.
- **The boot disk is 1000 GB**, twice the guide's, because it holds the checkpoints and exports of four
  runs. It is slower than the NVMe array; a 4B checkpoint will take minutes to write.

## Files

On the VM: `00_env.sh` (source first), `01_prepare_data.sh`, `02_run_dapo.sh`, `03_run_opd.sh`,
`04_run_all.sh`, `upload_export.sh`, with `_common.sh` (the task and batch shape both methods share,
the upload hook, the done markers) and `_locate.sh` (finds the SkyRL checkout).

On your Mac, in `gcp/`: `config.sh` (settings), `01_create_vm.sh`, `02_setup_node.sh`, `03_start_ray.sh`,
`04_push_secrets.sh`, `ssh.sh`, `status.sh`, `down.sh`. `node_setup.sh` is the part that runs on the VM
and `vm_startup.sh` is the VM's boot script; the others copy and call them.

## Verified and assumed

Verified on 2026-10-05, from the Mac, without a GPU or a VM:

- Every run script's flags parse into the entrypoint's config and pass `validate_cfg` (and
  `validate_opd_cfg` for OPD) at PR head `e1a51156`, for the full runs and the smoke runs.
- The sequence logic with a stand-in for `uv`: finished stages are skipped, a failed upload is retried
  without retraining, a failed run stops the sequence, the token is absent from the training command.
- The Mac-side `gcp/` scripts against a stand-in for `gcloud`: create with retries, the unpushed-branch
  guard, driver + reboot + storage + polled software phase, Ray start, the key push (values arrive
  intact and never appear on a command line), status, stop.
- Read-only queries against the project (2026-10-05): the image family resolves, the subnet `b200-vpc`
  exists in `us-west3`, `a4-highgpu-8g` is offered in `us-west3-b` with 8 B200s, and a firewall rule on
  `b200-vpc` allows ssh from outside.

Not verified:

- Creating the VM and everything after it. The first attempt, on 2026-10-05, stopped at the retired image
  family, which is what led to the current default.
- `gcp/node_setup.sh`, the part that runs on the VM (driver install, NVMe array, `uv`, checkouts, Ray). It
  needs Linux and has only been syntax-checked. Expect to fix something in it on the first real node.
- That this stack runs on B200s at all (the smoke test is the first check), and how long a step takes.
- That the node setup works on the `pytorch-2-9` image. The guide's steps were written for the retired
  `pytorch-2-7` image; this one differs at least in shipping driver 580 already.
- That an `a4-highgpu-8g` accepts a single network interface. The guide attaches ten for multi-node RDMA;
  this kit attaches one.
- The step counts and the teacher step, which are read off the post's charts and one old script.
