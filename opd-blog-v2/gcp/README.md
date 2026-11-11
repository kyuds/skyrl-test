# GCP nodes for opd-blog-v2

Get the spot GPU nodes, set them up, and stop paying when the jobs are done. Every command here runs on your Mac,
from `~/dev/skyrl`. Starting the jobs is `../orchestrate.sh` (`../README.md`, "Running it").

## What it asks for

Spot nodes, in every zone of every region that offers them, all requested in parallel:

| Kind | Machine | GPUs | At most | Names | Used for |
|---|---|---|---|---|---|
| B200 | `a4-highgpu-8g` | 8 × B200 | 2 nodes | `kyuds-opd-v2-b200-1`, `-2` | 1a's SFT, its eval, then 1b; Qwen3-32B's eval and answers, 2c's warm-up, then 2c |
| H100 | `a3-highgpu-8g` | 8 × H100 | 7 nodes | `kyuds-opd-v2-h100-1` to `-7` | `-1` to `-4`: the M2RL and Qwen3-8B evals, then 2b, 2a, 2d and MOPD; `-5` to `-7`: the teacher servers (`../teachers/`) |

Which job runs on which node is the job table in `../config.sh` (`V2_JOBS`). With fewer H100s, set `V2_NODE_MAP`
there (`../README.md`, "Fewer nodes"); `GCP_H100_NODES=3` in front of the commands below asks for 3. Each node is
independent; they do not form a cluster, so every job fits on one node.

Boot disks are 2,000 GB. The largest user is `b200-1`: 1a's checkpoint (about 100 GB) and three exports (16 GB each),
then 1b's two checkpoints and exports, plus the pretokenized SFT data (about 20 GB).

These are new VMs. The six `kyuds-opd-exp-*` VMs from `opd-full-experiments` are still there, stopped, and billing for
their disks; that kit's `gcp/down.sh delete` removes them.

## Before you start

The login lasts about a day, so refresh it first, or it can expire while nodes are being requested:

```bash
gcloud auth login
```

Check the plan. This creates nothing:

```bash
bash skyrl-test/opd-blog-v2/gcp/01_create_nodes.sh
```

## Get the nodes

**Billing starts as soon as a node is up, even while the script waits for the others.** Plug the Mac in and keep the
lid open: `caffeinate` does not stop a closed lid from sleeping the Mac.

```bash
nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/01_create_nodes.sh --yes > ~/opd_v2_nodes.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd_v2_nodes.log
```

It ends when it holds every node, or after 4 hours. A progress line appears every 5 minutes. To stop early, keeping
what it has:

```bash
touch ~/opd-blog-v2-gcp/stop
```

Running it again picks up where it left off: nodes that exist are kept, and stopped ones are started again. Refusals
are handled as in the last kit: no capacity moves on to the next zone, a quota refusal leaves that region alone for 30
minutes, and anything unexpected is printed.

## Set the nodes up

This installs everything on every node with a job, at the same time: driver check, the NVMe array, SkyRL at the pinned
commit with this kit inside it, your W&B and Hugging Face keys, and Ray. It starts no job.

The VMs clone the kit from GitHub, so commit and push it first; the script refuses otherwise. Your keys must be in the
shell, which they are after:

```bash
source ~/.zprofile
```

Check the plan:

```bash
bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh
```

Set up:

```bash
nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/02_setup_nodes.sh --yes > ~/opd_v2_setup.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd_v2_setup.log
```

It takes 15 to 25 minutes on a fresh node, mostly the software phase, and ends with one line per node: `ready` or
`FAILED` with a log path. Without `--only` it sets up the nodes with a job in `config.sh`; `--only b200-1,h100-3`
sets up exactly those, with a job or not. It may run while `01_create_nodes.sh` is still asking for the others, so a
node can start its jobs while the rest arrive. Only one copy runs at a time. Keys go over ssh's stdin into
`~/.opd_secrets` on the VM, mode 600; they are never on a command line, printed, or in this repo.

## Check on the nodes

The jobs' states are `../orchestrate.sh` (`../README.md`). For the nodes themselves, liveness and the GPUs each running
node holds:

```bash
bash skyrl-test/opd-blog-v2/gcp/status.sh alive
```

Log into a node:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh b200-1
```

## After a preemption

A preempted spot VM is stopped, not deleted. Its boot disk keeps the data, checkpoints, exports, models, logs and keys.
The NVMe array is wiped, so the HF cache goes with it, and Ray is down.

1. `orchestrate.sh` shows the node's jobs as `node-terminated` (or `node-stopped`).
2. Run `02_setup_nodes.sh --yes` again, with `--only <node>`. It starts the VM in its zone, rebuilds the array and
   starts Ray. A stopped VM can only restart where it is, so with no capacity there it fails; try again later.
3. Run the same stage again with `orchestrate.sh run <stage> --only <node> --yes`. Done jobs are skipped, the
   interrupted one resumes from its last checkpoint (every 10 OPD steps, every 250 SFT steps; an eval keeps the GPUs
   that finished), and the models download again.

**When its zone has no capacity** (step 2 fails with STOCKOUT), move the node instead. `migrate.sh` snapshots the
stopped node's boot disk, deletes the VM and the disk once the snapshot is READY, and creates the node again under the
same name in any zone with room, its boot disk made from the snapshot, so every checkpoint, export, job record and
log comes along. It checks the files arrived, then deletes the snapshot. Interrupted, it resumes where it was when run
again. It refuses while `01_create_nodes.sh` runs (that would bring the old node back) and refuses a running node.
The snapshot costs a little for the time it exists; a disk made from it in another region is billed as network
egress, roughly $0.01 to $0.08 per GB. The plan first:

```bash
bash skyrl-test/opd-blog-v2/gcp/migrate.sh h100-3
```

Then (it asks for capacity for up to 4 hours; `touch ~/opd-blog-v2-gcp/stop` ends it early, keeping the snapshot):

```bash
nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/migrate.sh h100-3 --yes > ~/opd_v2_migrate.log 2>&1 < /dev/null &
```

Then steps 2 and 3 above, `02_setup_nodes.sh --only <node> --yes` and the same stage again. Tested on 2026-11-10
against a stand-in gcloud (the normal path, every interruption point, no capacity anywhere); not yet on a real node.

## Stop paying

Stops every running node. Boot disks are kept, with every export on them, and you pay for the disks only:

```bash
bash skyrl-test/opd-blog-v2/gcp/down.sh stop all
```

Deletes every node and its boot disk. It asks first, so run it in a terminal, not under `nohup`. Upload the exports you
want before this:

```bash
bash skyrl-test/opd-blog-v2/gcp/down.sh delete all
```

Either also takes one node, such as `h100-1`.

## Knobs

Set in the environment. `config.sh` in this folder lists them all.

| Variable | Default | Meaning |
|---|---|---|
| `GCP_H100_NODES`, `GCP_B200_NODES` | 7, 2 | Most nodes of each kind |
| `GCP_H100_ZONES`, `GCP_B200_ZONES` | empty | Empty means every zone that offers the machine |
| `GCP_TIMEOUT_SECS` | 14400 | When `01_create_nodes.sh` stops asking |
| `GCP_BOOT_DISK_GB` | 2000 | Boot disk size |

State lives in `~/opd-blog-v2-gcp/`, outside this public repo, because gcloud's messages can carry the project id.

## Verified and assumed

Verified before 2026-11-09, with the previous job layout: `01_create_nodes.sh` got all six nodes, `02_setup_nodes.sh`
set them up, and all five smoke tests of that layout passed on them. `node_setup.sh` carries the fix for a
half-stopped Ray cluster (tested on a B200 on 2026-11-02) and the fix for its `stop-run` and `report` messages.

Changed on 2026-11-09 and not yet run on a node: `02_setup_nodes.sh` reads its nodes from the job table instead of
the old run list.

## Files

`config.sh` (settings and helpers), `01_create_nodes.sh`, `02_setup_nodes.sh`, `node_setup.sh` (runs on the VM),
`status.sh` (node liveness), `ssh.sh`, `down.sh`, `vm_startup.sh` (open-file limits at boot).
