# GCP nodes for opd-blog-v2

Get four spot GPU nodes, set them up, and stop paying when the runs are done. Every command here runs on your Mac,
from `~/dev/skyrl`. Starting the runs is in each experiment's README.

## What it asks for

Spot nodes, in every zone of every region that offers them, all requested in parallel:

| Kind | Machine | GPUs | At most | Names | Runs on it |
|---|---|---|---|---|---|
| B200 | `a4-highgpu-8g` | 8 × B200 | 2 nodes | `kyuds-opd-v2-b200-1`, `-2` | GRPO teacher then phase 1.2; phase 2 |
| H100 | `a3-highgpu-8g` | 8 × H100 | 2 nodes | `kyuds-opd-v2-h100-1`, `-2` | phase 1.1; phase 3 |

The run of each node is set in `../config.sh` (`V2_RUNS`). The B200s take the runs with the most memory: the 8B
models, and phase 2's 32B teacher. Each node is independent; they do not form a cluster.

Boot disks are 2,000 GB, not the 1,000 GB of the last kit. `b200-1` holds the GRPO run's exports and checkpoints
(about 540 GB) and then phase 1.2's (about 470 GB).

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

This installs everything on every node with a run, at the same time: driver check, the NVMe array, SkyRL at the pinned
commit with this kit inside it, your W&B and Hugging Face keys, and Ray. It starts no run.

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
`FAILED` with a log path. `--only b200-1,h100-2` limits it to those nodes. Keys go over ssh's stdin into
`~/.opd_secrets` on the VM, mode 600; they are never on a command line, printed, or in this repo.

## Check on the runs

One line per run: node state, preemptions, step, ETA, log age, GPU use and free disk.

```bash
bash skyrl-test/opd-blog-v2/status.sh
```

The same, plus the end of one run's log:

```bash
bash skyrl-test/opd-blog-v2/status.sh p1_opd_m2rl_sft_from_rl_math
```

Node-level liveness, and the GPUs each running node holds:

```bash
bash skyrl-test/opd-blog-v2/gcp/status.sh alive
```

Log into a node:

```bash
bash skyrl-test/opd-blog-v2/gcp/ssh.sh b200-1
```

## After a preemption

A preempted spot VM is stopped, not deleted. Its boot disk keeps the data, checkpoints, exports, logs and keys. The
NVMe array is wiped, so the HF cache goes with it, and Ray is down.

1. `status.sh` shows the node as `PREEMPTED`.
2. Run `02_setup_nodes.sh --yes` again, with `--only <node>`. It starts the VM in its zone, rebuilds the array and
   starts Ray. A stopped VM can only restart where it is, so with no capacity there it fails; try again later.
3. Start the run with the same command as before, from its README. It resumes from its last checkpoint (every 10
   steps), so at most 10 steps are lost, plus the model downloads again.

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
| `GCP_H100_NODES`, `GCP_B200_NODES` | 2, 2 | Most nodes of each kind |
| `GCP_H100_ZONES`, `GCP_B200_ZONES` | empty | Empty means every zone that offers the machine |
| `GCP_TIMEOUT_SECS` | 14400 | When `01_create_nodes.sh` stops asking |
| `GCP_BOOT_DISK_GB` | 2000 | Boot disk size |

State lives in `~/opd-blog-v2-gcp/`, outside this public repo, because gcloud's messages can carry the project id.

## Verified and assumed

Verified on 2026-11-09:

- With real `gcloud`, read-only: `01_create_nodes.sh` plans 2 H100 nodes in 31 zones and 2 B200 nodes in 16, with the
  current image and 2,000 GB disks; `02_setup_nodes.sh` and `status.sh` find the four nodes, none created yet.
- Against a stand-in `gcloud`: `status.sh` reports a training, waiting, died, stalled and preempted run correctly and
  prints the end of a log; `02_setup_nodes.sh` plans, filters with `--only`, and refuses to start without keys.
- `01_create_nodes.sh`, `down.sh`, `ssh.sh`, `gcp/status.sh` and `vm_startup.sh` are the last kit's, which got six
  nodes on 2026-10-31, with only names and sizes changed.
- `node_setup.sh` is the last kit's, including its fix for a half-stopped Ray cluster, which was tested on a B200 on
  2026-11-02. Its `report` and `stop-run` now take the run's script, so one node can hold two runs one after the other.

Not verified: `02_setup_nodes.sh --yes` end to end. It is the last kit's `launch_all.sh` setup, minus starting a run,
but this copy has not set up a real node.

## Files

`config.sh` (settings and helpers), `01_create_nodes.sh`, `02_setup_nodes.sh`, `node_setup.sh` (runs on the VM),
`status.sh`, `ssh.sh`, `down.sh`, `vm_startup.sh` (open-file limits at boot).
