# OPD full experiments

For now this kit only does one thing: find out how much GPU compute we can get on GCP. That decides which
experiment plan to run (`plan/opd-blog/experiment-sequence.md`, in the workspace). Node setup and the run scripts
come later.

## What it asks for

Spot nodes, in every zone of every region that offers them, all requested in parallel:

| Kind | Machine | GPUs | At most | Names |
|---|---|---|---|---|
| H100 | `a3-highgpu-8g` | 8 × H100 | 4 nodes | `kyuds-opd-exp-h100-1` to `-4` |
| B200 | `a4-highgpu-8g` | 8 × B200 | 2 nodes | `kyuds-opd-exp-b200-1`, `-2` |

The names use hyphens because GCP does not allow underscores in VM names. Each name is fixed, so the caps cannot be
exceeded. Each node is independent: they do not form a cluster.

## Before you start

Run these from `~/dev/skyrl`, on your Mac.

The login lasts about a day, so refresh it first, or it can expire during the 4 hours:

```bash
gcloud auth login
```

Check the plan. This creates nothing:

```bash
bash skyrl-test/opd-full-experiments/gcp/01_create_nodes.sh
```

## Get the nodes

**Billing starts as soon as a node is up, even while the script waits for the others.** Plug the Mac in and keep the
lid open: `caffeinate` does not stop a closed lid from sleeping the Mac.

```bash
nohup caffeinate -i bash skyrl-test/opd-full-experiments/gcp/01_create_nodes.sh --yes > ~/opd_exp_nodes.log 2>&1 < /dev/null &
```

```bash
tail -f ~/opd_exp_nodes.log
```

It ends when it holds every node, or after 4 hours. Requests still in flight then get 5 minutes to finish. The log
ends with the nodes it holds, requests by region, and every quota refusal, which together answer how much compute
is available. A progress line appears every 5 minutes.

To stop early, keeping what it has:

```bash
touch ~/opd-full-experiments-gcp/stop
```

To get less, or only one kind:

```bash
GCP_H100_NODES=2 GCP_B200_NODES=0 nohup caffeinate -i bash skyrl-test/opd-full-experiments/gcp/01_create_nodes.sh --yes > ~/opd_exp_nodes.log 2>&1 < /dev/null &
```

Running it again picks up where it left off: nodes that exist are kept, and stopped ones are started again.

### How refusals are handled

| Refusal | What it does |
|---|---|
| No capacity | Tries the next zone, and goes round again a minute after each pass |
| Quota | Leaves that kind alone in that region for 30 minutes |
| Rate limit, server error | Carries on |
| gcloud login expired | Warns and waits. `gcloud auth login` in another terminal is picked up by the running script |
| Anything else | Prints it and leaves that kind alone in that zone for 10 minutes |

Quota refusals are not permanent, because requests in flight hold quota for a moment and other people in the project
release theirs.

If a node is preempted while the script is running, the script starts it again in its own zone.

## Check on the nodes

Every node and its state:

```bash
bash skyrl-test/opd-full-experiments/gcp/status.sh
```

Also ssh into each running node and count its GPUs:

```bash
bash skyrl-test/opd-full-experiments/gcp/status.sh alive
```

`TERMINATED` means preempted or stopped. Log into a node:

```bash
bash skyrl-test/opd-full-experiments/gcp/ssh.sh h100-1
```

If a script says gcloud's login has expired, that is on the Mac only. The nodes are not affected:

```bash
gcloud auth login
```

## Stop paying

Stops every running node. Boot disks are kept, and you pay for those only:

```bash
bash skyrl-test/opd-full-experiments/gcp/down.sh stop all
```

Deletes every node and its boot disk. It asks first, so run it in a terminal, not under `nohup`:

```bash
bash skyrl-test/opd-full-experiments/gcp/down.sh delete all
```

Either also takes one node, such as `h100-1`. `down.sh` refuses to run while `01_create_nodes.sh` is running, because
the script would bring the nodes back.

## Knobs

Set in the environment. `gcp/config.sh` lists them all.

| Variable | Default | Meaning |
|---|---|---|
| `GCP_H100_NODES`, `GCP_B200_NODES` | 4, 2 | Most nodes of each kind |
| `GCP_H100_ZONES`, `GCP_B200_ZONES` | empty | Empty means every zone that offers the machine |
| `GCP_TIMEOUT_SECS` | 14400 | When it stops asking |
| `GCP_RETRY_SLEEP` | 60 | Pause between passes over the zones |
| `GCP_BOOT_DISK_GB` | 1000 | Boot disk size |
| `GCP_VERBOSE` | 0 | 1 prints every capacity refusal |

State lives in `~/opd-full-experiments-gcp/`, outside this public repo, because gcloud's messages can carry the
project id. `attempts.tsv` there records every request.

## Verified and assumed

Verified on 2026-10-31, on the Mac's bash 3.2, against a stand-in for `gcloud`. No real VM was requested.

- All 4 + 2 nodes obtained in parallel with the right names and flags, and the caps never exceeded.
- The 4-hour timeout (with a shortened clock), the stop file, and a second copy refusing to start.
- Stockouts, quota refusals with a timed retry, and an API rate limit not mistaken for a GPU quota refusal.
- A login that expires mid-run and comes back, a preempted node restarted in its zone, and a re-run picking up
  existing nodes.
- A create that reports failure but worked does not produce a second node.
- The `nohup` command above surviving its shell's exit and a SIGHUP, and `status.sh`, `ssh.sh` and `down.sh`.

Not verified:

- The real list of zones: gcloud's login had expired, so `gcloud compute machine-types list` was not run.
- Quota outside `us-central1`, where H100 spot quota was seen on 2026-10-05. The report will show it.
- That `b200-vpc` has a subnet in every region. It was checked only for the US regions the last kit used.
- That an `a4-highgpu-8g` boots with one network interface: no B200 has ever come up for this project.
- The refusal messages come from what the previous kit matched and from memory of GCP's wording, so a new wording
  would show up as "other" and be printed.

## Files

In `gcp/`: `config.sh` (settings and helpers), `01_create_nodes.sh`, `status.sh`, `ssh.sh`, `down.sh`, and
`vm_startup.sh` (raises the open-file limits at boot, copied from `opd-gcp-spot-full-repro`).
