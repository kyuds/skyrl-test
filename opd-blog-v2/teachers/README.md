# Teacher servers: OPD teachers on nodes of their own

Since 2026-11-11, the OPD runs on the H100s (2a, 2b, 2d and MOPD) score with teachers served from two extra H100
nodes, h100-5 and h100-6. Each run's student has all 8 GPUs of its node. 2c, on b200-2, still runs its teacher on 4 of
its own GPUs.

## Why

The first launch of stage 2 (2026-11-11, 4 GPUs for the student and 4 for its teacher on each node) ran at about 35
minutes a step with the eval, so 400 steps would have taken 9 to 10 days. The logs showed why:

- **The student ran out of KV cache.** Each of its 4 vLLM engines held about 388,000 tokens (99% used), so only 30 to
  60 of its 256 answers ran at once, with 60 to 75 queued.
- **The teacher was mostly idle.** Teachers only score, and they score each answer as it finishes. 2a's RL_Math
  engines averaged about 2,300 tokens a second against a peak of 37,000. 2b's Qwen3-32B engines averaged about 4,000
  against 16,500.

So the student needed the GPUs, and the teacher could be shared. Twice the student GPUs is about twice the KV cache,
training throughput and eval throughput, which is expected to roughly halve a step. That estimate is from the
numbers above; it was not measured before the switch.

Ruled out:
- **6 student GPUs and 2 for the teacher:** 1,024 answers a step do not split evenly over 6 GPUs, and MOPD needs a
  GPU for each of its 3 teachers.
- **The student and teacher sharing all 8 GPUs:** SkyRL at the pinned commit launches a teacher on GPUs of its own.

## What runs where

`config.sh`, `V2_TEACHER_SERVERS`:

| Model | Node | Router port | Engines | Serves |
|---|---|---|---|---|
| Qwen3-32B | h100-5 | 31000 | 4 × 2 GPUs | 2b, 2d |
| M2RL-RL_Math | h100-6 | 31001 | 4 × 1 | 2a, MOPD's math rows |
| M2RL-RL_Coding | h100-6 | 31002 | 2 × 1 | MOPD's code rows |
| M2RL-RL_IF | h100-6 | 31003 | 2 × 1 | MOPD's IF rows |

Each served teacher scores up to 18,433 tokens (`V2_TEACHER_MAX_LEN`), sections 2 and 3's 2,048-token prompt and
16,384-token answer plus one.

## How it works

- **The server** (`serve.sh`, `serve.py`): the job `serve_teachers_<node>` in stage 2 runs SkyRL's own
  `launch_teacher` (`skyrl/train/opd/teacher_launch.py`). That is the same vLLM servers, frozen-model arguments and
  vllm-router a run launches for a teacher on its own GPUs, under SkyRL's Ray runtime environment.
  - One change: the router gets a fixed port instead of a random one.
  - Before reporting up, it scores one token through the router with SkyRL's own preflight.
  - Then it checks the router and every server once a minute. Five failed minutes in a row end the job as failed.
  - The job never finishes on its own.
- **The runs** (`common/run_lib.sh`, `v2_remote_teacher`): the teacher is `backend=vllm` with `server_url` set to
  `http://<node's internal IP>:<port>`.
  - The IP comes from `~/opd-store/nodes.env`, which `orchestrate.sh run` writes on every node it starts.
  - Each run sends up to 16 requests per engine at once. A request may take up to 10 minutes, and a failed one is
    retried 3 times.
  - At launch, a run waits up to 6 hours for its teacher to answer `/health`, so servers and runs can start together.
- **The network:** all nodes are on `b200-vpc`, whose firewall rule `b200-allow-internal` lets in all TCP from
  10.128.0.0/9, across regions.

## Running it

`orchestrate.sh run 2 --yes` starts the servers and the runs together; the runs wait for the servers. The PROGRESS
column of `orchestrate.sh` shows each server's last check, e.g. `serving qwen3_32b:31000 ok (08:05Z)`. Logs:

- `~/opd-store/logs/serve_teachers_h100_5.log` on the node: the launch, then one line per teacher once it is up.
- `~/opd-store/logs/skyrl/serve_teachers_h100_5/`: vLLM's and the router's own logs.

Stop a server with `orchestrate.sh stop h100-5`. Start it again with `orchestrate.sh run 2 --only h100-5 --yes`.

## What could go wrong

- **One teacher node stops four runs.** If h100-5 is preempted, 2b and 2d fail on their next teacher request once the
  retries run out. If h100-6 is, 2a and MOPD do. Each resumes from its last checkpoint (every 10 steps) once the
  server is back:
  1. Restart the node: `gcp/02_setup_nodes.sh --only <node>`, or `gcp/migrate.sh` if its zone has no capacity.
  2. Start its server: `orchestrate.sh run 2 --only <node> --yes`.
  3. Rerun the failed runs with `orchestrate.sh run 2 --yes`.

  A teacher node recreated elsewhere has a new IP. `orchestrate.sh run` rewrites `nodes.env` on each node it starts.
- **Shared servers.** 2b and 2d share Qwen3-32B, and 2a and MOPD share RL_Math, so one run's burst can slow the other's
  scoring. The estimate, from 2b's first launch: two runs at the faster pace need about 30% of the Qwen3-32B node's
  peak. Not measured under load yet.
- **Cross-region traffic.** Each run step sends its 1,024 answers to the teacher and gets the scores back, roughly
  1 GB a step across regions. That is an estimate, not a measurement: about 1.6 TB over the four runs, billed as
  network egress at roughly $0.01 to $0.08 per GB.
- **Not the first launch's layout.** The runs restarted from step 0 with 8 student GPUs. A checkpoint saved on 4 GPUs
  does not load on 8 (one file per GPU, named by GPU count), so the first launch's 8 steps were set aside. The recipe
  is unchanged; only where it runs changed.
