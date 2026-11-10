# OPD blog v2

The experiments behind the OPD blog post, as planned on 2026-11-10 with Charlie Ruan. Everything runs on GCP spot
nodes (2 B200 and 4 to 6 H100, 8 GPUs each) with SkyRL pinned at `988d977f` (PR #2679, the OPD stack up to MOPD).
W&B project `opd-blog-v2` (evals in `opd-blog-v2-evals`), Hub repo `kyuds/opd-blog-v2`.

**Every hyperparameter is in a `HYPERPARAMETERS` block at the top of its run script, each value with the source it
came from, and in a table with links in that folder's README.** Nothing in `common/` has a default for them.

## The story

0. **Reproduce published numbers first.** Qwen3-8B and Qwen3-32B (thinking) on AIME with the Qwen3 report's settings;
   M2RL-SFT, RL_Math, RL_Coding and RL_IF on AIME, LiveCodeBench and IFEval/IFBench with M2RL's.
   [`0-evals`](0-evals/README.md)
1. **We support on-policy distillation, as Thinking Machines ran it.** Qwen3-8B-Base SFT'd on 384,000 OpenThoughts-3
   rows of every domain (1a; target 60% on AIME 2024), then distilled from Qwen3-8B on DeepMath prompts (1b; target 70%). The
   same distillation on the Base model with no SFT (1c) shows why the SFT comes first. [`1-tml`](1-tml/README.md)
2. **But OPD breaks easily: initial teacher-student alignment matters.** One student, M2RL-SFT, on M2RL's math
   prompts. 2a: its own RL'd teacher, M2RL-RL_Math (aligned). 2b: Qwen3-32B (stronger, misaligned). Then two fixes for
   2b: 2c, an SFT warm-up on Qwen3-32B's answers; 2d, clipping the teacher's signal. [`2-alignment`](2-alignment/README.md)
3. **Most people use MOPD.** M2RL-SFT distilled from M2RL's math, code and instruction-following experts at once,
   against each expert's published number. No teacher is trained. [`3-mopd`](3-mopd/README.md)

**What it trains:**

- Data: Qwen3-32B's answers to M2RL's 22,056 math prompts (2c's warm-up data).
- 2 SFTs: 1a (8B), 2c's warm-up (4B).
- 2 OPD runs with an 8B student and Qwen3-8B as teacher: 1b, 1c.
- 4 OPD runs with a 4B student (M2RL-SFT): 2a from M2RL-RL_Math (4B), and 2b, 2c, 2d from Qwen3-32B.
- 1 MOPD run, from three 4B teachers.
- Plus evals of 7 published models and of every result.

## Layout

| Folder | What |
|---|---|
| [`orchestrate.sh`](orchestrate.sh) | On your Mac: starts each stage's jobs on their nodes, and shows where every job is |
| [`config.sh`](config.sh) | Every setting shared by the kit, and the job table: stage, node, job, deps, command |
| [`gcp/`](gcp/README.md) | Getting, setting up, and stopping the nodes |
| [`data/`](data/README.md) | RL prompts, AIME, and the pretokenized SFT data |
| [`0-evals/`](0-evals/README.md) ... [`3-mopd/`](3-mopd/README.md) | One folder per section: run scripts and a README with every setting |
| `common/` | What every job shares: the OPD recipe and SFT launcher (`run_lib.sh`), the job runner (`job.sh`, `chain.sh`), the trainer entrypoint (`kit_main.py`), the model fixer (`chat_model.py`), the config check |

## The plan

Three stages, each started by hand with `orchestrate.sh run <stage>`. A stage is not a barrier: each job waits only
for its own deps (`config.sh`), and stage 1's SFT keeps running through stage 2.

| Node | Stage 1: evals, data, SFT | Stage 2: what can train now | Stage 3: after 1a's SFT |
|---|---|---|---|
| b200-1 | `sft_1a`: Qwen3-8B-Base on OpenThoughts-3 (about 1 day) | (still 1a) | `opd_1b`, then its eval |
| b200-2 | `traces_2c`: Qwen3-32B's answers for 2c | `sft_2c` warm-up, then `opd_2c`, then its eval | |
| h100-1 | eval: Qwen3-8B on AIME | `opd_1c`, then its eval | |
| h100-2 | eval: Qwen3-32B on AIME | `opd_2b`, then its eval | |
| h100-3 | eval: M2RL-SFT on AIME | `opd_2d`, then its eval | |
| h100-4 | eval: M2RL-RL_Math on AIME | `opd_2a`, then its eval | |
| h100-5 | eval: M2RL-SFT on LiveCodeBench, IFEval, IFBench | `mopd_3`, then all six benchmarks | |
| h100-6 | evals: RL_Coding on LiveCodeBench, RL_IF on IFEval and IFBench | (spare) | eval of 1a's SFT model |

Why this order:

- **No OPD run starts before its teacher's and its student's published numbers are reproduced.** Each waits for both
  evals (`config.sh`); MOPD waits for all five M2RL evals. The SFT and the Qwen3-32B answers do not train against a
  teacher, so they start at once.
- **The critical path is 1a then 1b** (SFT then OPD of the 8B model), so 1a starts first, alone on a B200.
- **2c's whole chain runs on the other B200**: Qwen3-32B's answers, the warm-up, then the OPD with Qwen3-32B as teacher.
  Qwen3-32B fits one B200 GPU but needs two H100s.
- **Every final eval runs on the node that trained the model**, chained after the run, so no export has to move.
  1a's output goes to the Hub, where the eval on h100-6 fetches it.

**Rough timeline** (none of it measured): stage 1 evals take a few hours (Qwen3-32B's several), Qwen3-32B's answers
several hours on a B200, 1a about a day (5 billion tokens, estimated at 35% GPU use), each OPD run 1 to 2 days, and
the runs with Qwen3-32B as teacher on H100s probably longer. End to end, 3 to 4 days. The in-training AIME eval runs
every 5 steps at 8 samples: about 10 to 15 minutes each on 4 H100s, roughly 12 hours over a 300-step run (estimated
from vLLM's KV-cache limits, not measured). `EVAL_EVERY` and `EVAL_N` in each run script change it.

## Running it

Every command runs on your Mac, from `~/dev/skyrl`, unless it says otherwise.

**1. Push the kit.** The nodes run what is on GitHub; `orchestrate.sh` and `02_setup_nodes.sh` refuse otherwise.

**2. Check the configs** against SkyRL's own validators (all jobs, normal and smoke; no GPU needed):

```bash
bash skyrl-test/opd-blog-v2/common/check_configs.sh
```

**3. Get and set up the nodes**, as [`gcp/README.md`](gcp/README.md) says: `01_create_nodes.sh`, then
`02_setup_nodes.sh`. The six nodes from the earlier smoke tests are stopped with their disks; `02_setup_nodes.sh`
starts them again.

**4. Smoke-test the new parts** (recommended; none of them has run on a GPU yet). On a B200 and an H100, from
`~/SkyRL` (log in with `bash skyrl-test/opd-blog-v2/gcp/ssh.sh <node>`):

```bash
bash skyrl-test/opd-blog-v2/1-tml/sft.sh --smoke
```

```bash
bash skyrl-test/opd-blog-v2/0-evals/run.sh eval_qwen3_32b_math Qwen/Qwen3-32B qwen all --smoke
```

```bash
bash skyrl-test/opd-blog-v2/2-alignment/traces.sh --smoke
```

```bash
bash skyrl-test/opd-blog-v2/2-alignment/sft_2c.sh --smoke
```

```bash
bash skyrl-test/opd-blog-v2/2-alignment/opd.sh 2b --smoke
```

```bash
bash skyrl-test/opd-blog-v2/1-tml/opd.sh 1c --smoke
```

On an H100, the Qwen3-32B ones also test two GPUs per engine.

**5. Start stage 1.** The plan first, then start it:

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh run 1
```

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh run 1 --yes
```

**6. Follow it.** Every job's state and progress, every node's GPU use and free disk, and what can start next:

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh
```

The end of one job's log:

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh log eval_qwen3_8b_math
```

**7. Before stage 2, read the stage-1 results:** every eval within about 3 points of its target
([`0-evals/README.md`](0-evals/README.md)). Then:

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh run 2 --yes
```

**8. Stage 3** once 1a is done:

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh run 3 --yes
```

`run <stage>` never starts a job that is done, skips a busy or broken node with a reason, and stops a node's list at
the first job whose deps are not done yet. Running a stage again later starts the rest. To stop a node's jobs
(checkpoints and outputs stay):

```bash
bash skyrl-test/opd-blog-v2/orchestrate.sh stop h100-2
```

**After a preemption:** `02_setup_nodes.sh --only <node> --yes`, then the same `run <stage> --only <node> --yes`. The
interrupted job resumes from its last checkpoint ([`gcp/README.md`](gcp/README.md)).

## Fewer nodes

With 4 H100s, move h100-5's jobs after h100-1's and h100-6's after h100-4's, in your Mac's shell, before
`02_setup_nodes.sh` and `orchestrate.sh`:

```bash
export V2_NODE_MAP="h100-5=h100-1 h100-6=h100-4"
```

MOPD then starts after 1c's eval, a day or two later, and 2a a few hours later (after two more evals). The end date
moves by about as long as 1c takes.

## Decisions in this kit

- **Section 1 follows Thinking Machines' cookbook at the blog's commit** (`330b73d8d7`, 2025-10-27). The current
  cookbook runs Qwen3.5-9B; its last Qwen3 version (`58e781e78e`) has the same values. Sources and links:
  [`1-tml/README.md`](1-tml/README.md).
- **1a SFTs on every OpenThoughts-3 domain** (math, code, science), as the cookbook does. `OT3_DOMAINS=math` in
  `1-tml/sft.sh` keeps math only.
- **Sections 2 and 3 follow M2RL's MT-OPD**: its paper's appendix A and its launch script, pinned at `c94ce97c`. Where
  they disagree (400 steps in the script, 300 in the paper), the paper wins.
- **2c's warm-up uses Qwen3-32B's answers to all 22,056 math prompts**, the set its OPD then trains on, and M2RL's own
  recipe for SFT on teacher answers (MT-OFF: lr 5e-5, batch 512). No paper gives a warm-up recipe.
- **Every Base or SFT'd model gets a post-trained Qwen3's tokenizer files and context length**
  (`common/chat_model.py`), so it stops at `<|im_end|>` and fits the evals' 32,768-token answers.
- **Each section's evals use one setting**: the Qwen3 report's for section 1 and the Qwen teachers, M2RL's for sections 2
  and 3 and the M2RL models.
- **MOPD has three teachers**, math, code and IF; M2RL's had five.

## Tradeoffs and what could go wrong

- **Nothing in this layout has run on a GPU.** New since the last smoke tests: the SFT trainer at 8B and 16k tokens,
  the eval harness, Qwen3-32B at two GPUs per engine (as eval model, answer generator and OPD teacher), and the warm-up.
  Step 4's smoke tests cover the plumbing, not throughput.
- **2a and 2b differ in teacher size as well as lineage** (4B against 32B). The smoke tests' reverse KL (0.11 against
  0.81) and 2c and 2d fixing 2b with the same 32B teacher are the evidence that alignment, not size, is the cause.
- **Qwen3-32B is a slow teacher on H100s.** 2b and 2d score every rollout on two engines of two GPUs; they may take
  longer per step than 2a.
- **2c's warm-up may do most of the work itself** at M2RL's SFT learning rate; its step-0 eval says
  ([`2-alignment/README.md`](2-alignment/README.md)).
- **The critical path is long and single.** If 1a fails late, 1b moves by a day. Its checkpoints every 250 steps
  bound a preemption's cost to about 1.5 hours.
- **Section 1's learning rates are the cookbook's**, with no warmup: 1e-4 for the SFT, 5e-5 for full-FT OPD.
  [`1-tml/README.md`](1-tml/README.md) says how to rerun gentler.
- **Reproduction may miss for reasons that are not ours to fix:** LiveCodeBench's date windows and prompt, M2RL's
  eval top-p, and QwQ's math evaluator versus ours are assumptions ([`0-evals/README.md`](0-evals/README.md)).
- **Thinking Machines' numbers will be close, not equal:** their 400k subset is not public, and they trained on Tinker.
- **Novelty:** the Hub account `mhseo2` is posting checkpoints of a near-identical study (OPD of Klear-8B and
  Qwen3-8B-Base students on OpenThoughts-3 math, with SFT warm-ups on Qwen3-32B answers). No paper yet.

## Verified and assumed

Verified on 2026-11-10, on the Mac:

- `check_configs.sh`: all 50 job configs (25 jobs, normal and smoke) pass SkyRL's own SFT and OPD validators, the GPU
  budget and the length checks, and print the values each run's `HYPERPARAMETERS` block sets.
- Every source value: the cookbook at `330b73d8d7` and `58e781e78e`, the Tinker SDK of 2025-10-17, M2RL's launch
  script, slime's defaults and loss code at `c94ce97c`, and the Qwen3 report's tables and section 4.6, read directly.
- `chat_model.py` on Qwen3-0.6B-Base with Qwen3-0.6B's files: ends at `<|im_end|>`, takes the 40,960 context,
  renders the student's prompt.
- `prepare_sft.py` on 300 OpenThoughts-3 rows, read back by SkyRL's own loader. The eval request builder and all
  three graders, on hand-written answers.

Verified earlier, on real nodes (2026-11-09, previous layout): node setup, Ray, data prep, and OPD smoke runs with an
8B and a 4B student, Qwen3-32B and RL_Math as single teachers of M2RL-SFT, and three routed teachers.

Not verified: anything else on a GPU (see the tradeoffs), `orchestrate.sh` and `common/chain.sh` against real nodes,
and every timing above.
