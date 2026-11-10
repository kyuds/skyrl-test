# OPD blog, v2

The experiments for SkyRL's on-policy distillation blog: one GRPO teacher run and four distillation runs on GCP spot
nodes, all on M2RL's data. Each folder's README has its exact commands and settings; this page is the plan and the
order of work.

## Storyline

1. **SkyRL supports on-policy distillation, as Thinking Machines described it.** A student samples, a teacher scores
   every token, and the student moves toward the teacher. Two experiments, one per model family, each distilling a
   student from an RL-trained teacher of its own lineage on the teacher's own RL prompts:
   - M2RL-SFT from M2RL-RL_Math.
   - Qwen3-8B from Qwen3-8B plus GRPO, trained here on the same prompts.

   OPD is normally done on SFT-trained students, so the student already writes answers the teacher can grade. Both
   students are here: each is its teacher's own starting point.
2. **OPD breaks easily.** Distilling M2RL-SFT from Qwen3-32B, a better teacher, with nothing else changed, is expected
   to crash. Qwen3-32B is 8 times larger and scores higher on every benchmark both models report, math included (72.9
   against RL_Math's 70.4 on AIME 2025). A better teacher can still fail to teach: alignment between teacher and student
   comes first, through an SFT warm-up or clipping. Those fixes are not run yet; this phase shows the crash.
3. **In practice people use MOPD.** M2RL-SFT distilled from three M2RL experts at once, math, coding and instruction
   following, each scoring the rows of its own domain.

## The runs

| Phase | Run | Student | Teacher(s) | Data | Thinking | Node |
|---|---|---|---|---|---|---|
| teacher | `grpo_qwen3_8b_nothink` | Qwen3-8B, GRPO | none (reward) | math | off | `b200-1` |
| 1.1 | `p1_opd_m2rl_sft_from_rl_math` | M2RL-SFT | M2RL-RL_Math | math | on | `h100-1` |
| 1.2 | `p1_opd_qwen3_8b_from_grpo` | Qwen3-8B | the GRPO teacher | math | off | `b200-1`, after the teacher |
| 2 | `p2_opd_m2rl_sft_from_qwen3_32b` | M2RL-SFT | Qwen3-32B | math | on | `b200-2` |
| 3 | `p3_mopd_m2rl_sft_from_math_code_if` | M2RL-SFT | M2RL-RL_Math, RL_Coding, RL_IF | math, code, IF | on | `h100-2` |

- **Models:** M2RL's are 4B models from Qwen3-4B-Base, from the paper "To Mix or To Merge" (arXiv 2602.12566,
  huggingface.co/collections/Jackwang111/m2rl).
- **Data:** "math" is M2RL's math RL set, 22,056 prompts from DAPO-17k and Skywork-OR1. Phase 3 adds M2RL's coding and
  IF prompts. All of it comes from NVIDIA's Nemotron-3-Nano RL blend, in M2RL's exact prompt formats. Every M2RL teacher
  was RL-trained on its domain of this data, and the GRPO teacher is trained on the math part.
- **Eval:** every run evaluates on AIME 2024 and 2025.
- **Logging:** everything goes to W&B project `opd-blog-v2`. Chosen exports go to the Hub as
  `kyuds/opd-blog-v2/<run>_step<N>`.
- **Recipe:** the OPD runs share one recipe, M2RL's multi-teacher OPD: 256 prompts × 4 samples, one optimizer step
  per training step, learning rate 1e-6, 300 steps (`phase1-opd/README.md`). Within phase 1, and between 1.1, 2 and 3,
  only the teachers differ.

## Order of work

From `~/dev/skyrl` on your Mac unless a README says otherwise.

1. **Commit and push this kit.** The VMs clone it from GitHub, branch `opd-blog-v2`.
2. **Check the configs.** This builds all ten configs (five runs, normal and smoke) with the pinned SkyRL and prints
   what each will do:

   ```bash
   bash skyrl-test/opd-blog-v2/common/check_configs.sh
   ```

3. **Get and set up the nodes:** `gcp/README.md`.
4. **On each node, a smoke test, then the run:** `grpo-qwen3-8b/`, `phase1-opd/` (1.1), `phase2-opd/` and
   `phase3-mopd/` can all start at once.
5. **When the GRPO run is done:** pick its teacher step and start phase 1.2 on the same node (`phase1-opd/README.md`).
6. **Watch:**

   ```bash
   bash skyrl-test/opd-blog-v2/status.sh
   ```

   plus W&B, as each README's "What to watch" says.
7. **When a run is done:** upload the exports you want (`common/upload_export.sh`), then stop its node
   (`gcp/README.md`).

## Layout

| Path | What | Runs on |
|---|---|---|
| `config.sh` | Pinned SkyRL commit, names, models, and which run goes on which node | both |
| `status.sh` | One line per run: node, preemptions, step, ETA, log age | Mac |
| `gcp/` | Get, set up, inspect and stop the nodes | Mac (`node_setup.sh` on the VM) |
| `data/` | The data; called by every run | VM |
| `common/` | `kit_main.py` (entrypoint and math grader), `run_lib.sh` (shared recipe and launch), `00_env.sh`, `upload_export.sh`, `check_configs.sh` | VM; `check_configs.sh` on the Mac |
| `grpo-qwen3-8b/`, `phase1-opd/`, `phase2-opd/`, `phase3-mopd/` | One script per run, and its README | VM |

SkyRL is pinned to `988d977f`, the head of `kyuds/opd-mopd` (PR #2679, the OPD stack through MOPD) on 2026-11-09. The
stack is not merged. To move the pin, change `SKYRL_COMMIT`, check out the same commit in your `SkyRL/`, and run
`check_configs.sh` before pushing. Configs changed during the stack, and it catches renamed or removed keys.

## Verified and assumed

Verified on 2026-11-09, on the Mac, with no GPU:

- All ten configs build with the pinned SkyRL's own parser and validators, `validate_opd_cfg` included. They fit 8
  GPUs, and every engine's context covers its longest sequence.
- The data on a slice of the real blend, read by SkyRL's own dataset loader. The MOPD route label reaches the
  environment, and the grader passes 15 cases under the ANTLR version SkyRL pins (`data/README.md`).
- Qwen3-32B and the four M2RL models share one chat template, byte for byte, so every teacher scores the student's
  own format and M2RL-SFT is used as published.
- The GCP scripts' plans against real `gcloud`, and `status.sh` against a stand-in (`gcp/README.md`).
- That Ray passes `uv run --with math-verify==0.9.0` on to its workers, read in Ray's uv hook. Not run on a cluster.

Not verified:

- **No GPU:** nothing here has run on one yet. The smoke tests are the first check, and the first real steps show
  memory and speed.
- **Run times:** estimates in each README.
- **Code and IF evaluation for phase 3:** not set up (`phase3-mopd/README.md`).
- **The cause of a phase 2 crash:** if it happens, the run shows that this teacher breaks OPD, not which of its
  differences does.

## Tradeoffs and what could go wrong

- **Unmerged stack:** the kit pins a commit of the OPD stack, not merged SkyRL. Results describe that code. If the
  stack changes before the blog, rerun or say which commit was used.
- **Selection on AIME:** teacher steps and peak students are chosen on AIME, which is also the reported eval. Report
  the student at every step and the GRPO teacher at the step used, not each run's maximum.
- **Thinking off for the GRPO teacher:** cheaper by about 25 times, but phase 1's two experiments then differ in mode
  as well as model. Within each experiment, student and teacher share the mode.
- **Unknown capacity:** the plan needs 2 B200 and 2 H100 nodes. With fewer, the runs go one after another; `V2_RUNS`
  in `config.sh` assigns them.
