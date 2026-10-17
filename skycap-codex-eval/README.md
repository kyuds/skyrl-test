# skycap-codex-eval

An end-to-end check of skycap's Responses support with the real Codex CLI, in **eval mode**: Codex solves a set of graded programming tasks through skycap (token mode, vLLM), every rollout is recorded, nothing is trained. Then:
- `02_analyze.py` checks the graph structure of every record and explains each deviation;
- skycap-viewer's own `check` validates the records;
- `03_screenshots.py` captures the viewer's views for a review.

It answers the review request: "an end to end skycap run with codex in eval mode to just collect the traces and show them via the viewer … screenshots and analysis of the graph structure to show that it works as expected, or some test that captures this". The dialect's unit tests are in SkyRL (`skycap/tests/test_dialect_responses.py` and friends).

It reuses everything from [`../skycap-responses`](../skycap-responses/README.md): the GCP VM and its `gcp/` scripts, vLLM, Codex and skycap-viewer. It adds its own skycap server (`:8082`, started and stopped by each run) and its own viewer (`:8766`), so eval records stay apart from the `:8080`/`:8081` ones.

**How to read this:**
- Commands run **on your Mac, from `~/dev/skyrl`**, unless they say otherwise.
- Commands for the VM go through `gcp/ssh.sh --command '…'`.
- Terminal **B** keeps the tunnel open, as in skycap-responses.

---

## Step 0: the VM is up (once)

Follow skycap-responses steps 0–3: get the VM, set it up, start the servers. vLLM must be serving; this kit only needs `:8000`.

```bash
bash skyrl-test/skycap-responses/gcp/alive.sh
```

Restart the tunnel in terminal B. It now also forwards `:8766`, the eval viewer.

```bash
bash skyrl-test/skycap-responses/gcp/tunnel.sh
```

## Step 1: get this kit onto the VM

Push this repo's branch, then pull it on the VM:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'git -C ~/SkyRL/skyrl-test pull'
```

## Step 2: start the eval viewer (once per VM boot)

It serves every eval run's records on `:8766`, and picks up new records as they are written.

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'mkdir -p ~/SkyRL/skyrl-test/skycap-codex-eval/runs && cd ~/SkyRL && nohup bash skyrl-test/skycap-codex-eval/04_view.sh > skyrl-test/skycap-codex-eval/runs/viewer.log 2>&1 < /dev/null &'
```

Open **http://127.0.0.1:8766**.

## Step 3: smoke run (about 2 minutes)

3 tiny tasks × 2 rollouts, to check the pipeline:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'cd ~/SkyRL && bash skyrl-test/skycap-codex-eval/01_eval.sh --tasks smoke --repeats 2 --run smoke'
```

Each rollout prints one line: `PASS` or `fail`, why Codex stopped, calls, paths, unbridged calls, seconds, the grader's verdict, and the trajectory id. The run's records appear in the viewer as the run `smoke`.

The first test, on 2026-10-17 with Qwen3-8B and Codex 0.154.0:
- 3 of 6 rollouts were accepted.
- All 6 were one path with 0 unbridged calls.
- The 3 failures were the model writing `\n` literally with `echo`.

## Step 4: the CodeContests run (long; detached)

16 CodeContests problems × 4 rollouts, 8 Codex processes at a time:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'mkdir -p ~/SkyRL/skyrl-test/skycap-codex-eval/runs && cd ~/SkyRL && nohup bash skyrl-test/skycap-codex-eval/01_eval.sh --tasks codecontests --n 16 --repeats 4 --concurrency 8 --run cc16x4 > skyrl-test/skycap-codex-eval/runs/cc16x4.log 2>&1 < /dev/null &'
```

Follow it. Ctrl-C stops only the `tail`, not the run:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'tail -f ~/SkyRL/skyrl-test/skycap-codex-eval/runs/cc16x4.log'
```

**How long:** smoke rollouts took 15–60 s each. Expect harder problems to take minutes each, so roughly 30–60 minutes for 64 rollouts. That's an estimate, not measured. A rollout that runs past `--timeout` (900 s) is stopped and recorded with `stop_reason: agent_timeout`.

**The first run:** downloads CodeContests' test split once (63 MB, into `.cache/`). It then checks every candidate problem against the dataset's own Python solutions, which takes about 30 s.

**The trajectories** show up in the viewer while the run goes. The groups view puts each problem's 4 rollouts together, coloured by reward.

**Options** (`--help` lists them all):

| Option | Default | What it does |
|---|---|---|
| `--tasks` | `smoke` | `codecontests` or `smoke` |
| `--n` | 16 | CodeContests problems, lowest Codeforces rating first |
| `--split` | `test` | `test` (165 problems), `valid` (117) or `train` |
| `--max-rating` | 1600 | The highest rating taken |
| `--repeats` | 2 | Rollouts per problem (one viewer group) |
| `--concurrency` | 8 | Codex processes at once |
| `--timeout` | 900 | Seconds per rollout |
| `--run` | `<tasks>-<time>` | Names `records/<run>/` and `runs/<run>/` |

## Step 5: analyze

On the VM:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'cd ~/SkyRL && bash skyrl-test/skycap-codex-eval/02_analyze.sh cc16x4'
```

It prints:
- the shape counts;
- whether any invariant broke;
- skycap-viewer's `check` (format validation);
- skycap-viewer's `summary`.

It writes `runs/cc16x4/analysis.md` and `analysis.json`. It exits 1 if an invariant fails or the viewer's check finds a problem.

Copy the run to your Mac (`records/cc16x4/`, `runs/cc16x4/`, both git-ignored here):

```bash
bash skyrl-test/skycap-codex-eval/pull.sh cc16x4
```

Open the report:

```bash
open skyrl-test/skycap-codex-eval/runs/cc16x4/analysis.md
```

`02_analyze.sh` also runs on the Mac against the pulled records. It needs only the SkyRL checkout on `kyuds/skycap-responses`; the viewer steps are skipped without Node.

## Step 6: screenshots (on the Mac, through the tunnel)

Once, install a headless Chromium (about 100 MB, into `~/Library/Caches/ms-playwright`):

```bash
uv run --no-project --isolated --with playwright python -m playwright install chromium
```

Then:

```bash
uv run --no-project --isolated --with playwright --with httpx python skyrl-test/skycap-codex-eval/03_screenshots.py cc16x4
```

It writes PNGs and an index, `runs/cc16x4/screenshots/README.md`. The PNGs cover:
- the groups view and the rollouts view;
- the Path, Tree and Calls tabs of:
  - every forked rollout (up to 3, with Forks too);
  - the longest clean accepted rollout;
  - a clean failed one.

## Step 7: what to show the reviewer

- **`runs/cc16x4/analysis.md`:**
  - the invariant table (5 checks × every trajectory);
  - the shape counts;
  - every fork and unbridged call with its cause;
  - the tree of a clean multi-turn Codex rollout;
  - what Codex refused.
- **`runs/cc16x4/screenshots/`:** the same, as the viewer shows it.
- **The `02_analyze.sh` output:** the viewer's `check` and `summary`, an independent reading of the same records.

---

## What the analysis checks

`02_analyze.py` rebuilds every record with skycap's own reader (`skycap.record.load`) and checks the graph.

**Invariants.** These must hold for every trajectory, whatever the model did; any failure makes the script exit 1.

| Invariant | What it checks |
|---|---|
| `tokens` | Every node on every path has its tokens |
| `prompt` | Each call's `prompt_tokens` equals the tokens of its path: the ancestors' tokens plus the model node's scaffold (`<\|im_start\|>assistant\n`). Its `completion_tokens` equals the node's sampled tokens. So the stored sequence is exactly what vLLM was sent and returned |
| `logprobs` | Every sampled token has a finite logprob ≤ 0 |
| `dialect` | Every call came in on `openai_responses` |
| `samples` | The training rows (`skycap.samples.build_samples`) end at every leaf, and every sampled token is trained in exactly one row |

**Shape.** This is what a clean Codex rollout looks like:
- one root and one path;
- every call after the first bridged;
- no client-written copy of a model turn.

A rollout that isn't clean isn't a failure, but every deviation is explained:

| What it reports | What happened |
|---|---|
| resampled | The same prompt was sampled twice: a request sent again without a retry marker |
| a model turn came back changed: node N (vs node M) lacks / differs in … | The client sent back a different version of a model turn, e.g. without its reasoning |
| node N on is a re-rendered copy of node M's branch: the call at node K could not extend node J, because … | The model turn came back unchanged, but the renderer couldn't extend it, so the history was re-rendered and forked. The causes the record can show: the turn was cut off, it ended inside `<think>`, or the renderer declined |
| history edited | The client sent a different user, system or developer message: Codex compaction, or a mid-conversation developer message |
| no model turn before it to extend | A new conversation in the same trajectory |

The script was tested on records from this project's runs, where every classification matched a cause found by hand:
- a fork after an unclosed `<think>`;
- the `drop_reasoning` control;
- the `refusals` run's resamples;
- a cut-off turn;
- real Codex rollouts.

## Grading

- **What's judged:** Codex writes `solution.py`, which must read stdin and write stdout. `tasks.py` runs it on up to 30 tests per problem: public, then private, then generated.
- **Time limit:** each test gets 3× the problem's limit + 1 s, since the limits are set for C++.
- **Comparing answers:** whitespace-separated tokens must match, with YES and NO in any case.
- **Which problems are kept:** CodeContests has no checkers. A problem is kept only if up to five of the dataset's own Python 3 solutions all pass its tests. That drops most problems that accept several answers, plus ones Python can't do in time. On the test split it dropped *Windblume Ode* and *Find Array*.
- **What can still slip through:** a problem whose references all print the same one of several valid answers. So a few rewards of 0 may be right answers.

The reward is 1 only if every test passes. The trace analysis doesn't depend on rewards.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `00_env.sh` | — | Sourced by the scripts. Sources `../skycap-responses/00_env.sh` and adds `EVAL_PORT` (8082), `EVAL_VIEWER_PORT` (8766) and `EVAL_RECORDS` |
| `01_eval.sh`, `01_eval.py` | VM | The eval: its own skycap server, Codex per rollout, grading, and finishing each trajectory with `reward` and `stop_reason` |
| `tasks.py` | VM | The task sets, the reference-solution check, and the grader |
| `02_analyze.sh`, `02_analyze.py` | VM or Mac | The graph analysis, plus skycap-viewer's `check` and `summary` |
| `03_screenshots.py` | Mac | Viewer screenshots, with Playwright |
| `04_view.sh` | VM | The eval viewer on `:8766` |
| `pull.sh` | Mac | Copies a run's records and outputs from the VM |

**What each rollout records:**
- **Meta:** `task_id`, `instance_id`, `step: 0`, `repetition_id`, `source`, `rating` and `codex`. The viewer groups rollouts on `step` + `instance_id`.
- **Annotations:** `reward`, `stop_reason` (`completed`, `error` or `agent_timeout`; the viewer masks the last two), `verdict`, `tests_passed`, `tests_total`, `codex_exit` and `seconds`.
- **Codex's own log and events:** in `runs/<run>/codex/<task>/r<k>/`.
- **The workspace it worked in:** in `runs/<run>/work/<task>/r<k>/`.

## Known limits

- **`apply_patch`:** Codex 0.154.0 has no metadata for `Qwen/Qwen3-8B`, so it falls back to defaults. That leaves out `apply_patch`, while the system prompt still describes it. The model calls it anyway and Codex answers `unsupported call`. The report counts these under "Tool calls Codex answered with an error".
- **Codex's settings:** the same as `skycap-responses/04_codex.sh`:
  - retries off;
  - a one-hour stream idle timeout;
  - the engine's context window;
  - web search, `multi_agent` and `goals` off;
  - `--dangerously-bypass-approvals-and-sandbox`.

  So Codex runs the model's commands unsandboxed on the VM, inside each rollout's scratch directory.
- **One vLLM:** the eval shares it with skycap-responses' servers. Running a drive there at the same time slows both.
