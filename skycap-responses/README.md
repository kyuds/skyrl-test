# skycap-responses

This kit checks skycap's Responses API on one spot GPU VM on GCP, in both modes:
- vLLM serves the model;
- a skycap server in **token mode** (`:8080`) samples through it and records;
- one in **text mode** (`:8090`) forwards each call to vLLM's own `/v1/responses` and records what it answered;
- a driver on your Mac runs Responses conversations through them, over an ssh tunnel, and checks each record;
- real Codex runs on the VM;
- skycap-viewer shows the records.

The design is `plan/skycap-dialects/design.md` in the workspace. The code is SkyRL branch `kyuds/skycap-responses`.

**How to read this:**
- Every command runs **on your Mac, from `~/dev/skyrl`**, unless it says otherwise.
- Use two terminals:
  - **A** for the steps;
  - **B** for the tunnel, which stays open.
- Steps 1–3 are once per VM.
- Steps 4–8 can be repeated in any order.
- Step 9 stops the bill.

---

## Step 0: once, before the first VM

**Log gcloud in.** The login lasts about a day. When it expires, every script says so, and the VM is not affected.

```bash
gcloud auth login
```

**Set the project.** Its id is in [Charlie's guide](https://gist.github.com/CharlieFRuan/6eeae93d70ede0e81f12f91a4eb74d57). The scripts read it from your gcloud config, never from this repo.

```bash
gcloud config set project <project id>
```

**Push both branches.** The VM clones them from GitHub, and step 2 refuses to start while either is missing.
- **SkyRL:** commit `kyuds/skycap-responses` and push it to NovaSky-AI/SkyRL.
  - To use a fork instead, export `SKYRL_REPO=https://github.com/<you>/SkyRL.git` in terminal A before step 2.
- **skyrl-test:** push the branch this checkout is on.
  - The VM checks out the same branch name.
  - Override it with `KIT_BRANCH`.

**If you have never run `gcloud compute ssh` on this Mac,** run it once by hand after step 1 (`bash skyrl-test/skycap-responses/gcp/ssh.sh`). It creates `~/.ssh/google_compute_engine` and asks for a passphrase. If you skip this, step 2 waits on a prompt it hides.

---

## Step 1: get the VM (terminal A)

```bash
caffeinate -i bash skyrl-test/skycap-responses/gcp/01_create_vm.sh
```

- **What it asks for:** a spot H100 node, in the zones listed in `gcp/config.sh`. Each round tries every zone for one H100 (`a3-highgpu-1g`, VM `kyuds-skycap-h100x1`), then every zone for eight (`a3-highgpu-8g`, VM `kyuds-skycap-h100`), and keeps the first VM it gets. `caffeinate` keeps the Mac awake while it waits.
  - On the 8-GPU node, vLLM still uses one GPU unless step 3 passes `--tensor-parallel-size`.
- **What you'll see:**
  1. a summary of what it will create, then **type `yes`** (billing starts here);
  2. one line per round while no zone has capacity (it retries every 60 s, up to 240 rounds);
  3. finally, `... is running in <zone>. Next: ...02_setup_node.sh`.
- **Only one H100, never eight:**

```bash
GCP_KINDS=h100x1 caffeinate -i bash skyrl-test/skycap-responses/gcp/01_create_vm.sh
```

- **Also accept a B200 node** (`a4-highgpu-8g`, VM `kyuds-skycap-b200`), tried last:

```bash
GCP_KINDS="h100x1 h100 b200" caffeinate -i bash skyrl-test/skycap-responses/gcp/01_create_vm.sh
```

- **The VM already exists:**
  - running: the script says so and exits;
  - stopped or preempted: it restarts the VM in its own zone.

**Is it up?** This asks gcloud only, without ssh, so it answers within a second or two even while the VM boots. It prints, for example, `kyuds-skycap-h100x1  RUNNING  a3-highgpu-1g SPOT  us-central1-c  up since 2026-10-16 22:03`:

```bash
bash skyrl-test/skycap-responses/gcp/alive.sh
```

To keep watching, use `--watch`. It prints a line, and shows a Mac notification, each time the state changes (`STAGING` → `RUNNING`, a preemption, ...). It checks every 30 s, or every N s with `--watch N`. Leave it in a spare terminal, and press Ctrl-C to stop:

```bash
bash skyrl-test/skycap-responses/gcp/alive.sh --watch
```

---

## Step 2: set up the node (terminal A, 10–20 minutes)

`CODEX_VERSION` pins the Codex CLI. Others train on 0.153.4 (Lego-RL) and 0.154.0 (FineEnvs). Without it, you get the latest.

```bash
CODEX_VERSION=0.154.0 bash skyrl-test/skycap-responses/gcp/02_setup_node.sh
```

It runs three phases, printing a `---` header for each:

| Phase | What it does |
|---|---|
| driver | Checks NVIDIA driver 580 and the open-file limit. A fresh VM **reboots once** here (about 5 minutes) |
| storage | Builds a RAID-0 from the local SSDs at `/mnt/local_storage` for the model weights and the uv/HF caches |
| software | Installs uv. Clones SkyRL at the branch, with skyrl-test inside it. Builds the vLLM and skycap environments, then Node, Codex and skycap-viewer. Runs detached on the VM, and the script polls it every 20 s |

- **It ends with:** `node is ready. Next: ...03_serve.sh`.
- **If it is interrupted** (Ctrl-C, Wi-Fi, a closed lid), run it again. Finished phases are skipped, and the software phase keeps running on the VM.
- **If it says `FAILED`,** it prints the end of the VM's `~/skycap_setup_software.log`. Fix the cause and run it again.

---

## Step 3: start the servers (terminal A)

```bash
bash skyrl-test/skycap-responses/gcp/03_serve.sh
```

It starts these on the VM, detached:

| Server | Port | Notes |
|---|---|---|
| vLLM | 8000 | `Qwen/Qwen3-8B`, `--max-model-len 32768`, `--enable-scale-out`, and Qwen3's reasoning and tool parsers (`SKYCAP_ENGINE_PARSERS`), which only its `/v1` routes use: text mode's upstream |
| skycap, token mode | 8080 | Records to `records/port8080` |
| skycap, text mode | 8090 | Forwards to vLLM's `/v1/responses`, records to `records/text-8090` |
| skycap-viewer | 8765 | Reads `records/` |

- **The first start downloads the model,** which takes a few minutes. It prints a waiting line every minute.
- **It ends with** one `skycap :<port> {...}` line per skycap server, showing its `/healthz`.
- **A VM set up before text mode** has vLLM without the parsers and no `:8090`. Pull the kit (step 2), then run `03_serve.sh --restart`.
- **A VM served before 2026-10-22** also runs `:8081` and `:8091`, with skycap's `--length-as-completed`, which skycap no longer has. `03_serve.sh --restart` after pulling replaces them with the two above.
- **It skips anything already up,** so it is safe to run again.

**A different model or context length:** pass `SKYCAP_MODEL`, `SKYCAP_MAX_MODEL_LEN` or `SKYCAP_ENGINE_ARGS`. Use `--restart` when servers are already running:

```bash
SKYCAP_MODEL=Qwen/Qwen3.5-0.8B bash skyrl-test/skycap-responses/gcp/03_serve.sh --restart
```

On an 8-GPU node, add for example `SKYCAP_ENGINE_ARGS="--tensor-parallel-size 4"`.

---

## Step 4: open the tunnel (terminal B, leave it open)

```bash
bash skyrl-test/skycap-responses/gcp/tunnel.sh
```

- **What it forwards:** the skycap servers (`:8080`, `:8090`) and the viewer (`:8765`) on your Mac go to the VM, plus `:8766` for [skycap-codex-eval](../skycap-codex-eval/README.md)'s viewer. It prints one line, then stays quiet until you press Ctrl-C.
- **Run it again after** a preemption, a reboot, or a dropped connection.
- **Check it from terminal A.** You should get JSON with `"mode": "tokens"`:

```bash
curl -s 127.0.0.1:8080/healthz
```

---

## Step 5: smoke drive (terminal A)

```bash
uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --config configs/smoke.json
```

- **What it runs:** 12 trajectories against `:8080`. They cover every scenario except `resample`, with stream on and off.
- **The model name** comes from the server's `/healthz`.
- **Each trajectory prints one line:**
  - `ok`, or
  - `CHECK` followed by what differed.
- **At the end:** `N/12 as expected`. Results are saved to `skyrl-test/skycap-responses/runs/<time>.jsonl`.
- **The exit code is nonzero** if any trajectory needs a look.

A run whose model ended a turn inside `<think>` gets the note "ended inside <think> with no answer", and its path and unbridged checks are turned off. Qwen3 sometimes writes its whole answer in its thinking and stops without `</think>`. `renderers` won't bridge from such a turn, so the next call is re-rendered and forks. This is model behaviour, the same through chat completions.

Every record is checked for:
- one path per conversation;
- no unbridged calls;
- every call recorded as `openai_responses`;
- `length` where a cut-off was forced;
- each scenario's own checks (see "Scenarios" below).

---

## Step 6: look at the records

Open **http://127.0.0.1:8765** (through the tunnel). Records show up as they are written. Each server's directory is one run: `port8080` and `text-8090`.

| Flag | What it means |
|---|---|
| `forked` | A conversation that should be one path isn't |
| `unbridged` | A call was re-rendered instead of extending the previous one |
| `truncated` | A call was recorded with `finish_reason: length`: the reply went back `incomplete` |

Filter rollouts on the `scenario` annotation, the `ok` annotation, or any parameter.

A text summary, run on the VM:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'bash ~/SkyRL/skyrl-test/skycap-responses/05_view.sh summary'
```

The same summary as JSON, to gate on `stats.forked`:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'bash ~/SkyRL/skyrl-test/skycap-responses/05_view.sh summary --json'
```

---

## Step 7: more drives (terminal A)

**The cut-off, in both modes.** Expect status `incomplete` with reason `max_output_tokens`, streamed or not, and a record that says `length`:

```bash
uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --scenario cutoff --set server=http://127.0.0.1:8080,http://127.0.0.1:8090 --set stream=false,true
```

**The wider matrix.** This covers:
- tools and Codex-shaped loops;
- stream on and off;
- replay style;
- temperature;
- each combination twice.

It also has two negative controls: a resample, and a client that drops reasoning on replay.

```bash
uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --config configs/matrix.json
```

**Anything ad hoc.** `--set key=v1,v2` adds a grid axis, and values are parsed as JSON:

```bash
uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --scenario tools --set reasoning_effort=low,high --set replay=verbatim,strip --repeats 3
```

**The list of scenarios:**

```bash
uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --list
```

### Text mode

The same drives, against the text-mode servers. skycap forwards each call to vLLM's own `/v1/responses` and records what it answered: messages only, no tokens. So nothing is bridged and the viewer has no token view. The run is `text-8090` in the viewer.

The smoke config. Expect 12/12:

```bash
uv run --isolated --with openai --with httpx python skyrl-test/skycap-responses/03_drive.py --config configs/smoke.json --server http://127.0.0.1:8090
```

The matrix works the same way with `--server http://127.0.0.1:8090`.

**How the scenarios change in text mode:**

| Scenario | Token mode | Text mode |
|---|---|---|
| `overflow` | skycap refuses before sampling (`response.failed` / 400 `context_length_exceeded`); nothing recorded | vLLM refuses with its own 400, which skycap relays and records as a failure. The note shows vLLM's message |
| `refusals` | skycap refuses `truncation: "auto"`; the other fields are ignored and answered | skycap still refuses what the dialect can't map. `truncation: "auto"` and the ignored fields go to vLLM, which decides; the note shows what it did |
| `retry`, streamed | The retry gets the first reply back | The retry samples again (2 paths unless identical): skycap relays a text-mode stream as it arrives and keeps no copy to replay (`skycap/retry.py`) |
| `resample` | 1 unbridged call | 0: text mode never bridges |
| Any streamed run | One `output_item.done` per output item is checked | Not checked. vLLM closes an empty `message` item between reasoning and tool calls that its final response leaves out; the note says so. skycap records the final response's items |

**What the first text-mode runs found (2026-10-17, vLLM 0.31.0, Qwen3-8B):**
- smoke 12/12, with the token-mode smoke still 12/12 with the parsers on;
- vLLM ends a cut-off stream with a `response.completed` event whose response says `status: "incomplete"`, where the API sends a `response.incomplete` event. skycap relays it as sent and records it as `length` either way.

---

## Step 8: real Codex (terminal A, runs on the VM)

Codex runs on the VM, where step 2 installed it, against the token-mode server (`:8080`, the default). A reply cut off at the context window goes back `incomplete`, as the API says. Codex takes that as an error, and with retries off (below) it fails the turn and exits 1; the record keeps the call as `length`.

With the default task (FizzBuzz):

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'cd ~/SkyRL && bash skyrl-test/skycap-responses/04_codex.sh'
```

With your own task (double quotes inside the single quotes):

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'cd ~/SkyRL && bash skyrl-test/skycap-responses/04_codex.sh "Write primes.py that prints the primes below 100, then run it."'
```

**If step 3 served a model other than the default,** put the same `SKYCAP_MODEL=...` (and `SKYCAP_MAX_MODEL_LEN=...`) before `bash` inside the quotes. Codex's config is written from these values.

**What it does:**
1. Creates a trajectory on `:8080`.
2. Writes a Codex config to a scratch `CODEX_HOME`:
   - a custom provider with `wire_api = "responses"`;
   - retries off;
   - a one-hour stream idle timeout;
   - the engine's context window;
   - web search, `multi_agent` and `goals` off.
3. Runs `codex exec`.
4. Finishes the trajectory.

**What it prints:**
- `paths=`, `unbridged=`, `calls=`, `finish_reasons=` and `dialects=`;
- any failures;
- the record's path.

Codex's own log and events go to the VM's `runs/codex-<time>/`, and step 9 copies them to your Mac.

To print a finished run's summary again, from the Mac through the tunnel:

```bash
bash skyrl-test/skycap-responses/04_codex.sh --summary <trajectory id>
```

Some things are known only after a real run, which is the design's "capture spike" (codex-harness.md section 5):

| Question | Where to look |
|---|---|
| Does Codex still send a non-function tool? | A 400 in `codex.log`, and a failure line |
| Do developer messages appear mid-conversation? | Forks in the viewer |
| Does `codex exec` call `/models` or another model name? | `codex.log`, and the skycap logs (see "When something goes wrong") |

---

## Step 9: keep the records, stop the VM (terminal A)

**Copy everything to your Mac.** This copies the VM's records to `skyrl-test/skycap-responses/records/<vm>/`, and its drive results and Codex runs to `runs/<vm>/`. All of these are git-ignored.

```bash
bash skyrl-test/skycap-responses/gcp/pull_records.sh
```

**Stop the VM.** The boot disk is kept (checkouts, records), and you pay for the disk only. The local SSDs (model weights, caches) are discarded. Steps 1–3 bring it back.

```bash
bash skyrl-test/skycap-responses/gcp/down.sh stop
```

**Or delete it,** with its disk and every record on it. It asks you to type the VM name.

```bash
bash skyrl-test/skycap-responses/gcp/down.sh delete
```

Press Ctrl-C in terminal B; the tunnel is gone with the VM anyway.

### Viewing pulled records on the Mac

Once, install Node 22.15+ and skycap-viewer into the kit's `.tools/`. This needs the SkyRL checkout here on `kyuds/skycap-responses`. It also installs a Codex CLI there, which is harmless.

```bash
bash skyrl-test/skycap-responses/_tools.sh
```

Then serve the pulled records. Replace `<vm>` with the VM's name, e.g. `kyuds-skycap-h100x1` or `kyuds-skycap-h100`. The viewer needs no tokenizer and no network.

```bash
SKYCAP_RECORDS=$PWD/skyrl-test/skycap-responses/records/<vm> bash skyrl-test/skycap-responses/05_view.sh
```

---

## When something goes wrong

**Is the VM alive?** `gcp/alive.sh` gives one line from gcloud. For a stopped VM, it also says whether the stop was a preemption. Add `--watch` to be told when the state changes (see step 1).

**Where things stand on the VM:** GPUs, disks, vLLM, both skycap servers, the records and the newest drive. This one connects over ssh:

```bash
bash skyrl-test/skycap-responses/gcp/status.sh
```

**Preempted** (`alive.sh` says `PREEMPTED`, or the tunnel drops):
- The boot disk survives; the weights and caches do not.
- Run steps 1, 2 and 3 again, then reopen the tunnel (step 4).
  - Step 2 rebuilds the RAID and the environments, because the uv cache lived on the local SSDs.
  - The model downloads again in step 3.

**`gcloud` errors about reauthentication:** run `gcloud auth login`. The VM is not affected.

**A server is down or misbehaving:** read its log on the VM. Swap `engine.log` for `skycap-<port>.log` (8080, 8090) or `viewer.log`:

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh --command 'tail -n 50 ~/SkyRL/skyrl-test/skycap-responses/runs/engine.log'
```

Restart all of them:

```bash
bash skyrl-test/skycap-responses/gcp/03_serve.sh --restart
```

Stop them without stopping the VM:

```bash
bash skyrl-test/skycap-responses/gcp/03_serve.sh --stop
```

**A shell on the VM.** An interactive shell loads `~/.skycap_env` (uv on PATH) through `.bashrc`.

```bash
bash skyrl-test/skycap-responses/gcp/ssh.sh
```

**Picking up new commits:**
1. Push them.
2. Run step 2 again: the software phase pulls both branches.
3. Run `03_serve.sh --restart`.

---

## Reference

### Files

| File | What it does |
|---|---|
| `gcp/config.sh` | Settings: kinds and zones, image, disk, network, repos and branches, VM name. Each one can be overridden from the environment |
| `gcp/01_create_vm.sh` | Step 1. Creates the spot VM, retrying until there is capacity, or restarts it after a preemption |
| `gcp/02_setup_node.sh` | Step 2. Driver, local SSDs, checkouts, environments, Node, Codex, viewer |
| `gcp/03_serve.sh` | Step 3. vLLM, skycap `:8080` (token mode) and `:8090` (text mode), and the viewer, detached on the VM. Also takes `--restart` and `--stop` |
| `gcp/tunnel.sh` | Step 4. Forwards `:8080`, `:8090`, `:8765` and `:8766` (skycap-codex-eval) to the Mac |
| `gcp/alive.sh` | Whether the VM is up, from gcloud alone (no ssh). `--watch` reports each change |
| `gcp/ssh.sh`, `gcp/status.sh` | A shell or a command on the VM; one screen of state |
| `gcp/pull_records.sh`, `gcp/down.sh` | Step 9 |
| `gcp/node_setup.sh`, `gcp/vm_startup.sh` | Run on the VM. The other scripts copy and call them |
| `00_env.sh` | Sourced by the scripts below. Finds SkyRL and exports the knobs: `SKYCAP_MODEL` (default `Qwen/Qwen3-8B`), `SKYCAP_MAX_MODEL_LEN` (32768), the ports, `SKYCAP_RECORDS` |
| `01_engine.sh` | Runs `vllm serve $SKYCAP_MODEL --enable-scale-out`, which serves `/inference/v1/generate`, the route token mode calls, with Qwen3's reasoning and tool parsers for text mode's upstream (`SKYCAP_ENGINE_PARSERS`; empty for none) |
| `02_skycap.sh` | Runs `skycap serve --mode tokens`, recording to `records/port<port>`; with `SKYCAP_MODE=text`, `--mode text` forwarding to vLLM's `/v1`, recording to `records/text-<port>` |
| `03_drive.py`, `scenarios.py`, `configs/` | The driver, its scenarios, and parameter grids (`smoke.json`, `matrix.json`) |
| `04_codex.sh` | Real Codex (`codex exec`) through one trajectory |
| `05_view.sh` | skycap-viewer on `records/`, or its `summary` |
| `_tools.sh` | Installs Node 22, the Codex CLI and skycap-viewer into `.tools/` |

`records/`, `runs/` and `.tools/` are git-ignored.

### Parameters

Any key in a config's `defaults`, `params` or `grid`, or in `--set`, becomes a run parameter. Each run records its parameters as trajectory annotations, so skycap-viewer can filter on them.

| Parameter | Effect |
|---|---|
| `stream` | `true` streams. The driver reads the final event and checks for one `output_item.done` per item |
| `replay` | How replies go back: `verbatim`, `strip` (no ids or statuses), or `drop_reasoning`. `drop_reasoning` is a lossy client: forks are expected, so path checks are off |
| `temperature`, `top_p`, `max_output_tokens`, `instructions`, `tool_choice`, `parallel_tool_calls` | Sent as the Responses fields. Token mode ignores `tool_choice` and `parallel_tool_calls`, as its chat path does |
| `reasoning_effort` | Sent as `reasoning.effort` (ignored in token mode) |
| `extra_body`, `headers` | Any other request fields or headers, as objects |
| `server` | Which skycap server (default `$SKYCAP_URL`, else `http://127.0.0.1:8080`) |
| `model` | Default: `$SKYCAP_MODEL`, else the server's tokenizer |
| `steps` (codex_like), `cutoff_tokens` (cutoff), `overflow_words` (overflow) | Scenario-specific |

A config is JSON:
- `defaults`: parameters every run starts from;
- `repeats`;
- `runs`: a list. Each entry has a `scenario`, plus optional `params`, `grid` (every combination is its own trajectory) and `repeats`.

See `configs/` for examples.

### Scenarios

To add a scenario:
1. Write an async function in `scenarios.py`, decorated with `@scenario`.
2. Set its expectations on the session: `expect_paths`, `expect_unbridged`, `expect_finish_reasons`, `check(...)`.
3. Name it in a config, or run it with `--scenario`.

| Scenario | What it checks |
|---|---|
| `chat` | Three turns stay one path, and every call is bridged |
| `tools` | A tool loop (`get_time`, `add`) plus a follow-up: one path, all bridged |
| `codex_like` | Codex's request shape: `instructions`, two `developer` messages, an environment message, and `exec_command` / `read_file` over several steps. Qwen3's template has no `developer` role, so this exercises the developer fold |
| `cutoff` | `max_output_tokens` cuts the first answer, and the conversation goes on. The reply is `incomplete` with reason `max_output_tokens`, as the API says, and the record says `length`. Paths aren't checked, because a turn cut inside its thinking may not bridge ([#2462](https://github.com/NovaSky-AI/SkyRL/pull/2462)) |
| `resample` | The same request twice, with no retry marker: two samples, so two paths unless they came out identical. The second call counts as one unbridged call: skycap checks every call after a trajectory's first, and this one has no model turn to extend |
| `retry` | The same request, the second marked `x-stainless-retry-count: 1`: the same response comes back, and one call is recorded |
| `overflow` | A prompt past the context window. Streamed: `response.failed` / `context_length_exceeded`. Not streamed: HTTP 400. Nothing is recorded |
| `refusals` | 400s for `previous_response_id`, `custom` and `web_search` tools, `truncation: auto`, `item_reference` and encrypted reasoning. Answers despite `tool_choice: required`, `parallel_tool_calls: false`, a JSON schema, `reasoning.effort: none` and logprobs |

### Without GCP (an Anyscale workspace, or any GPU node)

The same scripts run directly on a node, from the SkyRL checkout root, with this repo cloned inside it.

Steps:
1. Check out `kyuds/skycap-responses`.
2. Run `CODEX_VERSION=0.154.0 bash skyrl-test/skycap-responses/_tools.sh`.
3. Start the servers:
   - `nohup bash skyrl-test/skycap-responses/01_engine.sh > skyrl-test/skycap-responses/runs/engine.log 2>&1 < /dev/null &`
   - `02_skycap.sh` the same way, for token mode on `:8080`;
   - and with `SKYCAP_PORT=8090 SKYCAP_MODE=text`, for text mode.
4. Run steps 5–8, minus the `gcp/ssh.sh --command` wrapper.
5. For the viewer, run `05_view.sh` and forward port 8765.

On Anyscale, Ray already holds the GPUs in its ledger. This vLLM is a plain process on one GPU, which is fine while nothing else runs there.

### What is different from Charlie's guide

`gcp/` follows [Charlie's guide](https://gist.github.com/CharlieFRuan/6eeae93d70ede0e81f12f91a4eb74d57), cut down to one node. It is adapted from `opd-gcp-spot-full-repro/gcp/`, which ran unchanged on a real node. These parts are kept:
- spot creation with retries across zones;
- the image family that replaced the guide's retired one;
- the driver 580 check and the open-file limit;
- the NVMe RAID;
- restart after a preemption.

**What's changed:**
- **The default node is one H100,** since vLLM serving one 8B model needs only one GPU. An 8×H100 node is the fallback.
- **No Ray.**
- **No secrets.** Nothing pushes W&B or HF tokens.
- **A serve step.**
- **The driver runs from your Mac,** through the tunnel.

### Assumptions

Not yet verified on a node:
- GCP offers `a3-highgpu-1g` as spot in the zones `gcp/config.sh` lists for `a3-highgpu-8g`. A zone where it isn't offered is dropped after one refused request.
- vLLM 0.31.0 (SkyRL's pin) keeps `--enable-scale-out`. This was read in 0.30.0.
- `codex exec` at the version you pin accepts:
  - `--disable multi_agent --disable goals`;
  - top-level `web_search = "disabled"`.
  
  Lego-RL found that `[tools] web_search = false` didn't remove web search on 0.153.4.
