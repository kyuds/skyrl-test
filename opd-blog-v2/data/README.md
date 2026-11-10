# Data

Every run uses the same data: M2RL's RL data, from arXiv 2602.12566 ("To Mix or To Merge", COLM 2026). It is NVIDIA's
`Nemotron-3-Nano-RL-Training-Blend` split by domain, so every teacher in this blog is trained on, and distills over,
the prompts its domain's M2RL expert was RL-trained on.

You do not need to run anything here: every run script calls `prepare.sh` first, and it skips what is already there.
To do it by hand on a node, from `~/SkyRL`:

```bash
bash skyrl-test/opd-blog-v2/data/prepare.sh
```

The first call downloads about 8 GB and takes 10 to 20 minutes. Everything is
written to the boot disk, so it survives a preemption.

## What it writes

In `~/opd-store/data/v2/`:

| File | Rows | What | Used by |
|---|---|---|---|
| `math_train.parquet` | 22,056 | M2RL's math set: DAPO-17k and Skywork-OR1 without OmniMath | GRPO teacher, phases 1 and 2 |
| `mopd_train.parquet` | 57,800 | Math above, plus competitive coding (19,169) and instruction following (16,575) | Phase 3 |
| `aime24.parquet`, `aime25.parquet` | 30 each | AIME 2024 and 2025 | Eval for every run |

Science, agent and structured-output rows of the blend are left out: no run uses them.

Each row has `prompt` (a chat), `env_class`, `reward_model.ground_truth`, `data_source`, `skyrl_mopd_domain`
(`math`, `code` or `if`: the teacher that scores the row in phase 3) and `extra_info` (its line in the blend and
NVIDIA's pass rate). `prepare_data.py` stops if the counts differ from the paper's, so a changed blend cannot slip in.

SkyRL drops prompts over 2,048 tokens, as M2RL did. In a 1,112-row slice of the blend that was 4 rows: 3 code, 1 IF.

## The prompts

Exactly M2RL's (`data/process_rl_data.py` and `data/download_aime.py` in github.com/Mosi-AI/M2RL):

- **Math:** `Solve the following math problem step by step. The last line of your response should be of the form
  Answer: $Answer (without quotes) where $Answer is the answer to the problem.` then a blank line and the blend's
  question. The question already ends with `Don't forget to put your answer in \boxed{}.` (DAPO rows) or
  `Remember to put your final answer inside \boxed{}.` (Skywork rows), so the prompt asks for both forms.
- **Code and IF:** the blend's prompt, unchanged. Code prompts ask for one Python block; IF prompts start with
  `Please follow these instructions in your response:`.
- **AIME:** the math instruction with `Answer: \boxed{$Answer}`, then the problem.

The blend stores its DAPO and Skywork rows as pointers. `prepare_data.py` fills them from the source datasets the way
the blend's own `create_nanov3_jsonl.py` does, line for line.

Every source is pinned to a revision, so every node writes the same rows:

| Source | Revision |
|---|---|
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | `74d18444` |
| `BytedTsinghua-SIA/DAPO-Math-17k` | `65877096` |
| `Skywork/Skywork-OR1-RL-Data` (math) | `1cdedc52` |
| `math-ai/aime24` | `83a7f387` |
| `math-ai/aime25` | `563bb840` |

## How answers are graded

`common/kit_main.py` registers two environments.

- **`opd_v2_math`, math rows:** reward 1 or 0. It takes the answer after the last `</think>`, from the last
  `\boxed{}` or else the last `Answer:` line, and compares it with math-verify. An answer whose thinking never closes was
  cut off and scores 0. The reward trains the GRPO teacher; under OPD it is only logged.
- **`opd_v2_noreward`, code and IF rows:** reward 0. OPD learns from its teacher alone, so phase 3 needs no sandbox
  and no IF checker to train.

SkyRL's own `aime` env would not do: it compares strings, which suits DAPO's integer answers but not Skywork's. 125
of 444 sampled math answers are not integers, like `\frac{13}{4}`, and a correct `3.25` would score 0.

One limit: SkyRL pins ANTLR 4.9.3 (through omegaconf), and math-verify's parser for that version leaves percentages
unparsed. The grader drops `%` from both sides, so `20` matches `20\%`, but `0.2` does not.

## Chat templates

Nothing is changed. An OPD teacher scores the student's exact token ids, so a teacher whose chat template differed
from the student's would be scoring a format it never produced. M2RL-SFT, the three M2RL experts and Qwen3-32B share
one template, byte for byte (checked on 2026-11-09), so every M2RL run uses `Jackwang111/M2RL-SFT` as published. The
Qwen3-8B runs use Qwen3-8B's template with `enable_thinking: false` for student and teacher alike.

## Verified and assumed

Verified on 2026-11-09, on the Mac:

- `prepare_data.py` on the first 80 MB of the real blend, with the pinned DAPO and Skywork sources: 444 math, 343 code
  and 325 IF rows. The prompts match M2RL's processing, and AIME has 30 problems per year with integer answers.
- SkyRL's own dataset loader reads `mopd_train.parquet`: each row keeps its `env_class`, and `skyrl_mopd_domain`
  reaches the environment's extras, where MOPD's router reads it. 4 of 1,112 prompts were over 2,048 tokens.
- The grader: 15 cases, run off the main thread as SkyRL's generator does, with math-verify 0.9.0 and ANTLR 4.9.3:
  fractions, decimals, `\dfrac`, percentages, matrices, expressions, unclosed thinking, and answers before `</think>`
  ignored. All 125 sampled non-integer answers parse symbolically.

Not verified: the full blend's counts. The script checks them on the first real run.
