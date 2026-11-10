# Data

Four kinds of data, all written to the boot disk under `~/opd-store/data/v2/`, so they survive a preemption:

- **M2RL's RL prompts** (arXiv 2602.12566), NVIDIA's `Nemotron-3-Nano-RL-Training-Blend` split by domain. Sections 2
  and 3 distill over them.
- **DeepMath-103K**, the prompts of Thinking Machines' OPD. Section 1 distills over them.
- **AIME 2024 and 2025**, the in-training eval of every OPD run.
- **SFT data, pretokenized:** OpenThoughts-3 for 1a, and Qwen3-32B's answers for 2c's warm-up.

You do not need to run anything here. Every job prepares what it needs and skips what is already there. By hand on a
node, from `~/SkyRL`:

```bash
bash skyrl-test/opd-blog-v2/data/prepare.sh
```

The first downloads about 8 GB and takes 10 to 20 minutes. 1a's SFT data is written by `1-tml/sft.sh` itself, from
the settings in its `HYPERPARAMETERS` block: it downloads OpenThoughts-3 (28 GB) and tokenizes 384,000 rows of
every domain; only b200-1 needs it.

## What it writes

| File | Rows | What | Used by |
|---|---|---|---|
| `math_train.parquet` | 22,056 | M2RL's math set: DAPO-17k and Skywork-OR1 without OmniMath | 2a to 2d, 2c's traces |
| `mopd_train.parquet` | 57,800 | Math above, plus competitive coding (19,169) and instruction following (16,575) | MOPD |
| `deepmath_train.parquet` | about 103,000 | DeepMath-103K: the question alone, graded against `final_answer` | 1b |
| `aime24.parquet`, `aime25.parquet` | 30 each | AIME 2024 and 2025, M2RL's prompt | Every OPD run's eval, and `0-evals` |
| `sft_ot3_all_384000/` | 384,000 | OpenThoughts-3, every domain, pretokenized | 1a |
| `sft_2c_qwen3_32b_traces/` | up to 22,056 | Qwen3-32B's correct answers, pretokenized | 2c's warm-up |

Each SFT folder has a `<folder>.stats.json` beside it: rows, finished and unfinished answers, rows cut at the length
limit, tokens. It sits outside the folder because SkyRL's SFT loader reads every `.json` inside as data.

Science, agent and structured-output rows of the blend are left out: no job uses them. SkyRL drops prompts over
2,048 tokens, as M2RL did; in a 1,112-row slice of the blend that was 4 rows.

## The RL prompts

- **Math (M2RL):** `Solve the following math problem step by step. The last line of your response should be of the
  form Answer: $Answer (without quotes) where $Answer is the answer to the problem.`, a blank line, then the blend's
  question, which already asks for `\boxed{}`.
- **Code and IF (M2RL):** the blend's prompt, unchanged.
- **AIME (M2RL):** the math instruction with `Answer: \boxed{$Answer}`, then the problem.
- **DeepMath (Thinking Machines):** the `question` column alone, as their cookbook's `distillation/datasets.py` sends it.

The blend stores its DAPO and Skywork rows as pointers; `prepare_data.py` fills them the way the blend's own
`create_nanov3_jsonl.py` does, and stops if the counts differ from the paper's.

## The SFT data

`prepare_sft.py` tokenizes each row as the OPD student will see it, with a post-trained Qwen3 tokenizer:

- prompt, no loss: `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n`
- answer, loss: `<think>\n{reasoning}\n</think>\n\n{final answer}<|im_end|>`

**OpenThoughts-3** follows Thinking Machines' cookbook at `330b73d8d7` (2025-10-27, the blog's version): the set
shuffled, the first 128 × 3,000 = 384,000 rows of every domain (850,000 math, 250,000 code and 100,000 science rows in
the whole set, per its [dataset card][ot3]), cut at 16,384 tokens. The shuffle is seeded here instead of their
streaming buffer, so the rows are not theirs.

OpenThoughts-3 keeps QwQ-32B's answers that ran out of tokens before closing their think block: 62% of one code
shard, 68% of one math shard and 12% of one science shard (checked 2026-11-09). The cookbook trains on them as they
are, so they stay, as `<think>\n{reasoning so far}` with no `</think>` and no end token. The model is never taught to
stop mid-thought. In a 300-row sample, 148 rows were finished, 13 finished but cut, 139 unfinished (all cut at 16,384),
and the mean was 13,131 tokens, so 384,000 rows are about 5 billion tokens.

**2c's traces** keep only Qwen3-32B's correct answers that finished within 16,384 tokens (`2-alignment/README.md`).

## Sources, pinned

| Source | Revision |
|---|---|
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | `74d18444` |
| `BytedTsinghua-SIA/DAPO-Math-17k` | `65877096` |
| `Skywork/Skywork-OR1-RL-Data` (math) | `1cdedc52` |
| `zwhe99/DeepMath-103K` | `5cf055d1` |
| `open-thoughts/OpenThoughts3-1.2M` | `61bcf9d4` |
| `math-ai/aime24` | `83a7f387` |
| `math-ai/aime25` | `563bb840` |

## How math answers are graded

`common/kit_main.py` registers two environments.

- **`opd_v2_math`, math rows:** reward 1 or 0. It takes the answer after the last `</think>`, from the last
  `\boxed{}` or else the last `Answer:` line, and compares it with math-verify. An answer whose thinking never closes
  was cut off and scores 0. Under OPD the reward is only logged, as a train-time accuracy.
- **`opd_v2_noreward`, code and IF rows:** reward 0. OPD learns from its teacher alone, so MOPD needs no sandbox and no
  IF checker to train.

SkyRL pins ANTLR 4.9.3 (through omegaconf), and math-verify's parser for that version leaves percentages unparsed.
The grader drops `%` from both sides, so `20` matches `20\%`, but `0.2` does not. `0-evals` grades math with the same
function.

## Verified and assumed

Verified on 2026-11-09, on the Mac:

- `prepare_data.py` on the first 80 MB of the real blend: 444 math, 343 code and 325 IF rows, M2RL's prompts.
- `prepare_sft.py` on 300 OpenThoughts-3 rows and on a 2-row traces file: the rows decode to the layout above, an
  unfinished answer has no end token, a finished one ends in `<|im_end|>`, and SkyRL's own `load_from_pretokenized`
  reads the folder (300 rows, loss on the answer only).

Not verified: the full blend's counts and DeepMath's row count (both checked on the first real run), and the time
and memory of tokenizing 384,000 rows on a node.

[ot3]: https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M
