"""Convert Nemotron-Cascade-RL-Math to the DAPO AIME format, so the `aime` environment grades it.

Writes one parquet with the same columns and prompt template as `aime-2024-cleaned.parquet`. Rows the grader cannot
mark correct even when given the exact answer are dropped: 15 of the 14,476, whose answers contain "<" (the grader's
`Answer:` pattern stops there). Every other row is kept, including the 18% with answers that are not integers. Those are
compared as strings after light normalization, so `9/2` does not match `\\frac{9}{2}`, and some correct rollouts on them
score as wrong.

Run from the SkyRL checkout, in its environment (skyrl_gym has the grader):
    uv run --isolated --extra fsdp python <this file> --output <file.parquet>
"""

import argparse
import json
import os
import urllib.request

import pandas as pd
from skyrl_gym.envs.aime.utils import compute_score

# Pinned, so every node trains on the same rows.
URL = (
    "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-Math/resolve/"
    "fcd07b1417edc12cf4642cae7d269e4f3a84c812/math.json"
)
EXPECTED_ROWS = 14476
EXPECTED_DROPPED = 15
DATA_SOURCE = "nemotron-cascade-rl-math"

# SkyRL's examples/train/algorithms/dapo/data_preprocess_aime2025.py, the AIME-2024 parquet's own template.
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def to_rows(records):
    """Returns the rows in the DAPO format, and how many records were dropped."""
    rows = []
    dropped = 0
    for i, record in enumerate(records):
        problem = record["problem"]
        answer = str(record["answer"]).strip()
        if not compute_score(f"Answer: {answer}", answer)["acc"]:
            dropped += 1
            continue
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"content": PROMPT_PREFIX + problem + PROMPT_SUFFIX, "role": "user"}],
                "ability": "MATH",
                "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
                "extra_info": {"index": i, "raw_problem": problem, "source": record["source"], "split": "train"},
            }
        )
    return rows, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="The parquet file to write")
    parser.add_argument("--input-file", default=None, help="A local copy of math.json, instead of downloading it")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            records = json.load(f)
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "skyrl-test"})
        with urllib.request.urlopen(request, timeout=300) as response:
            records = json.load(response)
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} problems, got {len(records)}")

    rows, dropped = to_rows(records)
    if dropped != EXPECTED_DROPPED:
        raise ValueError(f"Expected to drop {EXPECTED_DROPPED} ungradable rows, dropped {dropped}")

    # Written in one step, so a preemption cannot leave a partial file that looks finished.
    partial = f"{args.output}.partial"
    pd.DataFrame(rows).to_parquet(partial)
    os.replace(partial, args.output)
    print(f"Wrote {len(rows)} rows to {args.output} ({dropped} ungradable rows dropped)")


if __name__ == "__main__":
    main()
