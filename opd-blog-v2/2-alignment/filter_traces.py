"""Keeps the rows of a pretokenized SFT store (data/prepare_sft.py's output) whose answer is at most N tokens.

The answer is the row's loss-masked tokens: the model's reply after the prompt, as the SFT trains on it. Writes the
kept rows in prepare_sft.py's layout (part-*.parquet in <out>, <out>.stats.json beside it), so sft_2c.sh reads it as it
reads the full store. Run from ~/SkyRL:
  uv run --isolated --extra fsdp python <kit>/2-alignment/filter_traces.py --data <store> --max-answer-tokens 8192 \
    --out <store>_answers_le8192
"""

import argparse
import glob
import json
import os
import shutil

import datasets


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="a pretokenized store: a folder of part-*.parquet")
    p.add_argument("--max-answer-tokens", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-proc", type=int, default=8)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {args.data}")
    ds = datasets.Dataset.from_parquet(files)
    ds = ds.map(lambda b: {"n_answer": [sum(m) for m in b["loss_mask"]], "n_tokens": [len(i) for i in b["input_ids"]]},
                batched=True, num_proc=args.num_proc)
    kept = ds.filter(lambda b: [n <= args.max_answer_tokens for n in b["n_answer"]], batched=True,
                     num_proc=args.num_proc)
    if len(kept) == 0:
        raise SystemExit(f"no row of {args.data} has an answer of at most {args.max_answer_tokens} tokens")
    tokens, answer_tokens = sum(kept["n_tokens"]), sum(kept["n_answer"])
    kept = kept.remove_columns(["n_answer", "n_tokens"])

    tmp = args.out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n_shards = max(1, len(kept) // 8192)
    for i in range(n_shards):
        kept.shard(n_shards, i, contiguous=True).to_parquet(os.path.join(tmp, f"part-{i:05d}.parquet"))
    source = {}
    if os.path.exists(args.data + ".stats.json"):
        source = json.load(open(args.data + ".stats.json"))
    stats = {
        "source": f"{source.get('source', 'traces')}, answers of at most {args.max_answer_tokens} tokens",
        "tokenizer": source.get("tokenizer"),
        "rows": len(kept),
        "rows_before": len(ds),
        "max_answer_tokens": args.max_answer_tokens,
        "tokens": tokens,
        "mean_tokens": round(tokens / len(kept)),
        "mean_answer_tokens": round(answer_tokens / len(kept)),
        "max_length": source.get("max_length"),
        "filtered_from": args.data,
    }
    shutil.rmtree(args.out, ignore_errors=True)
    os.replace(tmp, args.out)
    # Beside the folder, not in it: the SFT loader reads every .json under the folder as data.
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
