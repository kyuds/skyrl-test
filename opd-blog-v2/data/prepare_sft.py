"""Writes pretokenized SFT data for SkyRL's SFT trainer: parquet shards whose rows are ``input_ids`` and a 0/1
``loss_mask`` (skyrl/train/dataset/pretokenized.py).

    cd ~/SkyRL && uv run --isolated --extra fsdp python skyrl-test/opd-blog-v2/data/prepare_sft.py ot3 --out <dir>
    cd ~/SkyRL && uv run --isolated --extra fsdp python skyrl-test/opd-blog-v2/data/prepare_sft.py traces \\
        --traces <answers.jsonl> --tokenizer Jackwang111/M2RL-SFT --out <dir>

Sources:
  ot3     OpenThoughts-3 (every answer written by QwQ-32B), the data of Thinking Machines' SFT: their cookbook's
          off_policy_reasoning.py at 330b73d8d7 (2025-10-27, the blog's version) shuffles the whole set, takes 3000
          steps x 128 = 384,000 rows of every domain (math, code, science), and truncates at 16,384 tokens. Same here,
          with a fixed seed instead of their streaming shuffle buffer, so the rows are not theirs. --domains keeps
          some domains only (1-tml/sft.sh's OT3_DOMAINS; empty, as the cookbook, by default).
  traces  section 2c: Qwen3-32B's correct answers (2-alignment/traces.sh writes the jsonl: question, response).

Every row is one user turn and one assistant turn, tokenized exactly as the OPD student sees it:
  prompt  apply_chat_template([user], add_generation_prompt=True, enable_thinking=True)
          = <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n           loss_mask 0
  answer  <think>\n{reasoning}\n</think>\n\n{final answer}<|im_end|>               loss_mask 1
A row is cut at --max-length tokens, prompt included, as the cookbook does.
Unfinished answers: OpenThoughts-3 keeps QwQ-32B's answers that ran out of tokens before closing their think block
(checked 2026-11-09 on three shards: 62% of a code shard, 68% of a math shard, 12% of a science shard). The cookbook
trains on them as they are, so they stay: "<think>\n{reasoning so far}" with no </think> and no end token, so the
model is never taught to stop mid-thought. --finished-only drops them instead (the traces source always has it on).
Writes how many rows were finished, unfinished, cut and dropped to <out>.stats.json.
"""

import argparse
import json
import os
import shutil
from collections import Counter

OT3 = ("open-thoughts/OpenThoughts3-1.2M", "61bcf9d4eb38b30295efc2021227a63cc5bb34c8", 120)  # repo, revision, shards


def split_answer(text: str):
    """(body, finished): the answer in Qwen3's layout, and whether it closed its think block. ``(None, False)`` for an
    empty one. An unfinished body is the reasoning so far, without </think>."""
    text = text or ""
    if "</think>" not in text:
        reasoning = text.replace("<think>", "", 1).strip("\n ")
        return (f"<think>\n{reasoning}" if reasoning else None), False
    reasoning, final = text.split("</think>", 1)
    reasoning = reasoning.replace("<think>", "", 1).strip("\n ")
    final = final.strip("\n ")
    if not reasoning or not final:
        return None, False
    return f"<think>\n{reasoning}\n</think>\n\n{final}", True


def make_encoder(tokenizer_name: str, max_length: int, finished_only: bool):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    assert end == 151645, f"{tokenizer_name} has no <|im_end|> at 151645"
    dropped = {"input_ids": [], "loss_mask": [], "status": "dropped", "n_tokens": 0}

    def encode(question: str, answer: str):
        body, finished = split_answer(answer)
        if not question or body is None or (finished_only and not finished):
            return dropped
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True, enable_thinking=True, tokenize=False
        )
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tok(body, add_special_tokens=False)["input_ids"] + ([end] if finished else [])
        ids = prompt_ids + answer_ids
        mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
        status = "finished" if finished else "unfinished"
        if len(ids) > max_length:
            ids, mask, status = ids[:max_length], mask[:max_length], status + "_cut"
        if sum(mask) == 0:
            return dropped
        return {"input_ids": ids, "loss_mask": mask, "status": status, "n_tokens": len(ids)}

    return encode


def load_ot3(rows: int, seed: int, max_shards: int, domains):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    repo, revision, shards = OT3
    patterns = [f"data/train-{i:05d}-of-{shards:05d}.parquet" for i in range(min(max_shards, shards))]
    path = snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=patterns)
    files = [os.path.join(path, pattern) for pattern in patterns]  # not a glob: the snapshot may hold other shards
    ds = load_dataset("parquet", data_files=files, split="train")
    print(f"OpenThoughts-3: {len(ds)} rows in {len(files)} shards")
    if domains:
        ds = ds.filter(lambda r: r["domain"] in domains, num_proc=16)
        print(f"kept the {len(ds)} rows of {sorted(domains)}")
    ds = ds.shuffle(seed=seed).select(range(min(rows, len(ds)))).flatten_indices()

    def pair(row):
        turns = row["conversations"]
        question = next((t["value"] for t in turns if t["from"] == "human"), "")
        answer = next((t["value"] for t in turns if t["from"] == "gpt"), "")
        return {"question": question, "answer": answer}

    return ds.map(pair, remove_columns=[c for c in ds.column_names if c != "domain"], num_proc=16)


def load_traces(path: str):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    return ds.map(lambda r: {"question": r["question"], "answer": r["response"]}, remove_columns=ds.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["ot3", "traces"])
    parser.add_argument("--out", required=True, help="output folder of parquet shards")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="a post-trained Qwen3 tokenizer (they are all one)")
    parser.add_argument("--rows", type=int, default=384_000, help="ot3: rows to keep after the shuffle")
    parser.add_argument("--max-length", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=OT3[2], help="ot3: read only the first N shards (testing)")
    parser.add_argument("--traces", help="traces: the jsonl 2-alignment/traces.sh wrote")
    parser.add_argument("--finished-only", action="store_true", help="drop answers that never close their think block")
    parser.add_argument("--domains", default="", help="ot3: comma-separated domains to keep (math, code, science); empty keeps all, as the cookbook")
    parser.add_argument("--num-proc", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    out = os.path.expanduser(args.out)

    if args.source == "ot3":
        ds = load_ot3(args.rows, args.seed, args.max_shards, {d for d in args.domains.split(",") if d})
    else:
        if not args.traces:
            raise SystemExit("traces needs --traces <jsonl>")
        ds = load_traces(args.traces)
    domains = Counter(ds["domain"]) if "domain" in ds.column_names else None

    encode = make_encoder(args.tokenizer, args.max_length, args.finished_only or args.source == "traces")
    ds = ds.map(lambda r: encode(r["question"], r["answer"]), remove_columns=ds.column_names, num_proc=args.num_proc)
    status = Counter(ds["status"])
    tokens = sum(ds["n_tokens"])
    ds = ds.filter(lambda r: r["status"] != "dropped", num_proc=args.num_proc).remove_columns(["status", "n_tokens"])

    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n_shards = max(1, len(ds) // 8192)
    for i in range(n_shards):
        ds.shard(n_shards, i, contiguous=True).to_parquet(os.path.join(tmp, f"part-{i:05d}.parquet"))
    stats = {
        "source": args.source,
        "tokenizer": args.tokenizer,
        "rows": len(ds),
        "status": dict(status),
        "tokens": tokens,
        "mean_tokens": round(tokens / max(1, len(ds))),
        "max_length": args.max_length,
        "domains": dict(domains) if domains else None,
        "seed": args.seed,
    }
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    # Beside the folder, not in it: the SFT loader reads every .json under the folder as data.
    with open(out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
