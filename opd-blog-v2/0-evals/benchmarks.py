"""Builds an eval's requests: what to send the model (requests.jsonl, read by generate.py) and what the grader needs
(grading.jsonl, read by grade.py), one row each per problem, joined on ``id``.

    python 0-evals/benchmarks.py bench --benches aime24,aime25 --data ~/opd-store/data/v2 --tools ~/opd-store/tools \\
        --out <dir>
    python 0-evals/benchmarks.py prompts --parquet ~/opd-store/data/v2/math_train.parquet --n 20000 --samples 2 --out <dir>

Benchmarks, and samples per problem as M2RL reports them (arXiv 2602.12566, appendix, "Evaluation settings": Avg@64
for AIME'24 and '25, Avg@8 for LCB v5, Avg@1 for LCB v6, Avg@4 for IFEval, Avg@1 for IFBench):
  aime24, aime25  M2RL's prompts, from data/prepare_data.py (the kit's in-training eval set)
  lcb_v5, lcb_v6  LiveCodeBench code generation, problems dated [2024-08-01, 2025-02-01) and [2025-02-01, 2025-05-01).
                  ASSUMED: M2RL names the versions, not the dates; these are the windows Qwen3 and most 2025 reports use.
                  The prompt is LiveCodeBench's own generic one (lcb_runner/prompts/code_generation.py), as one user turn.
  ifeval          google/IFEval, 541 prompts, sent as they are
  ifbench         allenai/IFBench's test set (the repo's data/IFBench_test.jsonl), sent as they are
``prompts`` instead turns RL rows (a ``prompt`` chat and ``reward_model.ground_truth``) into math requests: how
2-alignment/traces.sh asks Qwen3-8B for 2c's warm-up answers.
"""

import argparse
import base64
import json
import os
import pickle
import random
import zlib
from typing import Dict, Iterator, List, Tuple

# Samples per problem, Avg@K: M2RL's (https://arxiv.org/abs/2602.12566, appendix A, "Evaluation"). AIME's 64 is also the
# Qwen3 report's (https://arxiv.org/abs/2505.09388, section 4.6: "For each question, we sample 64 times").
BENCH_SAMPLES = {"aime24": 64, "aime25": 64, "lcb_v5": 8, "lcb_v6": 1, "ifeval": 4, "ifbench": 1}
BENCH_SETS = {
    "math": ["aime24", "aime25"],
    "code": ["lcb_v5", "lcb_v6"],
    "if": ["ifeval", "ifbench"],
    "all": ["aime24", "aime25", "lcb_v5", "lcb_v6", "ifeval", "ifbench"],
}
LCB = ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", ["test4.jsonl", "test5.jsonl", "test6.jsonl"])
LCB_WINDOWS = {"lcb_v5": ("2024-08-01", "2025-02-01"), "lcb_v6": ("2025-02-01", "2025-05-01")}
IFEVAL = ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl")

# LiveCodeBench's generic prompt (lcb_runner/prompts/code_generation.py @ 28fef95e): its system message and question
# template, sent as one user turn because Qwen3 is used without a system prompt.
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
    "correct Python program that matches the specification and passes all tests."
)
LCB_WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
LCB_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
    "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)


def expand(benches: str) -> List[str]:
    names: List[str] = []
    for part in benches.split(","):
        for name in BENCH_SETS.get(part, [part]):
            if name not in BENCH_SAMPLES:
                raise SystemExit(f"unknown benchmark {name!r}; known: {sorted(BENCH_SAMPLES)} and sets {sorted(BENCH_SETS)}")
            if name not in names:
                names.append(name)
    return names


def aime(data_dir: str, year: str) -> Iterator[Tuple[Dict, Dict]]:
    import pyarrow.parquet as pq

    for i, r in enumerate(pq.read_table(os.path.join(data_dir, f"aime{year}.parquet")).to_pylist()):
        rid = f"aime{year}-{i}"
        yield {"id": rid, "messages": r["prompt"]}, {"id": rid, "answer": r["reward_model"]["ground_truth"]}


def lcb_prompt(p: Dict) -> str:
    prompt = f"### Question:\n{p['question_content']}\n\n"
    if p["starter_code"]:
        prompt += f"### Format: {LCB_WITH_STARTER}\n```python\n{p['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return f"{LCB_SYSTEM}\n\n{prompt}"


def lcb_tests(p: Dict) -> Dict:
    """LiveCodeBench's evaluation sample (CodeGenerationProblem.get_evaluation_sample): public then private tests."""
    public = json.loads(p["public_test_cases"])
    try:
        private = json.loads(p["private_test_cases"])
    except Exception:
        private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(p["private_test_cases"].encode("utf-8")))))
    tests = public + private
    return {
        "input_output": json.dumps(
            {
                "inputs": [t["input"] for t in tests],
                "outputs": [t["output"] for t in tests],
                "fn_name": json.loads(p["metadata"]).get("func_name"),
            }
        )
    }


def lcb(windows: List[str]) -> Iterator[Tuple[Dict, Dict]]:
    from huggingface_hub import hf_hub_download

    repo, revision, files = LCB
    seen = set()
    counts = {w: 0 for w in windows}
    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
        with open(path) as f:
            for line in f:
                p = json.loads(line)
                day = p["contest_date"][:10]
                for w in windows:
                    lo, hi = LCB_WINDOWS[w]
                    if lo <= day < hi and (w, p["question_id"]) not in seen:
                        seen.add((w, p["question_id"]))
                        counts[w] += 1
                        rid = f"{w}-{p['question_id']}"
                        meta = {"id": rid, "sample": lcb_tests(p), "difficulty": p["difficulty"], "date": day}
                        yield {"id": rid, "messages": [{"role": "user", "content": lcb_prompt(p)}]}, meta
    print("LiveCodeBench problems per window:", counts)


def if_rows(name: str, path: str) -> Iterator[Tuple[Dict, Dict]]:
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = f"{name}-{r['key']}"
            meta = {"id": rid, "key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
            yield {"id": rid, "messages": [{"role": "user", "content": r["prompt"]}]}, meta


def build_bench(args) -> None:
    benches = expand(args.benches)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    rows: List[Tuple[Dict, Dict]] = []
    for name in benches:
        if name in ("aime24", "aime25"):
            items = list(aime(os.path.expanduser(args.data), name[4:]))
        elif name.startswith("lcb_"):
            continue  # all windows in one pass over the files, below
        elif name == "ifeval":
            from huggingface_hub import hf_hub_download

            items = list(if_rows(name, hf_hub_download(IFEVAL[0], IFEVAL[2], repo_type="dataset", revision=IFEVAL[1])))
        else:
            items = list(if_rows(name, os.path.join(os.path.expanduser(args.tools), "IFBench", "data", "IFBench_test.jsonl")))
        rows += [(dict(req, bench=name), dict(meta, bench=name)) for req, meta in items]
    windows = [b for b in benches if b.startswith("lcb_")]
    if windows:
        rows += [(dict(req, bench=meta["id"].split("-")[0]), dict(meta, bench=meta["id"].split("-")[0])) for req, meta in lcb(windows)]
    write(rows, out, lambda req: args.samples or BENCH_SAMPLES[req["bench"]])


def build_prompts(args) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.expanduser(args.parquet)).to_pylist()
    random.Random(args.seed).shuffle(table)
    rows = []
    for i, r in enumerate(table[: args.n]):
        rid = f"{args.bench}-{i}"
        req = {"id": rid, "bench": args.bench, "messages": r["prompt"]}
        rows.append((req, {"id": rid, "bench": args.bench, "answer": r["reward_model"]["ground_truth"]}))
    write(rows, os.path.expanduser(args.out), lambda req: args.samples or 1)


def write(rows: List[Tuple[Dict, Dict]], out: str, samples) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "requests.jsonl"), "w") as req_f, open(os.path.join(out, "grading.jsonl"), "w") as meta_f:
        for req, meta in rows:
            req_f.write(json.dumps(dict(req, n=samples(req))) + "\n")
            meta_f.write(json.dumps(meta) + "\n")
    per_bench: Dict[str, int] = {}
    for req, _ in rows:
        per_bench[req["bench"]] = per_bench.get(req["bench"], 0) + 1
    print(f"wrote {len(rows)} problems to {out}: {per_bench}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--benches", required=True, help="names or sets: math, code, if, all")
    b.add_argument("--data", required=True, help="the kit's data folder (aime24/25.parquet)")
    b.add_argument("--tools", required=True, help="the folder holding the IFBench clone")
    b.add_argument("--samples", type=int, help="override every benchmark's sample count (testing)")
    b.add_argument("--out", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--parquet", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bench", default="math")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    build_bench(args) if args.cmd == "bench" else build_prompts(args)


if __name__ == "__main__":
    main()
