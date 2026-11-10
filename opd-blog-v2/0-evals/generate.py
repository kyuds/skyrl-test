"""Samples a model's answers to requests.jsonl with vLLM, on every GPU of the node: one process per engine of --tp
GPUs, each with a share of the problems, so an 8-GPU node is 8 independent engines at --tp 1, or 4 at --tp 2 (no Ray;
Qwen3-32B needs --tp 2 on an 80 GB H100, everything else fits one GPU).

    cd ~/SkyRL && uv run --isolated --extra fsdp python skyrl-test/opd-blog-v2/0-evals/generate.py \\
        --model Qwen/Qwen3-8B --requests <dir>/requests.jsonl --out <dir>/responses.jsonl \\
        --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 32768 [--tp 2]

The prompt is the model's own chat template with thinking on, the same render the OPD student and teacher use.
Writes one line per sample: id, bench, sample, response, finish_reason ("length" = cut at --max-tokens), tokens.
A shard that finished is kept, so a rerun after a crash only samples the shards that did not.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def worker(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    with open(args.requests) as f:
        requests = [json.loads(line) for line in f]
    mine = requests[args.worker :: args.num_workers]
    tok = AutoTokenizer.from_pretrained(args.model)
    from transformers import AutoConfig

    ids = []
    for r in mine:
        text = tok.apply_chat_template(r["messages"], add_generation_prompt=True, enable_thinking=True, tokenize=False)
        ids.append(tok(text, add_special_tokens=False)["input_ids"])
    longest = max((len(x) for x in ids), default=0)
    # The context is capped at the model's (40,960 for Qwen3); a prompt too long for its full answer budget gets less.
    limit = getattr(AutoConfig.from_pretrained(args.model), "max_position_embeddings", None) or (longest + args.max_tokens)
    max_len = min(longest + args.max_tokens, limit)
    prompts, params = [], []
    for k, (r, x) in enumerate(zip(mine, ids)):
        prompts.append(TokensPrompt(prompt_token_ids=x))
        params.append(
            SamplingParams(
                n=int(r["n"]),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=max(1, min(args.max_tokens, max_len - len(x))),
                seed=args.seed + 1000 * args.worker + k,
            )
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="mp" if args.tp > 1 else None,
        max_model_len=max_len,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    start = time.time()
    outputs = llm.generate(prompts, params)
    tmp = f"{args.out}.shard{args.worker}.tmp"
    n = 0
    with open(tmp, "w") as f:
        for r, out in zip(mine, outputs):
            for j, c in enumerate(out.outputs):
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "bench": r["bench"],
                            "sample": j,
                            "response": c.text,
                            "finish_reason": c.finish_reason,
                            "tokens": len(c.token_ids),
                        }
                    )
                    + "\n"
                )
                n += 1
    os.replace(tmp, f"{args.out}.shard{args.worker}")
    print(f"worker {args.worker}: {len(mine)} problems, {n} samples in {time.time() - start:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=0, help="GPUs to use; 0 = every GPU on the node")
    parser.add_argument("--tp", type=int, default=1, help="GPUs per engine (tensor parallelism)")
    parser.add_argument("--worker", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker >= 0:
        worker(args)
        return

    gpus = args.gpus or int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
    engines = gpus // args.tp
    if engines < 1:
        raise SystemExit(f"{gpus} GPUs found (nvidia-smi -L), {args.tp} needed per engine")
    procs = {}
    for i in range(engines):
        if os.path.exists(f"{args.out}.shard{i}"):
            print(f"shard {i} already sampled")
            continue
        cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--worker", str(i), "--num-workers", str(engines)]
        devices = ",".join(str(i * args.tp + k) for k in range(args.tp))
        procs[i] = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices))
    failed = [i for i, p in procs.items() if p.wait() != 0]
    if failed:
        raise SystemExit(f"workers {failed} failed; run again to retry only them")
    with open(args.out + ".tmp", "w") as out:
        for i in range(engines):
            with open(f"{args.out}.shard{i}") as f:
                out.write(f.read())
    os.replace(args.out + ".tmp", args.out)
    for i in range(engines):
        os.remove(f"{args.out}.shard{i}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
