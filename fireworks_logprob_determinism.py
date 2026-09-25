"""How stable are Fireworks' echoed prompt logprobs across identical requests?

A reverse-KL advantage is (log p_student - log p_teacher) per token; if the served teacher's
logprobs wobble by ~0.1 nats between calls, that noise lands directly in the advantage.
Sends the SAME integer prompt N times (sequentially and concurrently) and once inside an
integer[][] batch, and reports per-position max |delta| against the first response.

    uv run --with httpx python skyrl-test/fireworks_logprob_determinism.py --model accounts/fireworks/models/qwen3p8-max
"""
from __future__ import annotations
import argparse, asyncio, os, statistics, sys
BASE = "https://api.fireworks.ai/inference/v1/completions"
TEXT = ("On-policy distillation samples trajectories from the student model and uses a teacher to grade each "
        "token. The per-token reverse KL is the advantage. Dense supervision greatly improves compute efficiency "
        "compared to sparse rewards. The student only optimizes the immediate next token.") * 3

async def main() -> int:
    import httpx
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args(); key = os.environ["FIREWORKS_API_KEY"]; H = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(BASE, headers=H, json={"model": a.model, "prompt": TEXT, "max_tokens": 1, "return_token_ids": True})
        r.raise_for_status(); ids = r.json().get("prompt_token_ids") or r.json()["choices"][0]["prompt_token_ids"]
        R = len(ids) // 2
        body = lambda p: {"model": a.model, "prompt": p, "max_tokens": 0, "temperature": 1.0, "logprobs": True, "echo_last": R}
        async def lps(p):
            rr = await c.post(BASE, headers=H, json=body(p)); rr.raise_for_status()
            return [[e["logprob"] for e in ch["logprobs"]["content"][:R]] for ch in rr.json()["choices"]]
        seq = [ (await lps(ids))[0] for _ in range(a.n) ]                      # sequential repeats
        conc = [x[0] for x in await asyncio.gather(*(lps(ids) for _ in range(a.n)))]  # concurrent repeats
        batched = await lps([ids] * 4)                                          # same prompt, 4x in one request
        ref = seq[0]
        def dev(rows): return [max(abs(x - y) for x, y in zip(ref, row)) for row in rows]
        d_seq, d_conc, d_bat = dev(seq[1:]), dev(conc), dev(batched)
        print(f"model={a.model.split('/')[-1]} tokens={len(ids)} scored={R}")
        print(f"  sequential repeats x{a.n-1}: max|Δ| per call = {[round(x,4) for x in d_seq]}")
        print(f"  concurrent repeats x{a.n}:  max|Δ| per call = {[round(x,4) for x in d_conc]}")
        print(f"  inside integer[][] x4:     max|Δ| per choice = {[round(x,4) for x in d_bat]}")
        allrows = seq[1:] + conc + batched
        per_pos = [max(abs(ref[i] - row[i]) for row in allrows) for i in range(R)]
        print(f"  overall: max|Δ|={max(per_pos):.4f} median-over-positions of max|Δ|={statistics.median(per_pos):.4f} "
              f"fraction of positions with |Δ|>0.05: {sum(p > 0.05 for p in per_pos)/R:.2f}")
    return 0

if __name__ == "__main__": sys.exit(asyncio.run(main()))
