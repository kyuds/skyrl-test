"""Do identical scoring requests land on numerically different replicas?

Fires K identical echo_last scoring requests concurrently, clusters the returned logprob
vectors (exact match), and prints cluster sizes, max pairwise deviation between clusters,
and any response headers that differ between clusters (to see whether it's a replica/route).

    uv run --with httpx python skyrl-test/fireworks_logprob_replicas.py --model accounts/fireworks/models/gpt-oss-120b --k 24
"""
from __future__ import annotations
import argparse, asyncio, collections, os, sys
BASE = "https://api.fireworks.ai/inference/v1/completions"
TEXT = ("On-policy distillation samples trajectories from the student model and uses a teacher to grade each "
        "token. The per-token reverse KL is the advantage. Dense supervision greatly improves compute efficiency "
        "compared to sparse rewards. The student only optimizes the immediate next token.") * 3

async def main() -> int:
    import httpx
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--k", type=int, default=24)
    a = ap.parse_args(); H = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(BASE, headers=H, json={"model": a.model, "prompt": TEXT, "max_tokens": 1, "return_token_ids": True})
        r.raise_for_status(); ids = r.json().get("prompt_token_ids") or r.json()["choices"][0]["prompt_token_ids"]; R = len(ids) // 2
        body = {"model": a.model, "prompt": ids, "max_tokens": 0, "temperature": 1.0, "logprobs": True, "echo_last": R}
        async def one():
            rr = await c.post(BASE, headers=H, json=body); rr.raise_for_status()
            vec = tuple(e["logprob"] for e in rr.json()["choices"][0]["logprobs"]["content"][:R])
            hdr = {k.lower(): v for k, v in rr.headers.items() if k.lower().startswith(("fireworks", "x-", "server", "via", "cf-"))}
            return vec, hdr
        res = await asyncio.gather(*(one() for _ in range(a.k)))
    clusters = collections.defaultdict(list)
    for i, (vec, hdr) in enumerate(res): clusters[vec].append((i, hdr))
    vecs = list(clusters)
    print(f"model={a.model.split('/')[-1]} K={a.k} scored_positions={R} distinct_logprob_vectors={len(vecs)} sizes={sorted((len(v) for v in clusters.values()), reverse=True)}")
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            d = [abs(x - y) for x, y in zip(vecs[i], vecs[j])]
            print(f"  cluster{i} vs cluster{j}: max|Δ|={max(d):.4f} mean|Δ|={sum(d)/len(d):.4f} frac>0.05={sum(x>0.05 for x in d)/len(d):.2f}")
    # headers that differ across clusters
    keys = set().union(*(h.keys() for _, hs in clusters.items() for _, h in hs))
    for k in sorted(keys):
        vals = {ci: sorted({h.get(k, "") for _, h in hs})[:3] for ci, (_, hs) in enumerate(clusters.items())}
        if len({tuple(v) for v in vals.values()}) > 1: print(f"  header differs across clusters: {k} -> {vals}")
    return 0

if __name__ == "__main__": sys.exit(asyncio.run(main()))
