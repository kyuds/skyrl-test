"""Latency / throughput of teacher-style prefill scoring on Fireworks.

    export FIREWORKS_API_KEY=...
    uv run --with httpx python skyrl-test/fireworks_score_latency.py \
        --model accounts/fireworks/models/gpt-oss-120b --seq-len 3000 --n 8

Sends `n` distinct ~seq_len-token sequences concurrently, each scored with
echo_last = seq_len // 2, logprobs=true, max_tokens=1, and then the same `n` sequences as ONE
batched integer[][] request. Reports per-request latency, aggregate tokens/s, the `usage`
block (to see cached vs uncached billing), and the logprob entry counts per choice.
Spend: ~2 * n * seq_len tokens (e.g. 48k tokens ≈ $0.007 on gpt-oss-120b).
"""
from __future__ import annotations

import argparse, asyncio, json, os, statistics, sys, time

BASE = "https://api.fireworks.ai/inference/v1/completions"
PARA = ("On-policy distillation samples trajectories from the student model and uses a teacher to grade "
        "each token. The per-token reverse KL is the advantage. Dense supervision greatly improves "
        "compute efficiency compared to sparse rewards from reinforcement learning. ")


async def main() -> int:
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/gpt-oss-120b")
    ap.add_argument("--seq-len", type=int, default=3000)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_AI_API_KEY")
    if not key:
        print("no key", file=sys.stderr); return 2
    H = {"Authorization": f"Bearer {key}"}

    async with httpx.AsyncClient(timeout=300) as c:
        # 1. get enough token ids from the server (no local tokenizer)
        text = PARA * (a.seq_len // 40 + 4)
        r = await c.post(BASE, headers=H, json={"model": a.model, "prompt": text, "max_tokens": 1,
                                                "temperature": 1.0, "return_token_ids": True})
        r.raise_for_status(); out = r.json()
        ids = out.get("prompt_token_ids") or out["choices"][0].get("prompt_token_ids")
        if len(ids) < a.seq_len:
            print(f"only {len(ids)} ids available; lower --seq-len"); return 1
        L = a.seq_len; R = L // 2
        # n distinct sequences: rotate so no two share a prefix (defeats prefix caching)
        seqs = [ids[(i * 97) % (len(ids) - L):][:L] for i in range(a.n)]
        body = lambda p: {"model": a.model, "prompt": p, "max_tokens": 1, "temperature": 1.0,
                          "logprobs": True, "echo_last": R, "return_token_ids": True}

        # 2. n concurrent single-sequence requests
        async def one(p):
            t = time.perf_counter(); rr = await c.post(BASE, headers=H, json=body(p)); dt = time.perf_counter() - t
            return dt, rr.status_code, (rr.json() if rr.status_code == 200 else rr.text[:200])
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(p) for p in seqs))
        wall = time.perf_counter() - t0
        lat = [d for d, s, _ in res if s == 200]
        bad = [(s, o) for _, s, o in res if s != 200]
        counts = [len((o["choices"][0].get("logprobs") or {}).get("content") or []) for _, s, o in res if s == 200]
        usage = [o.get("usage") for _, s, o in res if s == 200][:1]
        print(f"[concurrent x{a.n}, L={L}, echo_last={R}] ok={len(lat)} bad={len(bad)} wall={wall:.2f}s "
              f"lat min/mean/max={min(lat):.2f}/{statistics.mean(lat):.2f}/{max(lat):.2f}s "
              f"agg={len(lat)*L/wall:,.0f} tok/s  logprob entries/choice={sorted(set(counts))}")
        print("   usage (first):", json.dumps(usage[0]) if usage else None)
        if bad: print("   errors:", bad[:2])

        # 3. one batched request with integer[][] prompt
        t = time.perf_counter()
        rr = await c.post(BASE, headers=H, json=body(seqs)); dt = time.perf_counter() - t
        if rr.status_code != 200:
            print(f"[batched integer[][]] HTTP {rr.status_code}: {rr.text[:300]}"); return 0
        o = rr.json(); ch = o.get("choices", [])
        counts = [len((x.get("logprobs") or {}).get("content") or []) for x in ch]
        print(f"[batched integer[][] x{a.n}] lat={dt:.2f}s choices={len(ch)} logprob entries/choice={sorted(set(counts))} "
              f"agg={a.n*L/dt:,.0f} tok/s usage={json.dumps(o.get('usage'))}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
