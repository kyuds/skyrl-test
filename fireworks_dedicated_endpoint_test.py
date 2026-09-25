"""Dedicated (on-demand) Fireworks endpoint as an OPD teacher: create, test, DELETE.

Creates a `qwen3-8b` on-demand deployment on the cheapest validated shape
(`qwen3-8b-minimal` = 1x H200, FP8, $8/GPU-hour => $0.133/min), waits for READY, triggers the
scale-up with a warm-up request, runs the same scoring checks as the serverless probes
(echo_last alignment, max_tokens=0, temperature invariance, top-5 cap, 24-way determinism,
16-way latency), and then DELETES the deployment in a `finally`. Safety rails:
  * minReplicaCount=0, maxReplicaCount=1, scaleToZeroWindow=300s (self-limits if teardown fails)
  * hard timeout on readiness (default 15 min) -> delete and exit
  * nothing is created unless --yes is passed; without it the plan is printed
Measured 2026-09-17: 12.6 min to READY, ~17 min end to end, upper-bound cost ~$2.5.
Teardown uses DELETE ?ignoreChecks=true (a plain DELETE is refused within an hour of traffic).

    uv run --with httpx python skyrl-test/fireworks_dedicated_endpoint_test.py --yes
"""
from __future__ import annotations
import argparse, asyncio, collections, json, os, statistics, sys, time

ACCOUNT = "kyuds"
CP = "https://api.fireworks.ai/v1"
INF = "https://api.fireworks.ai/inference/v1/completions"
TEXT = ("On-policy distillation samples trajectories from the student model and uses a teacher to grade each "
        "token. The per-token reverse KL is the advantage. Dense supervision greatly improves compute efficiency "
        "compared to sparse rewards. The student only optimizes the immediate next token.") * 3


async def main() -> int:
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually create the deployment (bills GPU-seconds)")
    ap.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    ap.add_argument("--shape", default="accounts/fireworks/deploymentShapes/qwen3-8b-minimal")
    ap.add_argument("--deployment-id", default="opd-teacher-probe")
    ap.add_argument("--ready-timeout-min", type=float, default=15)
    a = ap.parse_args()
    key = os.environ["FIREWORKS_API_KEY"]; H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    name = f"accounts/{ACCOUNT}/deployments/{a.deployment_id}"
    body = {"baseModel": a.base_model, "deploymentShape": a.shape, "minReplicaCount": 0, "maxReplicaCount": 1,
            "autoscalingPolicy": {"scaleToZeroWindow": "300s"}, "displayName": a.deployment_id}
    print("plan:", json.dumps({"POST": f"{CP}/accounts/{ACCOUNT}/deployments?deploymentId={a.deployment_id}&disableSpeculativeDecoding=true", "body": body}))
    if not a.yes:
        print("dry run only; pass --yes to create (bills GPU-seconds while a replica is up)"); return 0

    t_start = time.time()
    async with httpx.AsyncClient(timeout=600) as c:
        async def delete():
            # A plain DELETE is refused (code 9) if the deployment served requests in the last hour.
            r = await c.delete(f"{CP}/{name}?ignoreChecks=true", headers=H)
            print(f"[teardown] DELETE {name} -> HTTP {r.status_code} {r.text[:120]}")
            for _ in range(40):
                g = await c.get(f"{CP}/{name}", headers=H)
                if g.status_code == 404 or (g.status_code == 200 and g.json().get("state") in ("DELETED", "DELETING")):
                    print(f"[teardown] state={'404' if g.status_code == 404 else g.json().get('state')}"); break
                await asyncio.sleep(5)
            l = await c.get(f"{CP}/accounts/{ACCOUNT}/deployments", headers=H)
            print(f"[teardown] deployments remaining in account: {l.json().get('totalSize')}")

        try:
            r = await c.post(f"{CP}/accounts/{ACCOUNT}/deployments?deploymentId={a.deployment_id}&disableSpeculativeDecoding=true", headers=H, json=body)
            print(f"[create] HTTP {r.status_code} {r.text[:300]}")
            if r.status_code != 200: return 1
            # poll READY
            while True:
                g = (await c.get(f"{CP}/{name}", headers=H)).json()
                st = g.get("state"); rs = g.get("replicaStats", {})
                print(f"[poll] t={time.time()-t_start:6.0f}s state={st} replicas={json.dumps(rs)} status={g.get('status',{}).get('message','')[:80]}")
                if st == "READY": break
                if st in ("FAILED", "DELETED", "DELETING"): print("[poll] terminal state, aborting"); return 1
                if time.time() - t_start > a.ready_timeout_min * 60: print("[poll] timeout"); return 1
                await asyncio.sleep(15)
            print(f"[deploy] precision={g.get('precision')} accel={g.get('acceleratorType')}x{g.get('acceleratorCount')} shape={g.get('deploymentShape')}")

            # warm-up (triggers scale-up from 0); tolerate errors while the replica boots
            model = name
            t_w = time.time()
            for i in range(60):
                rr = await c.post(INF, headers=H, json={"model": model, "prompt": "hello", "max_tokens": 1})
                if rr.status_code == 200: break
                if i % 4 == 0: print(f"[warmup] t={time.time()-t_start:6.0f}s HTTP {rr.status_code} {rr.text[:100]}")
                await asyncio.sleep(10)
            else:
                print("[warmup] never came up"); return 1
            print(f"[warmup] first successful request after {time.time()-t_w:.0f}s; served model field: {rr.json().get('model')}")

            # alt model string form
            alt = f"{a.base_model}#{name}"
            rr2 = await c.post(INF, headers=H, json={"model": alt, "prompt": "hello", "max_tokens": 1})
            print(f"[model-string] '{alt}' -> HTTP {rr2.status_code}")

            # ---- scoring checks ----
            r0 = await c.post(INF, headers=H, json={"model": model, "prompt": TEXT, "max_tokens": 1, "return_token_ids": True}); r0.raise_for_status()
            ids = r0.json().get("prompt_token_ids") or r0.json()["choices"][0]["prompt_token_ids"]; R = len(ids) // 2
            sc = lambda **kw: {"model": model, "prompt": ids, "max_tokens": 0, "temperature": 1.0, "logprobs": True, "echo_last": R, "return_token_ids": True, **kw}
            r1 = await c.post(INF, headers=H, json=sc()); print(f"[echo_last max_tokens=0] HTTP {r1.status_code} {r1.text[:160] if r1.status_code != 200 else ''}")
            if r1.status_code != 200: return 1
            ch = r1.json()["choices"][0]; content = ch["logprobs"]["content"]
            ok = len(content) == R and [e["token_id"] for e in content] == ids[-R:] and (ch.get("token_ids") or [])[:R] == ids[-R:]
            print(f"[alignment] entries={len(content)} R={R} token_ids_match={ok} usage={json.dumps(r1.json().get('usage'))}")
            base = [e["logprob"] for e in content]
            r2 = await c.post(INF, headers=H, json=sc(temperature=0.5)); l2 = [e["logprob"] for e in r2.json()["choices"][0]["logprobs"]["content"]]
            print(f"[temperature] T=0.5 identical to T=1.0: {all(abs(x-y) < 1e-6 for x, y in zip(base, l2))}")
            r3 = await c.post(INF, headers=H, json=sc(top_logprobs=5)); k = max(len(e.get("top_logprobs") or []) for e in r3.json()["choices"][0]["logprobs"]["content"])
            print(f"[top-k] top_logprobs=5 -> max alternatives {k}")
            r4 = await c.post(INF, headers=H, json=sc(top_logprobs=20)); print(f"[top-k] top_logprobs=20 -> HTTP {r4.status_code} {r4.text[:120] if r4.status_code != 200 else 'accepted, max=' + str(max(len(e.get('top_logprobs') or []) for e in r4.json()['choices'][0]['logprobs']['content']))}")

            # determinism: 24 identical concurrent
            async def vec():
                rr = await c.post(INF, headers=H, json=sc()); rr.raise_for_status()
                return tuple(e["logprob"] for e in rr.json()["choices"][0]["logprobs"]["content"]), rr.headers.get("fireworks-server-processing-time")
            t_d = time.perf_counter(); res = await asyncio.gather(*(vec() for _ in range(24))); d_wall = time.perf_counter() - t_d
            print(f"[determinism] 24 x {R}-token scoring requests took {d_wall:.2f}s wall")
            clusters = collections.defaultdict(list)
            for v, pt in res: clusters[v].append(pt)
            vecs = list(clusters)
            print(f"[determinism] 24 identical concurrent requests -> distinct vectors={len(vecs)} sizes={sorted((len(x) for x in clusters.values()), reverse=True)}")
            for i in range(len(vecs)):
                for j in range(i + 1, min(len(vecs), i + 4)):
                    d = [abs(x - y) for x, y in zip(vecs[i], vecs[j])]
                    print(f"   c{i} vs c{j}: max|Δ|={max(d):.4f} mean|Δ|={sum(d)/len(d):.4f} frac>0.05={sum(x>0.05 for x in d)/len(d):.2f}")

            # latency: 16 concurrent x ~3000 tokens (distinct rotations)
            long = (await c.post(INF, headers=H, json={"model": model, "prompt": TEXT * 40, "max_tokens": 1, "return_token_ids": True})).json()
            lids = long.get("prompt_token_ids") or long["choices"][0]["prompt_token_ids"]; L = min(3000, len(lids) - 16 * 97 - 1)
            assert L >= 2000, f"not enough token ids for the latency test ({len(lids)}); lengthen TEXT"
            seqs = [lids[(i * 97) % (len(lids) - L):][:L] for i in range(16)]
            async def one(p):
                t = time.perf_counter(); rr = await c.post(INF, headers=H, json={"model": model, "prompt": p, "max_tokens": 0, "logprobs": True, "echo_last": L // 2}); return time.perf_counter() - t, rr.status_code
            t0 = time.perf_counter(); lat = await asyncio.gather(*(one(p) for p in seqs)); wall = time.perf_counter() - t0
            good = [d for d, s in lat if s == 200]
            print(f"[latency] 16 x {L} tokens concurrent: ok={len(good)} wall={wall:.2f}s mean={statistics.mean(good):.2f}s agg={len(good)*L/wall:,.0f} tok/s")
        finally:
            await delete()
    el = (time.time() - t_start) / 60
    print(f"[done] elapsed {el:.1f} min; upper-bound cost at $8/GPU-h = ${el * 8 / 60:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
