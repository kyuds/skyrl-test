"""Are echoed prompt logprobs raw model logprobs, or sampling-adjusted (temperature / top_p)?

For OPD the teacher logprob must be the raw log-softmax of the model. Fireworks documents a
separate `sampling_logprob`, but some stacks omit it and scale `logprob` instead. This sends
the same integer prompt with echo_last + logprobs=true + top_logprobs=5 under several sampling
settings and compares, per echoed position, the chosen-token logprob and the top-5 gaps.
If T=0.5 doubles the gaps (l_a - l_b) relative to T=1.0, `logprob` is temperature-scaled.

    uv run --with httpx python skyrl-test/fireworks_logprob_temperature_check.py --model accounts/fireworks/models/qwen3p8-max
"""
from __future__ import annotations
import argparse, os, statistics, sys
BASE = "https://api.fireworks.ai/inference/v1/completions"

def main() -> int:
    import httpx
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    key = os.environ.get("FIREWORKS_API_KEY"); H = {"Authorization": f"Bearer {key}"}; c = httpx.Client(timeout=120)
    text = "The quick brown fox jumps over the lazy dog because it wanted to reach the river before sunset."
    r = c.post(BASE, headers=H, json={"model": a.model, "prompt": text, "max_tokens": 1, "return_token_ids": True}); r.raise_for_status()
    ids = r.json().get("prompt_token_ids") or r.json()["choices"][0]["prompt_token_ids"]; R = len(ids) // 2
    def score(**sp):
        body = {"model": a.model, "prompt": ids, "max_tokens": 0, "logprobs": True, "top_logprobs": 5, "echo_last": R,
                "return_token_ids": True, "temperature": 1.0, **sp}
        rr = c.post(BASE, headers=H, json=body)
        if rr.status_code != 200: return None, f"HTTP {rr.status_code} {rr.text[:160]}"
        content = rr.json()["choices"][0]["logprobs"]["content"][:R]
        chosen = [e["logprob"] for e in content]
        tops = [{(t.get("token_id"), t.get("token")): t["logprob"] for t in (e.get("top_logprobs") or [])} for e in content]
        return (chosen, tops), None
    settings = {"T=1.0": {}, "T=1.0,top_p=0.5": {"top_p": 0.5}, "T=1.0,top_k=1": {"top_k": 1}, "T=0.5": {"temperature": 0.5}, "T=2.0": {"temperature": 2.0}}
    results = {}
    for name, sp in settings.items():
        res, err = score(**sp); results[name] = res
        print(f"{name:18s}", "ERROR " + err if err else f"chosen[:4]={[round(x,4) for x in res[0][:4]]}")
    base = results["T=1.0"]
    if not base: return 1
    for name, res in results.items():
        if not res or name == "T=1.0": continue
        same = all(abs(x - y) < 1e-4 for x, y in zip(base[0], res[0]))
        ratios = []
        for tb, tr in zip(base[1], res[1]):
            keys = [k for k in tb if k in tr]
            if len(keys) >= 2:
                (k1, k2) = keys[0], keys[1]
                gb, gr = tb[k1] - tb[k2], tr[k1] - tr[k2]
                if abs(gb) > 1e-3: ratios.append(gr / gb)
        print(f"  vs {name:16s} chosen identical={same}; top-5 gap ratio (res/base) median={statistics.median(ratios):.3f} over {len(ratios)} positions"
              if ratios else f"  vs {name:16s} chosen identical={same}; (no comparable top-5 pairs)")
    return 0

if __name__ == "__main__": sys.exit(main())
