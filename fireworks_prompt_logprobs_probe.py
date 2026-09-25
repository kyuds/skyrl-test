"""Probe: does Fireworks' /inference/v1/completions return logprobs for prefilled (prompt) tokens?

This is the one fact the OPD-teacher evaluation could not verify without an API key
(see research/readings/fireworks-opd-teacher.md). Run:

    export FIREWORKS_API_KEY=...      # or FIREWORKS_AI_API_KEY
    uv run --with httpx python skyrl-test/fireworks_prompt_logprobs_probe.py \
        --model accounts/fireworks/models/gpt-oss-120b

    # print the request bodies without sending anything:
    uv run --with httpx python skyrl-test/fireworks_prompt_logprobs_probe.py --dry-run

What it checks, in order (each prints PASS/FAIL and the raw evidence):
  1. string prompt + return_token_ids -> server's prompt_token_ids (so no local tokenizer is needed)
  2. integer-array prompt (prompt + "response" ids) + echo_last=R + logprobs=1 + max_tokens=1
     -> exactly one raw logprob per echoed token, aligned to the last R prompt ids
  3. same request with max_tokens=0 -> accepted or rejected
  4. same request at temperature 0.5 -> echoed `logprob` values unchanged (raw model logprobs,
     not sampling-adjusted); `sampling_logprob`, if present, may differ
  5. logprobs=5 -> top-5 alternatives per position (the cap OPD top-k would live under)
Total cost: a few thousand tokens on a serverless model, i.e. well under a cent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

BASE = "https://api.fireworks.ai/inference/v1/completions"


def _post(client, key, body):
    r = client.post(BASE, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def _logprob_rows(choice):
    """Normalise both response shapes to a list of (token_id|None, logprob, sampling_logprob|None)."""
    lp = choice.get("logprobs") or {}
    if "content" in lp and lp["content"] is not None:  # OpenAI-style (logprobs=true)
        return [(c.get("token_id"), c.get("logprob"), c.get("sampling_logprob")) for c in lp["content"]]
    if "token_logprobs" in lp:  # legacy (logprobs=<int>)
        ids = lp.get("token_ids") or [None] * len(lp["token_logprobs"])
        return [(i, l, None) for i, l in zip(ids, lp["token_logprobs"])]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/gpt-oss-120b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--response", default=" Paris. It is known for the Eiffel Tower.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_AI_API_KEY")
    if not key and not args.dry_run:
        print("no FIREWORKS_API_KEY / FIREWORKS_AI_API_KEY in env; use --dry-run to print requests", file=sys.stderr)
        return 2

    # --- 1. get token ids from the server itself -------------------------------------------
    tok_req = {"model": args.model, "prompt": args.prompt + args.response, "max_tokens": 1,
               "temperature": 1.0, "return_token_ids": True}
    if args.dry_run:
        print("[1] request:", json.dumps(tok_req))
    else:
        import httpx
        client = httpx.Client()
        status, out = _post(client, key, tok_req)
        if status != 200:
            print("[1] FAIL", status, out); return 1
        full_ids = out.get("prompt_token_ids") or out["choices"][0].get("prompt_token_ids")
        print(f"[1] PASS prompt_token_ids: {len(full_ids)} ids")
        # split: score the last R ids as the "response" (R = 40% of the sequence, >= 1)
        R = max(1, int(0.4 * len(full_ids)))
        prompt_ids, resp_ids = full_ids[:-R], full_ids[-R:]

    # --- 2. echo_last scoring on an integer prompt -----------------------------------------
    def score_req(max_tokens=1, temperature=1.0, logprobs=1):
        return {"model": args.model, "prompt": (prompt_ids + resp_ids) if not args.dry_run else "<prompt_ids + resp_ids>",
                "max_tokens": max_tokens, "temperature": temperature, "logprobs": logprobs,
                "echo_last": R if not args.dry_run else "<R>", "return_token_ids": True}

    if args.dry_run:
        print("[2] request:", json.dumps(score_req()))
        print("[3] request:", json.dumps(score_req(max_tokens=0)))
        print("[4] request:", json.dumps(score_req(temperature=0.5)))
        print("[5] request:", json.dumps(score_req(logprobs=5)))
        return 0

    status, out = _post(client, key, score_req())
    if status != 200:
        print("[2] FAIL", status, out); return 1
    ch = out["choices"][0]
    rows = _logprob_rows(ch)
    tok_ids = ch.get("token_ids") or []
    shape = "content" if (ch.get("logprobs") or {}).get("content") is not None else "legacy"
    # Live semantics (2026-09-17): with echo_last=R and max_tokens=C the response carries the R
    # echoed prompt tokens FIRST, then the C generated ones, in BOTH logprobs.content and token_ids.
    echoed, gen = rows[:R], rows[R:]
    ids_ok = all(r[0] == t for r, t in zip(echoed, resp_ids)) and tok_ids[:R] == resp_ids
    print(f"[2] shape={shape} rows={len(rows)} token_ids={len(tok_ids)} R={R} echoed={len(echoed)} "
          f"generated={len(gen)} echoed_token_ids_match_sent={ids_ok} finish={ch.get('finish_reason')} "
          f"-> {'PASS' if len(echoed) == R and ids_ok else 'FAIL'}")
    print("    first echoed rows (token_id, logprob, sampling_logprob):", echoed[:3])
    base_lps = [r[1] for r in echoed]

    status, out3 = _post(client, key, score_req(max_tokens=0))
    if status == 200:
        c3 = out3["choices"][0]; r3 = _logprob_rows(c3)
        print(f"[3] max_tokens=0 -> HTTP 200 rows={len(r3)} token_ids={len(c3.get('token_ids') or [])} "
              f"finish={c3.get('finish_reason')} usage={json.dumps(out3.get('usage'))}")
    else:
        print(f"[3] max_tokens=0 -> HTTP {status} {str(out3)[:200]}")

    status, out4 = _post(client, key, score_req(temperature=0.5))
    if status == 200:
        rows4 = _logprob_rows(out4["choices"][0])
        ech4 = rows4[:R]
        lps4 = [r[1] for r in ech4]
        same = len(lps4) == len(base_lps) and all(abs(a - b) < 1e-6 for a, b in zip(lps4, base_lps))
        print(f"[4] temperature=0.5: raw logprobs identical to T=1.0 -> {'PASS' if same else 'FAIL'}; "
              f"sampling_logprob present={any(r[2] is not None for r in ech4)}")
    else:
        print("[4] FAIL", status, str(out4)[:200])

    status, out5 = _post(client, key, score_req(logprobs=5))
    if status == 200:
        lp = out5["choices"][0].get("logprobs") or {}
        top = lp.get("top_logprobs") or [c.get("top_logprobs") for c in (lp.get("content") or [])]
        k = max((len(t or {}) for t in top), default=0)
        print(f"[5] logprobs=5 -> max alternatives per position = {k}")
    else:
        print("[5] FAIL", status, str(out5)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
