#!/usr/bin/env python3
"""Measure the append-to-cached-prefix turn tax on MLX radix serving.

The agentic tool-result turn is: long cached prefix + short new suffix.
Per depth D and suffix length K: prime the prefix once (cold, not measured),
then append K tokens of NEW text and stream one token, timing TTFT. The
suffix lead word is unique per run so the radix cache can never serve the
append as a full hit (R9700 lesson: uniqueness must survive truncation to
~4 chars, so leads differ in their first letters). Decode comparator: a
64-token streamed decode on the same cached prefix, ITL mean.

Cache verification: ``usage.prompt_tokens_details.cached_tokens`` when the
server runs with --enable-cache-report; a run without a verified hit is
recorded ``cache_hit_verified: false`` and excluded from the verdict.

Usage:
  python scripts/bench/measure_extend_cost.py --port 23334 \
      --depths 8192,32768,65536 --suffix-tokens 1,64 --runs 3 \
      --out benchmarks/session-endurance/extend-cost.json
"""
import argparse
import json
import random
import time
import urllib.request

UNIQUE_LEADS = (
    "alpha", "bravo", "cobalt", "delta", "echo", "fjord", "gamma", "helix",
    "ionic", "jade", "krypton", "lumen", "mesa", "nova", "onyx", "prism",
)
VOCAB = (
    "system record window process value table market region station final "
    "report ancient garden silver monitor velvet crimson harbor lantern "
    "meadow copper thunder whisper granite orchid saffron timber quartz"
).split()


def make_prefix(target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    target_chars = int(target_tokens * 6 * 1.05)
    out, n = [], 0
    while n < target_chars:
        s = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(8, 14)))
        s = s.capitalize() + ". "
        out.append(s)
        n += len(s)
    return "".join(out)[:target_chars]


def stream_request(base_url: str, prompt: str, max_tokens: int, timeout: int = 3600):
    """POST streaming completion; return (ttft, itls, usage)."""
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, times, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            ch = chunk.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                times.append(now)
    itls = [b - a for a, b in zip(times, times[1:])]
    return ttft, itls, usage


def cached_tokens(usage: dict):
    det = usage.get("prompt_tokens_details") or {}
    return det.get("cached_tokens")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--depths", default="8192,32768,65536")
    p.add_argument("--suffix-tokens", default="1,64")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="benchmarks/session-endurance/extend-cost.json")
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base = f"http://localhost:{args.port}"
    results = []
    lead_i = 0

    for depth in [int(x) for x in args.depths.split(",")]:
        prefix = make_prefix(depth, args.seed + depth)
        print(f"== depth {depth}: priming prefix (cold, unmeasured) ==")
        _, _, usage = stream_request(base, prefix, 1)
        prefix_tokens = usage.get("prompt_tokens", 0)
        print(f"   primed: {prefix_tokens} prompt tokens")

        # decode comparator on the cached prefix
        _, itls, usage = stream_request(base, prefix, 64)
        dec_cached = cached_tokens(usage)
        decode_ms = (sum(itls) / len(itls) * 1000) if itls else 0
        print(f"   decode comparator: {decode_ms:.1f} ms/token "
              f"(cached_tokens={dec_cached})")

        for k in [int(x) for x in args.suffix_tokens.split(",")]:
            ttfts, verified = [], []
            for run in range(args.runs):
                lead = UNIQUE_LEADS[lead_i % len(UNIQUE_LEADS)]
                lead_i += 1
                suffix = (" " + lead + " " +
                          " ".join("suffix" for _ in range(max(0, k - 1))))[:k * 7]
                ttft, _, usage = stream_request(base, prefix + suffix, 1)
                ct = cached_tokens(usage)
                ok = ct is not None and ct >= 0.9 * prefix_tokens
                ttfts.append(ttft)
                verified.append(ok)
                print(f"   append k={k:>3} run{run}: ttft={ttft * 1000:8.1f} ms  "
                      f"cached_tokens={ct}  verified={ok}")
            good = [t for t, v in zip(ttfts, verified) if v]
            med = sorted(good)[len(good) // 2] if good else None
            results.append({
                "depth_label": depth, "prefix_tokens": prefix_tokens,
                "suffix_tokens": k,
                "ttft_ms_median": round(med * 1000, 1) if med else None,
                "ttft_ms_all": [round(t * 1000, 1) for t in ttfts],
                "cache_hit_verified": all(verified) and bool(verified),
                "decode_ms_per_token": round(decode_ms, 1),
                "tax_ratio": (round(med * 1000 / decode_ms, 1)
                              if med and decode_ms else None),
            })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'depth':>8} {'k':>4} {'ttft_med':>10} {'decode':>8} {'ratio':>7} verified")
    for r in results:
        print(f"{r['prefix_tokens']:>8} {r['suffix_tokens']:>4} "
              f"{str(r['ttft_ms_median']):>10} {r['decode_ms_per_token']:>8} "
              f"{str(r['tax_ratio']):>7} {r['cache_hit_verified']}")
    print(f"receipts -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
