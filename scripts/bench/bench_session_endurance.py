#!/usr/bin/env python3
"""Session-endurance driver: multi-turn agentic-shaped load on radix serving.

Simulates the daily workload: a shared system+repo prefix, each turn appends
~1.5K tokens of tool-output-shaped text and decodes a capped reply on the
growing conversation. The prefix grows to --roll-at tokens then rolls back to
the base prefix (fresh conversation), as agent harnesses do. Per-turn JSONL:
TTFT, decode ITL mean, server-verified prompt_tokens, cached_tokens.

Arms: default radix-on; --flush-every N POSTs /flush_cache every N turns
(mitigation arm); run the same driver against a --disable-radix-cache serve
for the attribution control.

Usage:
  python scripts/bench/bench_session_endurance.py --port 23334 --turns 150 \
      --tag radixon --out benchmarks/session-endurance/turns-radixon.jsonl
"""
import argparse
import json
import random
import time
import urllib.request

VOCAB = (
    "def return import class self value index buffer tensor layer cache pool "
    "token batch request schedule commit branch diff test assert error trace "
    "path config launch serve probe gate receipt verdict ladder depth chunk"
).split()


def tool_output(rng: random.Random, approx_tokens: int) -> str:
    out, n = [], 0
    target = approx_tokens * 6
    while n < target:
        line = ("  " + " ".join(rng.choice(VOCAB)
                                for _ in range(rng.randint(6, 12))))
        out.append(line)
        n += len(line)
    return "tool_output[{}]:\n".format(rng.randint(1000, 9999)) + "\n".join(out)


def stream_turn(base_url: str, prompt: str, max_tokens: int):
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, times, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
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
    itl_mean = sum(itls) / len(itls) if itls else 0
    det = usage.get("prompt_tokens_details") or {}
    return {
        "ttft_s": round(ttft, 3) if ttft else None,
        "itl_mean_ms": round(itl_mean * 1000, 1),
        "decode_tok_per_s": round(1 / itl_mean, 1) if itl_mean else 0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": det.get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--turns", type=int, default=150)
    p.add_argument("--turn-tokens", type=int, default=1500)
    p.add_argument("--reply-tokens", type=int, default=256)
    p.add_argument("--roll-at", type=int, default=32000)
    p.add_argument("--base-tokens", type=int, default=4000)
    p.add_argument("--flush-every", type=int, default=0,
                   help="POST /flush_cache every N turns (0 = never)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--tag", default="radixon")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base_url = f"http://localhost:{args.port}"
    rng = random.Random(args.seed)
    base_prefix = ("SYSTEM: agentic session driver.\n" +
                   tool_output(rng, args.base_tokens))
    convo = base_prefix
    rolls = 0

    with open(args.out, "w") as f:
        for turn in range(1, args.turns + 1):
            convo += "\n\n" + tool_output(rng, args.turn_tokens) + "\nASSISTANT:"
            try:
                rec = stream_turn(base_url, convo, args.reply_tokens)
            except Exception as e:
                rec = {"error": repr(e)}
                f.write(json.dumps({"turn": turn, "tag": args.tag, **rec}) + "\n")
                f.flush()
                print(f"turn {turn}: ERROR {e!r} — stopping")
                return 1
            convo += " reply." * 20  # small stand-in for the reply text
            rec.update({"turn": turn, "tag": args.tag, "rolls": rolls,
                        "ts": time.strftime("%H:%M:%S")})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if turn % 10 == 0 or turn == 1:
                print(f"turn {turn:>3}: in={rec['prompt_tokens']} "
                      f"cached={rec['cached_tokens']} ttft={rec['ttft_s']}s "
                      f"decode={rec['decode_tok_per_s']} tok/s")
            if rec.get("prompt_tokens") and rec["prompt_tokens"] > args.roll_at:
                convo = base_prefix
                rolls += 1
                print(f"turn {turn}: rolled conversation (roll #{rolls})")
            if args.flush_every and turn % args.flush_every == 0:
                urllib.request.urlopen(
                    urllib.request.Request(f"{base_url}/flush_cache",
                                           method="POST"), timeout=60).read()
                print(f"turn {turn}: /flush_cache")
    print(f"done: {args.turns} turns, {rolls} rolls -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
