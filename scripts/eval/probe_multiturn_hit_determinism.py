#!/usr/bin/env python3
"""Does an agentic follow-up turn served from the restored hybrid-cache state
reproduce the cold turn?

Per depth: prime a user turn (reply kept as history), send a new user message
twice with the history cached (hit vs hit), flush the cache, send it cold, and
compare the greedy token ids of all three. Byte-identical hit-vs-cold means
the restored DeltaNet/Mamba state plus the recomputed tail is numerically the
cold prefill; hit-vs-hit must always agree.

    python scripts/eval/probe_multiturn_hit_determinism.py --port 23334 --depths 8192,24576 --max-tokens 96
"""
import argparse
import json
import random
import sys
import urllib.request

VOCAB = ("system record window process value table market region station final report ancient garden "
         "silver monitor velvet crimson harbor lantern meadow copper thunder whisper granite orchid saffron "
         "timber quartz").split()


def make_prefix(target_tokens, seed):
    rng = random.Random(seed)
    out, n = [], 0
    while n < int(target_tokens * 6):
        s = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(8, 14))).capitalize() + ". "
        out.append(s); n += len(s)
    return "".join(out)


def post(base, path, body):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:  # /flush_cache answers with plain text
        return {"raw": raw.decode("utf-8", errors="replace")}


def chat(base, messages, max_tokens):
    out = post(base, "/v1/chat/completions", {"model": "default", "messages": messages, "max_tokens": max_tokens,
                                               "temperature": 0, "logprobs": True, "top_logprobs": 1})
    ch = out["choices"][0]
    ids = [t["token"] for t in (ch.get("logprobs") or {}).get("content") or []]
    usage = out.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return ids, ch["message"].get("content") or "", cached, usage.get("prompt_tokens")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--depths", default="8192,24576")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--out", default="")
    a = p.parse_args()
    base = f"http://localhost:{a.port}"
    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        prefix = make_prefix(depth, 7 + depth)
        post(base, "/flush_cache", {})
        _, reply, _, pt = chat(base, [{"role": "user", "content": prefix}], 64)
        hist = [{"role": "user", "content": prefix}, {"role": "assistant", "content": reply or "Noted."},
                {"role": "user", "content": "Summarize the record above in one sentence, then name the first word."}]
        hit1, _, c1, pt1 = chat(base, hist, a.max_tokens)
        hit2, _, c2, _ = chat(base, hist, a.max_tokens)
        post(base, "/flush_cache", {})
        cold, _, c3, _ = chat(base, hist, a.max_tokens)
        d_hh, d_hc = first_diff(hit1, hit2), first_diff(hit1, cold)
        row = {"depth": depth, "primed_prompt_tokens": pt, "followup_prompt_tokens": pt1,
               "cached_hit1": c1, "cached_hit2": c2, "cached_cold": c3, "tokens": len(hit1),
               "hit_vs_hit_first_diff": d_hh, "hit_vs_cold_first_diff": d_hc}
        rows.append(row)
        print(f"depth {depth}: follow-up {pt1} tok, cached hit1={c1} hit2={c2} cold={c3}; "
              f"hit-vs-hit {'identical' if d_hh is None else f'diverge@{d_hh}'}; "
              f"hit-vs-cold {'identical' if d_hc is None else f'diverge@{d_hc}'} over {len(hit1)} tokens")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
    return 0 if all(r["hit_vs_hit_first_diff"] is None for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
