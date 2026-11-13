#!/usr/bin/env python3
"""Does an agentic follow-up turn served from the restored hybrid-cache state
reproduce the cold turn?

Per depth: prime a user turn (reply kept as history), send a new user message
twice with the history cached (hit vs hit), flush the cache, send it cold, and
compare the greedy token ids of all three. Byte-identical hit-vs-cold means
the restored DeltaNet/Mamba state plus the recomputed tail is numerically the
cold prefill.

Two hits can start from different checkpoints (the second is served from the
state the first one wrote), and a hybrid's hit differs from its cold run the
way two cold runs on different chunk sizes do, so a divergence alone is not a
failure. Each pair is graded on the log-probabilities the two runs give the
tokens they share: a pair passes when the texts are identical, or when the
shared tokens sit within --max-mean-gap / --max-gap of each other and the
step where they part is a near-tie (the other run's token within
--max-flip-margin of the chosen one, where the server returns it). --strict
restores the bit-identity criterion. The teacher-forced grade of the same
question is probe_followup_fidelity.py.

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
                                               "temperature": 0, "logprobs": True, "top_logprobs": 5})
    ch = out["choices"][0]
    rows = (ch.get("logprobs") or {}).get("content") or []
    run = {"ids": [t["token"] for t in rows], "lp": [t["logprob"] for t in rows],
           "top": [{x["token"]: x["logprob"] for x in t.get("top_logprobs") or []} for t in rows]}
    usage = out.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return run, ch["message"].get("content") or "", cached, usage.get("prompt_tokens")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def grade(a, b, lim):
    """Compare two greedy runs: where they part, how far apart the shared tokens' log-probs sit,
    and how close the parting step was to a tie in each run."""
    d = first_diff(a["ids"], b["ids"])
    n = len(a["ids"]) if d is None else d
    gaps = [abs(x - y) for x, y in zip(a["lp"][:n], b["lp"][:n])]
    g = {"first_diff": d, "shared": n, "gap_mean": sum(gaps) / len(gaps) if gaps else 0.0,
         "gap_max": max(gaps) if gaps else 0.0, "flip_margin": None}
    if d is not None and d < len(a["ids"]) and d < len(b["ids"]):
        margins = [r["lp"][d] - r["top"][d][o["ids"][d]] for r, o in ((a, b), (b, a))
                   if d < len(r["top"]) and o["ids"][d] in r["top"][d]]
        g["flip_margin"] = max(margins) if margins else None
    if d is None:
        g["verdict"] = "identical"
    elif lim["strict"]:
        g["verdict"] = "FAIL"
    else:
        ok = g["gap_mean"] <= lim["mean"] and g["gap_max"] <= lim["max"]
        if g["flip_margin"] is not None:
            ok = ok and g["flip_margin"] <= lim["flip"]
        g["verdict"] = "noise" if ok else "FAIL"
    return g


def show(g):
    if g["first_diff"] is None:
        return f"identical (gap mean {g['gap_mean']:.4f} max {g['gap_max']:.4f})"
    m = "n/a" if g["flip_margin"] is None else f"{g['flip_margin']:.3f}"
    return (f"diverge@{g['first_diff']} {g['verdict']} (gap mean {g['gap_mean']:.4f} max {g['gap_max']:.4f} "
            f"over {g['shared']} shared, flip margin {m})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--depths", default="8192,24576")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--out", default="")
    p.add_argument("--max-mean-gap", type=float, default=0.05, help="mean |dlogp| over the shared tokens, nats")
    p.add_argument("--max-gap", type=float, default=1.0, help="largest |dlogp| on a shared token, nats")
    p.add_argument("--max-flip-margin", type=float, default=0.7,
                   help="at the parting step, chosen minus the other run's token, nats")
    p.add_argument("--strict", action="store_true", help="any divergence fails")
    a = p.parse_args()
    lim = {"mean": a.max_mean_gap, "max": a.max_gap, "flip": a.max_flip_margin, "strict": a.strict}
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
        g_hh, g_hc, g_h2c = grade(hit1, hit2, lim), grade(hit1, cold, lim), grade(hit2, cold, lim)
        row = {"depth": depth, "primed_prompt_tokens": pt, "followup_prompt_tokens": pt1,
               "cached_hit1": c1, "cached_hit2": c2, "cached_cold": c3, "tokens": len(hit1["ids"]),
               "hit_vs_hit_first_diff": g_hh["first_diff"], "hit_vs_cold_first_diff": g_hc["first_diff"],
               "hit_vs_hit": g_hh, "hit_vs_cold": g_hc, "hit2_vs_cold": g_h2c}
        rows.append(row)
        print(f"depth {depth}: follow-up {pt1} tok, cached hit1={c1} hit2={c2} cold={c3}, {len(hit1['ids'])} tokens")
        print(f"  hit-vs-hit   {show(g_hh)}")
        print(f"  hit-vs-cold  {show(g_hc)}")
        print(f"  hit2-vs-cold {show(g_h2c)}")
    if a.out:
        json.dump({"limits": lim, "rows": rows}, open(a.out, "w"), indent=2)
    bad = [(r["depth"], k) for r in rows for k in ("hit_vs_hit", "hit_vs_cold", "hit2_vs_cold")
           if r[k]["verdict"] == "FAIL"]
    print("RESULT: " + ("PASS" if not bad else "FAIL " + ", ".join(f"{d}:{k}" for d, k in bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
