#!/usr/bin/env python3
"""Does a multi-token forward that continues a restored cache match the
one-token decode chain? Raw MLX on an mlx-vlm checkpoint, no server.

Teacher-forced on a `decode_fidelity.py` dump. The decode chain feeds one
token per forward. At checkpoints the probe restores the cache as it stood K
tokens earlier, feeds those K tokens in one forward (what a radix prefix hit
does with the tokens after the cached prefix), compares the last position's
distribution with the chain's, and steps three more tokens to see whether the
difference decays. K = 1 repeats the chain's own operation on a restored
cache, so it must give exactly zero: snapshot and restore are exact.

`--reset` clears mlx-vlm's saved position state before the K-token forward.
The language model then numbers those tokens from zero instead of from the
cache offset, which is what a prefix hit did before patch 033 seeded the
state; the two modes side by side measure what the wrong positions cost.

Run it alone: it holds the GPU.

    python scripts/bench/mlx_prefix_tail_probe.py --model <dir> --ref-dump dfid-base.json --docs 6 --out tail-keep.json
    python scripts/bench/mlx_prefix_tail_probe.py --model <dir> --ref-dump dfid-base.json --docs 6 --reset --out tail-reset.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
import mlx_decode_census as census  # noqa: E402
from decode_path_precision import forward, logprobs  # noqa: E402


def clone(v):
    if isinstance(v, mx.array):
        return mx.array(v)
    if isinstance(v, (list, tuple)):
        return type(v)(clone(x) for x in v)
    return v


def snapshot(eng):
    snap = [clone(c.state) for c in eng.cache]
    return snap


def restore(eng, snap, reset):
    """reset=True clears mlx-vlm's saved position state, as the runner does at every new prefill."""
    eng.cache = eng.lm.make_cache()
    if reset:
        eng.lm._position_ids = None
        eng.lm._rope_deltas = None
    for c, s in zip(eng.cache, snap):
        c.state = clone(s)


def kl(p, q):
    return mx.sum(mx.exp(p) * (p - q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref-dump", required=True)
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--tails", default="1,2,4,8,16,48")
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--after", type=int, default=3)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--reset", action="store_true", help="clear mlx-vlm's position state before the tail forward")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tails = [int(x) for x in a.tails.split(",")]
    dump = json.load(open(a.ref_dump))
    docs = dump["docs"][: a.docs] if a.docs else dump["docs"]
    eng = census.Engine(census.resolve_model_dir(a.model))
    recs = []
    t0 = time.time()
    for di, d in enumerate(docs):
        prompt = d["ids"]; toks = [s["tid"] for s in d["steps"]]
        n = len(toks)
        cps = [j for j in range(max(tails), n - a.after, a.every)]
        need = sorted({j - k for j in cps for k in tails})
        # chain: lp[j] is the distribution after prompt + toks[:j]
        eng.new_cache()
        logits = forward(eng, prompt, a.f32)[0, -1, :]
        lps, snaps = [], {}
        for j in range(n):
            lp = logprobs(logits)
            mx.eval(lp, eng.cache_arrays())
            lps.append(lp)
            if j in need:
                snaps[j] = snapshot(eng)
                mx.eval([x for s in snaps[j] for x in (s if isinstance(s, (list, tuple)) else [s]) if isinstance(x, mx.array)])
            if j + 1 < n:
                logits = forward(eng, [toks[j]], a.f32)[0, -1, :]
        for j in cps:
            for k in tails:
                restore(eng, snaps[j - k], a.reset)
                logits = forward(eng, toks[j - k : j], a.f32)[0, -1, :]
                vals = [kl(lps[j], logprobs(logits))]
                for m in range(a.after):
                    logits = forward(eng, [toks[j + m]], a.f32)[0, -1, :]
                    vals.append(kl(lps[j + m + 1], logprobs(logits)))
                mx.eval(vals)
                recs.append({"doc": di, "domain": d["domain"], "j": j, "k": k, "kl": [v.item() for v in vals]})
        print(f"doc {di} {d['domain']:5s} checkpoints {len(cps)} {time.time() - t0:.0f}s", flush=True)
        del lps, snaps
        mx.clear_cache()
    out = {"f32": a.f32, "reset": a.reset, "tails": tails, "every": a.every, "recs": recs, "summary": {}}
    for k in tails:
        sel = [r for r in recs if r["k"] == k]
        row = {"n": len(sel)}
        for m in range(a.after + 1):
            v = sorted(r["kl"][m] for r in sel)
            row[f"since{m}"] = {"median": statistics.median(v), "mean": statistics.mean(v), "p90": v[int(0.9 * len(v))], "max": v[-1]}
        out["summary"][k] = row
        print(f"tail {k:3d} n {len(sel):4d} " + " | ".join(
            f"since {m}: median {row[f'since{m}']['median']:.2e} mean {row[f'since{m}']['mean']:.2e} max {row[f'since{m}']['max']:.2e}" for m in range(a.after + 1)), flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
