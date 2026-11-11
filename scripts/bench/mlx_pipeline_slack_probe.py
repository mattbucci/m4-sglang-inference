#!/usr/bin/env python3
"""Raw-MLX probe: how much host time a pipelined decode step hides, and
whether submitting the step in parts hides more.

A pipelined loop builds the graph of the next step while the GPU still runs
the current one. ``mx.async_eval`` returns once the stream is down to a few
queued tasks, so what the GPU has left at that moment is all that covers the
host work before the next submission. Host time past it is GPU idle time. A
serving step does more host work than a raw loop: the scheduler's bookkeeping
plus a longer graph build.

Arms are the cross product of
  delay   a busy-wait of D ms before each graph build, standing in for the
          host work a serving step adds
  splits  the step is submitted in S+1 parts: ``mx.async_eval`` on the hidden
          state after S evenly spaced layers, then on the token as usual

A flat line over delay means the host work is hidden; a slope of one means the
GPU waits for it. Every arm re-prefills the same prompt and decodes greedily,
so the token ids of all arms must be equal (same graph, same order).

usage: mlx_pipeline_slack_probe.py <depths csv> <out jsonl>
env:   MODEL (default RecursiveSI/flint-int4), CHUNK (prefill chunk, 1024),
       DELAYS (ms csv, default 0,0.5,1,1.5,2), SPLITS (csv, default 0,1,2,4),
       STEPS (per segment, 96), CACHE_GB (MLX buffer cache limit, 2)
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx

SUBMIT_MS = [0.0]


def _arrays(out):
    if isinstance(out, mx.array):
        return [out]
    if isinstance(out, (list, tuple)):
        return [a for x in out for a in _arrays(x)]
    return []


def _submitting(base):
    def call(self, *args, **kwargs):
        out = base.__call__(self, *args, **kwargs)
        t0 = time.perf_counter()
        mx.async_eval(*_arrays(out))
        SUBMIT_MS[0] += (time.perf_counter() - t0) * 1e3
        return out

    return type("Submitting" + base.__name__, (base,), {"__call__": call})


def set_splits(layers, stock, splits):
    """Submit after ``splits`` evenly spaced layers; 0 restores every layer."""
    for layer, cls in zip(layers, stock):
        layer.__class__ = cls
    n = len(layers)
    marks = [round(n * (i + 1) / (splits + 1)) - 1 for i in range(splits)]
    for i in marks:
        layers[i].__class__ = _submitting(stock[i])
    return marks


def loop(eng, y, n, delay_ms):
    nxt = eng.step_full(y)
    mx.async_eval(nxt)
    rows, toks = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        if delay_ms:
            until = t0 + delay_ms * 1e-3
            while time.perf_counter() < until:
                pass
        SUBMIT_MS[0] = 0.0
        t1 = time.perf_counter()
        ahead = eng.step_full(nxt)
        t2 = time.perf_counter()
        mx.async_eval(ahead)
        t3 = time.perf_counter()
        toks.append(int(nxt.item()))
        t4 = time.perf_counter()
        mid = SUBMIT_MS[0]
        rows.append(((t4 - t0) * 1e3, (t2 - t1) * 1e3 - mid, mid + (t3 - t2) * 1e3, (t4 - t3) * 1e3))
        nxt = ahead
    mx.eval(nxt)
    return nxt, rows, toks


def main():
    import mlx_decode_census as c

    depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
    model = os.environ.get("MODEL", "RecursiveSI/flint-int4")
    chunk = int(os.environ.get("CHUNK", "1024"))
    steps = int(os.environ.get("STEPS", "96"))
    delays = [float(x) for x in os.environ.get("DELAYS", "0,0.5,1,1.5,2").split(",")]
    splits = [int(x) for x in os.environ.get("SPLITS", "0,1,2,4").split(",")]
    mx.set_cache_limit(int(float(os.environ.get("CACHE_GB", "2")) * 2**30))
    d = c.resolve_model_dir(model)
    eng = c.Engine(d)
    stock = [type(layer) for layer in eng.layers]
    arms = [(s, dl) for s in splits for dl in delays]
    for depth in depths:
        ids = c.prompt_ids(d, depth)
        ref = None
        for s, dl in arms + arms[::-1]:
            set_splits(eng.layers, stock, 0)
            eng.prefill(ids, chunk)
            marks = set_splits(eng.layers, stock, s)
            y = mx.array([ids[-1]], dtype=mx.int32)
            y, _, toks = loop(eng, y, 32, dl)
            segs, parts = [], []
            for _ in range(3):
                y, rows, t = loop(eng, y, steps, dl)
                toks += t
                segs.append(statistics.median(r[0] for r in rows))
                parts.append([statistics.median(r[i] for r in rows) for i in (1, 2, 3)])
            ref = ref or toks
            wall = statistics.median(segs)
            host, submit, item = (statistics.median(p[i] for p in parts) for i in range(3))
            row = {"model": os.path.basename(model.rstrip("/")), "depth": depth, "chunk": chunk,
                   "splits": s, "submit_after_layers": marks, "delay_ms": dl,
                   "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")},
                   "step_ms": round(wall, 3), "step_segs": [round(x, 3) for x in segs],
                   "step_less_delay_ms": round(wall - dl, 3), "tok_s": round(1000.0 / wall, 2),
                   "build_ms": round(host, 3), "submit_ms": round(submit, 3),
                   "item_ms": round(item, 3), "tokens": len(toks), "tokens_equal_first_arm": toks == ref}
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
    set_splits(eng.layers, stock, 0)


if __name__ == "__main__":
    main()
