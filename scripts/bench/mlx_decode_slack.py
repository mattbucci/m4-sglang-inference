#!/usr/bin/env python3
"""Host-slack probe for the pipelined MLX decode loop (raw MLX, no server).

MLX stops encoding a graph whenever more than ten command buffers are in
flight, so `mx.async_eval` returns with at most that much GPU work still
queued. Host time spent between two launches beyond that lookahead leaves
the GPU idle. This probe measures three things per decode step:

  phases   graph build, `mx.async_eval`, token wait, at zero extra delay
  slack    step time against a busy-wait host delay injected after the
           token wait; flat until the queued work runs out, then slope 1
  chain    K steps built lazily and evaluated together, which hides the
           build entirely and gives the per-step bound of encode and GPU

Run it under different `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`
settings to see how the buffer limits move the knee. A single-buffer run
(both limits very large, depth 24 only: large buffers raise prefill peak
memory) makes `mx.async_eval` return after pure encode, which separates
CPU encode time from GPU time.

usage: mlx_decode_slack.py --tag base --depths 24,8192 --out slack.jsonl
       mlx_decode_slack.py --report slack.jsonl

The report prints, per run, the step at each delay (best of the forward and
reversed pass) and the lookahead: host time per step the queued GPU work
covers, read from the slope-1 tail as build + delay - (step - floor).
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402

import mlx_decode_census as c  # noqa: E402


def spin(ms: float) -> None:
    if ms <= 0:
        return
    end = time.perf_counter() + ms / 1e3
    while time.perf_counter() < end:
        pass


def loop_phases(step, y, n, delay_ms=0.0):
    nxt = step(y)
    mx.async_eval(nxt)
    rows = []
    for _ in range(n):
        t0 = time.perf_counter()
        ahead = step(nxt)
        t1 = time.perf_counter()
        mx.async_eval(ahead)
        t2 = time.perf_counter()
        nxt.item()
        t3 = time.perf_counter()
        spin(delay_ms)
        t4 = time.perf_counter()
        rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t0))
        nxt = ahead
    mx.eval(nxt)
    med = [statistics.median(r[i] for r in rows) * 1e3 for i in range(4)]
    return nxt, {"build": med[0], "async_eval": med[1], "wait": med[2], "step": med[3]}


def loop_chain(step, y, k, n):
    builds, evals = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        cur = y
        for _ in range(k):
            cur = step(cur)
        t1 = time.perf_counter()
        mx.eval(cur)
        t2 = time.perf_counter()
        builds.append((t1 - t0) / k)
        evals.append((t2 - t1) / k)
        y = cur
    return y, {"build": statistics.median(builds) * 1e3, "eval": statistics.median(evals) * 1e3}


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    delays = sorted({s["delay"] for r in rows for s in r["slack"]})
    head = ["tag", "depth", "build", "encode+gpu k=8"] + [f"+{d:g}" for d in delays] + ["lookahead"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        best = {}
        for s in r["slack"]:
            best[s["delay"]] = min(best.get(s["delay"], 1e9), s["step"])
        build = statistics.median(s["build"] for s in r["slack"])
        floor = min(v for d, v in best.items() if d <= 0.25)
        tail = [d - (best[d] - floor) for d in delays[-2:]]
        knee = statistics.mean(tail)
        covered = best[delays[-1]] - floor < 0.3
        look = f">{build + delays[-1]:.1f}" if covered else f"{build + max(knee, 0.0):.1f}"
        chain = {c["k"]: c["eval"] for c in r["chain"]}
        cells = [r["tag"], str(r["depth"]), f"{build:.2f}", f"{chain.get(8, float('nan')):.2f}"]
        cells += [f"{best[d]:.2f}" for d in delays] + [look]
        print("| " + " | ".join(cells) + " |")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        return report(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--depths", default="24,8192")
    ap.add_argument("--delays", default="0,0.25,0.5,0.75,1,1.5,2,3,4")
    ap.add_argument("--chains", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mx.set_cache_limit(4 * 2**30)
    d = c.resolve_model_dir(a.model)
    eng = c.Engine(d)
    env = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
    r3 = lambda m: {k: round(v, 3) for k, v in m.items()}
    for depth in [int(x) for x in a.depths.split(",")]:
        ids = c.prompt_ids(d, depth)
        eng.prefill(ids, a.chunk)
        y = mx.array([ids[-1]], dtype=mx.int32)
        y, _ = loop_phases(eng.step_full, y, 32)
        row = {"tag": a.tag, "env": env, "depth": depth, "slack": [], "chain": []}
        delays = [float(x) for x in a.delays.split(",")]
        for delay in delays + delays[::-1]:
            y, m = loop_phases(eng.step_full, y, a.steps, delay)
            row["slack"].append({"delay": delay, **r3(m)})
        for k in [int(x) for x in a.chains.split(",")]:
            y, m = loop_chain(eng.step_full, y, k, max(8, a.steps // k))
            row["chain"].append({"k": k, **r3(m)})
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
