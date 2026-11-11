#!/usr/bin/env python3
"""Summarise a prefill census (SERVE_CENSUS_PREFILL rows from
scripts/bench/serve_census/sitecustomize.py): where each served prefill chunk
spends its time.

A request is a run of chunk launches with no idle gap between them. Per
request: tokens over the summed chunk time, and per chunk the scheduler gap
before the launch (GPU idle), the launch (graph build and submission) and the
wait for the chunk's token. Evaluation time is the tracked-state snapshot (it
evaluates synchronously, so with the radix cache on the chunk's forward runs
inside it), the launch's `mx.async_eval` and the wait: what a raw chunked
prefill spends in `mx.eval`. The rest is host time: the gap and the graph
build. `evals` is the count of synchronous `mx.eval` calls per chunk launch
(rows written before the hook counted them have none).

usage: prefill_census_report.py <census jsonl> [summary json] [--chunks]
"""
import json
import sys

PARTS = ("gap_ms", "launch_ms", "trunk_ms", "pool_sync_ms", "track_ms", "async_eval_ms", "wait_ms")


def requests(rows, idle_ms=250.0):
    cur = []
    for r in rows:
        if cur and (r.get("gap_ms") is None or r["gap_ms"] > idle_ms or r["pid"] != cur[-1]["pid"]):
            yield cur
            cur = []
        cur.append(r)
    if cur:
        yield cur


def summarise(chunks):
    tot = {k: sum(c.get(k) or 0.0 for c in chunks[1:]) for k in PARTS}
    first = chunks[0]
    for k in PARTS[1:]:
        tot[k] += first.get(k) or 0.0
    tokens = sum(c["tokens"] for c in chunks)
    wall = tot["gap_ms"] + tot["launch_ms"] + tot["wait_ms"]
    evals = tot["track_ms"] + tot["async_eval_ms"] + tot["wait_ms"]
    return {
        "chunks": len(chunks), "tokens": tokens, "wall_s": round(wall / 1e3, 3),
        "tok_s": round(tokens / (wall / 1e3), 1),
        "chunk_ms": round(wall / len(chunks), 2), "eval_ms": round(evals / len(chunks), 2),
        "host_ms": round((wall - evals) / len(chunks), 2), "host_share": round(1.0 - evals / wall, 4),
        "evals_per_chunk": round(sum(c.get("evals") or 0 for c in chunks) / len(chunks), 1)
        if "evals" in chunks[0] else None,
        "per_chunk_ms": {k[:-3]: round(v / len(chunks), 2) for k, v in tot.items()},
        "last_chunk": {k[:-3]: chunks[-1].get(k) for k in PARTS} | {"tokens": chunks[-1]["tokens"]},
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = [json.loads(line) for line in open(args[0]) if line.strip()]
    out = []
    for chunks in requests(rows):
        if len(chunks) < 2:
            continue
        s = summarise(chunks)
        out.append(s)
        p = s["per_chunk_ms"]
        print(f"{s['tokens']:>7} tokens in {s['chunks']} chunks: {s['tok_s']} tok/s served, "
              f"{s['chunk_ms']} ms per chunk, {s['eval_ms']} in evaluations, {s['host_ms']} host "
              f"({100 * s['host_share']:.2f} %)")
        print(f"        per chunk ms: gap {p['gap']}, launch {p['launch']} (trunk {p['trunk']}, pool "
              f"{p['pool_sync']}, track {p['track']}, async_eval {p['async_eval']}), wait {p['wait']}"
              + ("" if s["evals_per_chunk"] is None else f"; {s['evals_per_chunk']} synchronous evals"))
        if "--chunks" in sys.argv:
            for c in chunks:
                print("        " + " ".join(f"{k[:-3]}={c.get(k)}" for k in PARTS) + f" tokens={c['tokens']}")
    if len(args) > 1:
        json.dump({"requests": out}, open(args[1], "w"), indent=2)


if __name__ == "__main__":
    main()
