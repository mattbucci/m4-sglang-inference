#!/usr/bin/env python3
"""Raw-MLX probe: chunked prefill rate against the command-buffer limits and
the MLX buffer cache.

The command-buffer limits (MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER) are
read once at start, so each setting is its own process. The buffer cache limit
and whether the cache is cleared after every chunk vary inside the process
(ARMS). A server keeps the cache between chunks; `mlx_decode_census.py` clears
it. Repeat the first arm last: rates differ between processes by more than
most arms differ inside one.

One row per arm, depth and pass: prompt tokens over wall time, the rate of the
first and of the last (deepest) chunk, MLX's peak over the loaded weights and
the cache it holds at the end.

usage: mlx_prefill_limit_probe.py <depth:passes csv> <out jsonl>
env:   MODEL (default RecursiveSI/flint-int4), CHUNK (1024),
       ARMS (csv of <cache GB>:<clear 0|1>, default 2:1)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx


def prefill(eng, ids, chunk, clear):
    eng.new_cache()
    t0 = time.perf_counter()
    rates = []
    for i in range(0, len(ids), chunk):
        part = mx.array(ids[i : i + chunk], dtype=mx.int32)[None]
        tc = time.perf_counter()
        eng.trunk_module(part, cache=eng.cache)
        mx.eval(eng.cache_arrays())
        rates.append(part.shape[1] / (time.perf_counter() - tc))
        if clear:
            mx.clear_cache()
    return time.perf_counter() - t0, rates


def main():
    import mlx_decode_census as c

    depths, out = [tuple(int(v) for v in x.split(":")) for x in sys.argv[1].split(",")], sys.argv[2]
    model = os.environ.get("MODEL", "RecursiveSI/flint-int4")
    chunk = int(os.environ.get("CHUNK", "1024"))
    arms = [(float(a.split(":")[0]), a.split(":")[1] == "1")
            for a in os.environ.get("ARMS", "2:1").split(",")]
    mx.set_cache_limit(2 * 2**30)
    d = c.resolve_model_dir(model)
    eng = c.Engine(d)
    prefill(eng, c.prompt_ids(d, 1024), chunk, True)
    base = mx.get_active_memory()
    for cache_gb, clear in arms:
        mx.set_cache_limit(int(cache_gb * 2**30))
        mx.clear_cache()
        for depth, passes in depths:
            ids = c.prompt_ids(d, depth)
            for p in range(passes):
                mx.reset_peak_memory()
                secs, rates = prefill(eng, ids, chunk, clear)
                row = {"model": os.path.basename(model.rstrip("/")), "depth": len(ids), "chunk": chunk,
                       "pass": p, "cache_limit_gb": cache_gb, "clear_each_chunk": clear,
                       "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")},
                       "prefill_tok_s": round(len(ids) / secs, 1), "seconds": round(secs, 3),
                       "first_chunk_tok_s": round(rates[0], 1), "last_chunk_tok_s": round(rates[-1], 1),
                       "peak_over_weights_gb": round((mx.get_peak_memory() - base) / 2**30, 3),
                       "cache_gb": round(mx.get_cache_memory() / 2**30, 3)}
                print(json.dumps(row), flush=True)
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
