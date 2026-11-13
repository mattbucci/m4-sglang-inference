#!/usr/bin/env python3
"""Raw-MLX probe: what the GPU and the memory bus sustain with no model on top.

Two limits bound serving on unified memory. A decode step streams every
weight and every cached key and value once, so its floor is bytes over the
read rate (`bandwidth` rows). A prefill chunk reuses each weight across a
thousand rows, so its floor is arithmetic over the matmul rate (`compute`
rows). ceiling_report.py divides a checkpoint's bytes and FLOPs by the best
row of each kind.

Run it alone: no server, display asleep. Every case is the best of REPEATS.

usage: mlx_hardware_ceiling_probe.py <out json>
env:   GB (array size for the stream cases, default 4), REPEATS (default 5)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

GB = float(os.environ.get("GB", "4"))
REPEATS = int(os.environ.get("REPEATS", "5"))


def best(fn):
    fn()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def gpu_stream():
    n = int(GB * 1e9 / 4)
    x = mx.arange(n, dtype=mx.uint32)
    mx.eval(x)
    half = x[: n // 2]
    rows = []
    t = best(lambda: mx.eval(mx.max(x)))
    rows.append({"case": "gpu reduce (max over uint32)", "bytes": n * 4, "seconds": t})
    t = best(lambda: mx.eval(mx.sum(x)))
    rows.append({"case": "gpu reduce (sum over uint32)", "bytes": n * 4, "seconds": t})
    t = best(lambda: mx.eval(half + 1))
    rows.append({"case": "gpu map (read half, write half)", "bytes": n * 4, "seconds": t})
    del x, half
    f = mx.ones((int(GB * 1e9 / 2),), dtype=mx.bfloat16)
    mx.eval(f)
    t = best(lambda: mx.eval(mx.sum(f)))
    rows.append({"case": "gpu reduce (sum over bf16)", "bytes": f.size * 2, "seconds": t})
    del f
    mx.clear_cache()
    return rows


def qmm_stream(dim, count, fan):
    """One-row quantized matmuls over `count` distinct 4-bit weights: the decode read."""
    weights = []
    for _ in range(count):
        w = mx.random.normal((dim, dim)).astype(mx.float16)
        weights.append(mx.quantize(w, group_size=64, bits=4))
    mx.eval(weights)
    nbytes = sum(q.nbytes + s.nbytes + b.nbytes for q, s, b in weights)
    x = mx.random.normal((1, dim)).astype(mx.bfloat16)
    mx.eval(x)

    def chain():
        h = x
        for q, s, b in weights:
            h = mx.quantized_matmul(h, q, scales=s, biases=b, transpose=True, group_size=64, bits=4) * 0.01
        mx.eval(h)

    def fanout():
        outs = [mx.quantized_matmul(x, q, scales=s, biases=b, transpose=True, group_size=64, bits=4)
                for q, s, b in weights]
        mx.eval(outs)

    t = best(fanout if fan else chain)
    del weights
    mx.clear_cache()
    shape = "independent" if fan else "chained"
    return {"case": f"qmm one row, {count} x {dim}x{dim} 4-bit, {shape}", "bytes": nbytes, "seconds": t}


def cpu_stream():
    n = int(GB * 1e9 / 8)
    a = np.ones(n, dtype=np.float64)
    rows = []
    t = best(lambda: a.sum())
    rows.append({"case": "cpu reduce, 1 thread", "bytes": n * 8, "seconds": t})
    for threads in (4, 10, 14):
        parts = np.array_split(a, threads)
        with ThreadPoolExecutor(threads) as pool:
            t = best(lambda: list(pool.map(np.sum, parts)))
        rows.append({"case": f"cpu reduce, {threads} threads", "bytes": n * 8, "seconds": t})
    return rows


def gemm(dtype, rows, dim):
    a = mx.random.normal((rows, dim)).astype(dtype)
    b = mx.random.normal((dim, dim)).astype(dtype)
    mx.eval(a, b)
    t = best(lambda: mx.eval(a @ b))
    return {"case": f"matmul {rows}x{dim} @ {dim}x{dim} {str(dtype).split('.')[-1]}",
            "flops": 2.0 * rows * dim * dim, "seconds": t}


def qgemm(rows, dim, bits):
    w = mx.random.normal((dim, dim)).astype(mx.float16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    x = mx.random.normal((rows, dim)).astype(mx.bfloat16)
    mx.eval(q, s, b, x)
    t = best(lambda: mx.eval(mx.quantized_matmul(x, q, scales=s, biases=b, transpose=True,
                                                 group_size=64, bits=bits)))
    return {"case": f"qmm {rows} rows, {dim}x{dim} {bits}-bit", "flops": 2.0 * rows * dim * dim, "seconds": t}


def main():
    out = sys.argv[1]
    bandwidth = gpu_stream()
    for dim, count in ((2048, 256), (4096, 96), (8192, 24)):
        for fan in (False, True):
            bandwidth.append(qmm_stream(dim, count, fan))
    bandwidth += cpu_stream()
    for row in bandwidth:
        row["gb_per_s"] = round(row["bytes"] / 1e9 / row["seconds"], 1)
        print(f"{row['case']:58s} {row['gb_per_s']:7.1f} GB/s")
    compute = [gemm(mx.float32, 4096, 4096), gemm(mx.float16, 4096, 4096), gemm(mx.bfloat16, 4096, 4096),
               gemm(mx.bfloat16, 1024, 2048), gemm(mx.bfloat16, 1024, 5120)]
    for rows in (1024, 2048, 4096):
        for dim in (2048, 5120):
            compute.append(qgemm(rows, dim, 4))
    compute.append(qgemm(1024, 2048, 8))
    for row in compute:
        row["gflops"] = round(row["flops"] / 1e9 / row["seconds"], 1)
        print(f"{row['case']:58s} {row['gflops']:7.1f} GFLOPS")
    result = {"mlx": mx.__version__, "array_gb": GB, "repeats": REPEATS,
              "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
              "bandwidth": bandwidth, "compute": compute}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
