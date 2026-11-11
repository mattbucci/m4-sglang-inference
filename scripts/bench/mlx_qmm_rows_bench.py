#!/usr/bin/env python3
"""Raw MLX: stock 4-bit quantized matmul against the number of activation rows, weights cold.

One decode step multiplies one activation row through every linear; a
batched decode, a short prompt tail or a verify block multiplies a few. The
weights are read from DRAM either way, so a few rows should cost little more
than one until arithmetic becomes the limit. This sweeps the row count over
enough distinct matrices that every read misses the caches.

    SGLANG_USE_MLX=1 python scripts/bench/mlx_qmm_rows_bench.py --out-dim 17408 --in-dim 5120 \
        --matrices 44 --lead-dim --rows 1,2,3,4,5,6,8,12,15,16,24,32,64 --out rows.json

No server may be running: the bench needs the GPU to itself.
"""
import argparse
import json
import statistics
import sys
import time

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dim", type=int, default=16384)
    ap.add_argument("--in-dim", type=int, default=4096)
    ap.add_argument("--matrices", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--rows", default="1,2,3,4,5,6,8,12,16,24,32,64")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dtype", default="bfloat16", help="activation dtype")
    ap.add_argument("--scale-dtype", default="", help="scale and bias dtype, the activation dtype by default; "
                    "float32 against bfloat16 activations promotes the product to float32")
    ap.add_argument("--lead-dim", action="store_true", help="activations shaped (1, rows, in) as a model feeds them")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dt = getattr(mx, a.dtype)
    sdt = getattr(mx, a.scale_dtype or a.dtype)
    mats = []
    for i in range(a.matrices):
        w = mx.random.normal((a.out_dim, a.in_dim), key=mx.random.key(i)) * 0.02
        q = mx.quantize(w, group_size=a.group_size, bits=a.bits)
        # mx.quantize hands back float32 scales; a converted checkpoint
        # stores them in the model dtype.
        q = (q[0], q[1].astype(sdt), q[2].astype(sdt))
        mx.eval(q)
        mats.append(q)
        del w
    mx.clear_cache()
    packed_gb = sum(sum(x.nbytes for x in q) for q in mats) / 2**30
    print(f"{a.matrices} matrices {a.out_dim}x{a.in_dim} at {a.bits} bits, activations {a.dtype}, scales {a.scale_dtype or a.dtype}, {packed_gb:.2f} GB per sweep, "
          f"mlx {mx.__version__}", flush=True)

    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    result = {"mlx": mx.__version__, "out_dim": a.out_dim, "in_dim": a.in_dim, "matrices": a.matrices,
              "bits": a.bits, "group_size": a.group_size, "lead_dim": a.lead_dim, "dtype": a.dtype, "scale_dtype": a.scale_dtype or a.dtype, "packed_gb": round(packed_gb, 3), "rows": {}}
    one = None
    for order in (rows, rows[::-1]):
        for m in order:
            x = mx.random.normal((1, m, a.in_dim) if a.lead_dim else (m, a.in_dim)).astype(dt)
            mx.eval(x)
            times = []
            for _ in range(a.reps + 1):
                t0 = time.perf_counter()
                outs = [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=a.group_size, bits=a.bits)
                        for (w, s, b) in mats]
                mx.eval(outs)
                times.append((time.perf_counter() - t0) * 1e3)
                del outs
            result["rows"].setdefault(str(m), []).extend(times[1:])
    for m in rows:
        ms = statistics.median(result["rows"][str(m)])
        if one is None:
            one = ms
        row = {"sweep_ms": round(ms, 2), "gb_s": round(packed_gb / (ms / 1e3), 1),
               "ms_per_row": round(ms / m, 2), "against_one_row": round(ms / one, 3)}
        result["rows"][str(m)] = row
        print(f"rows {m:3d}: {ms:8.2f} ms per sweep, {row['gb_s']:6.1f} GB/s of packed weights, "
              f"{row['ms_per_row']:7.2f} ms per row, x{row['against_one_row']:.2f} of one row", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
