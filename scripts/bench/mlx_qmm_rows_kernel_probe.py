#!/usr/bin/env python3
"""Raw MLX: a multi-row 4-bit matvec kernel against the stock quantized matmul, weights cold.

Stock MLX multiplies a few activation rows through a quantized linear with
its one-row kernel launched once per row, so every row unpacks the weights
again; from about 15 rows it switches to a 32-row tile kernel. The kernel
here unpacks each weight once and accumulates every row from it: a
simdgroup owns ``RES`` output columns, each thread ``VPT`` input values per
block, and the lanes are summed at the end, the stock one-row layout with a
row loop inside.

    SGLANG_USE_MLX=1 python scripts/bench/mlx_qmm_rows_kernel_probe.py \
        --out-dim 17408 --in-dim 5120 --matrices 44 --rows 2,3,4,6,8,12,16 --out rows-kernel.json

``--check-only`` skips the timing and prints the error of both kernels against
a float32 product of the dequantized weights. No server may be running for
the timed sweep: the bench needs the GPU to itself.
"""
import argparse
import json
import statistics
import sys
import time

import mlx.core as mx

_SRC = """
    const uint lane = thread_index_in_simdgroup;
    const uint out_row = threadgroup_position_in_grid.y * (2 * RES) + simdgroup_index_in_threadgroup * RES;
    const uint kw = K_SIZE / 8;
    const uint kg = K_SIZE / G_SIZE;
    float acc[RES][M_ROWS];
    for (uint r = 0; r < RES; r++) {
        for (uint m = 0; m < M_ROWS; m++) {
            acc[r][m] = 0;
        }
    }
    for (uint k = 0; k < K_SIZE; k += VPT * 32) {
        const uint base = k + lane * VPT;
        float wf[RES][VPT];
        float sc[RES];
        float bi[RES];
        for (uint r = 0; r < RES; r++) {
            const uint row = out_row + r;
            const uint p = row * kw + base / 8;
            for (uint j = 0; j < VPT / 8; j++) {
                const uint pk = w[p + j];
                for (uint i = 0; i < 8; i++) {
                    wf[r][8 * j + i] = float((pk >> (4 * i)) & 0xf);
                }
            }
            sc[r] = float(scales[row * kg + base / G_SIZE]);
            bi[r] = float(biases[row * kg + base / G_SIZE]);
        }
        for (uint m = 0; m < M_ROWS; m++) {
            float xv[VPT];
            float s = 0;
            for (uint i = 0; i < VPT; i++) {
                xv[i] = float(x[m * K_SIZE + base + i]);
                s += xv[i];
            }
            for (uint r = 0; r < RES; r++) {
                float d = 0;
                for (uint i = 0; i < VPT; i++) {
                    d += wf[r][i] * xv[i];
                }
                acc[r][m] += sc[r] * d + bi[r] * s;
            }
        }
    }
    for (uint r = 0; r < RES; r++) {
        for (uint m = 0; m < M_ROWS; m++) {
            float t = simd_sum(acc[r][m]);
            if (lane == 0) {
                y[m * O_SIZE + out_row + r] = static_cast<T>(t);
            }
        }
    }
"""
_kernel = None


def rows_matvec(x, w, scales, biases, group_size=64, res=4, vpt=16):
    """``x`` (rows, in) times the 4-bit ``w`` (out, in / 8), transposed: (rows, out)."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="sglang_qmv_rows",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SRC,
        )
    rows, k = x.shape
    o = w.shape[0]
    if k % (vpt * 32) or o % (2 * res) or group_size % vpt or vpt % 8:
        raise ValueError(f"shape ({o}, {k}) does not fit res={res} vpt={vpt}")
    return _kernel(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("M_ROWS", rows), ("K_SIZE", k), ("O_SIZE", o),
                  ("G_SIZE", group_size), ("RES", res), ("VPT", vpt)],
        grid=(32, 2 * (o // (2 * res)), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, o)],
        output_dtypes=[x.dtype],
    )[0]


calls = {"kernel": 0, "stock": 0}


def install(min_rows=3, max_rows=14, res=4, vpt=16):
    """Route every 4-bit ``nn.QuantizedLinear`` call of ``min_rows``..``max_rows`` rows through the kernel."""
    import mlx.nn as nn

    stock_call = nn.QuantizedLinear.__call__

    def call(self, x):
        w = self["weight"]
        k = x.shape[-1]
        rows = x.size // k
        if (min_rows <= rows <= max_rows and self.bits == 4 and self.mode == "affine"
                and x.dtype == self["scales"].dtype and k % (vpt * 32) == 0
                and w.shape[0] % (2 * res) == 0 and self.group_size % vpt == 0):
            calls["kernel"] += 1
            y = rows_matvec(x.reshape(rows, k), w, self["scales"], self["biases"], self.group_size, res, vpt)
            y = y.reshape(*x.shape[:-1], w.shape[0])
            return y + self["bias"] if "bias" in self else y
        calls["stock"] += 1
        return stock_call(self, x)

    nn.QuantizedLinear.__call__ = call


def stock(x, w, scales, biases, group_size=64):
    return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=4)


def check(mats, rows, a, dt):
    """Error of the stock product and of the kernel against a float32 product."""
    w, s, b = mats[0]
    ref_w = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=a.group_size, bits=4)
    out = {}
    for m in rows:
        x = mx.random.normal((m, a.in_dim), key=mx.random.key(1000 + m)).astype(dt)
        ref = x.astype(mx.float32) @ ref_w.T
        scale = mx.abs(ref).mean().item()
        got = {"stock": stock(x, w, s, b, a.group_size),
               "kernel": rows_matvec(x, w, s, b, a.group_size, a.res, a.vpt)}
        row = {}
        for name, y in got.items():
            err = mx.abs(y.astype(mx.float32) - ref)
            row[name] = {"mean_rel": round(err.mean().item() / scale, 6), "max_rel": round(err.max().item() / scale, 6)}
        d = mx.abs(got["stock"].astype(mx.float32) - got["kernel"].astype(mx.float32))
        row["kernel_vs_stock"] = {"mean_rel": round(d.mean().item() / scale, 6),
                                  "equal_frac": round((d == 0).astype(mx.float32).mean().item(), 4)}
        out[str(m)] = row
        print(f"rows {m:3d}: against float32, stock mean {row['stock']['mean_rel']:.2e} max {row['stock']['max_rel']:.2e}, "
              f"kernel mean {row['kernel']['mean_rel']:.2e} max {row['kernel']['max_rel']:.2e}; "
              f"kernel equals stock on {100 * row['kernel_vs_stock']['equal_frac']:.1f} % of outputs", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dim", type=int, default=17408)
    ap.add_argument("--in-dim", type=int, default=5120)
    ap.add_argument("--matrices", type=int, default=44)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--rows", default="2,3,4,6,8,12,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--res", type=int, default=4, help="output columns per simdgroup")
    ap.add_argument("--vpt", type=int, default=16, help="input values per thread per block")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dt = getattr(mx, a.dtype)
    mats = []
    for i in range(a.matrices):
        w = mx.random.normal((a.out_dim, a.in_dim), key=mx.random.key(i)) * 0.02
        q = mx.quantize(w, group_size=a.group_size, bits=4)
        q = (q[0], q[1].astype(dt), q[2].astype(dt))
        mx.eval(q)
        mats.append(q)
        del w
    mx.clear_cache()
    packed_gb = sum(sum(x.nbytes for x in q) for q in mats) / 2**30
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    print(f"{a.matrices} matrices {a.out_dim}x{a.in_dim} at 4 bits in {a.dtype}, {packed_gb:.2f} GB per sweep, "
          f"res {a.res} vpt {a.vpt}, mlx {mx.__version__}", flush=True)
    result = {"mlx": mx.__version__, "out_dim": a.out_dim, "in_dim": a.in_dim, "matrices": a.matrices,
              "group_size": a.group_size, "dtype": a.dtype, "res": a.res, "vpt": a.vpt,
              "packed_gb": round(packed_gb, 3), "check": check(mats, rows, a, dt), "rows": {}}
    if not a.check_only:
        arms = {"stock": lambda x, q: stock(x, *q, a.group_size),
                "kernel": lambda x, q: rows_matvec(x, *q, a.group_size, a.res, a.vpt)}
        samples = {}
        for order in (rows, rows[::-1]):
            for m in order:
                x = mx.random.normal((m, a.in_dim)).astype(dt)
                mx.eval(x)
                for name in (("stock", "kernel") if order is rows else ("kernel", "stock")):
                    times = []
                    for _ in range(a.reps + 1):
                        t0 = time.perf_counter()
                        outs = [arms[name](x, q) for q in mats]
                        mx.eval(outs)
                        times.append((time.perf_counter() - t0) * 1e3)
                        del outs
                    samples.setdefault((m, name), []).extend(times[1:])
        for m in rows:
            st = statistics.median(samples[(m, "stock")])
            ke = statistics.median(samples[(m, "kernel")])
            result["rows"][str(m)] = {"stock_ms": round(st, 2), "kernel_ms": round(ke, 2),
                                      "kernel_over_stock": round(ke / st, 3),
                                      "kernel_gb_s": round(packed_gb / (ke / 1e3), 1)}
            print(f"rows {m:3d}: stock {st:8.2f} ms, kernel {ke:8.2f} ms per sweep, x{ke / st:.2f}, "
                  f"kernel reads {packed_gb / (ke / 1e3):6.1f} GB/s of packed weights", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
