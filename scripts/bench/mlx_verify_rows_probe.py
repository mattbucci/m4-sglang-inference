#!/usr/bin/env python3
"""Raw MLX: what a k-row forward on a cached prefix costs against the one-row decode step.

A speculative cycle verifies k drafted tokens in one forward. On a
memory-bound device that forward reads the shared weights once, so its cost
against k single steps is the ceiling on what a drafter could buy. Routed
MoE experts are the exception: every row gathers its own experts.

Each cycle feeds k tokens, takes the arg-max at every row and waits for it,
then feeds those k tokens back, so the cache deepens by k per cycle. Row
counts run in the order given and then reversed; both passes are pooled.

``--shape batch`` asks the other multi-row question: one new token of k
separate sequences, each on its own prompt and cache, which is what a
batched decode step of k concurrent requests would run.

    SGLANG_USE_MLX=1 python scripts/bench/mlx_verify_rows_probe.py --model <dir-or-repo> \
        --depths 24,8192 --rows 1,2,4,8,16 [--shape batch] [--fuse router,front,linears] --out rows.json

No server may be running: the probe needs the GPU to itself.
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_decode_census as census  # noqa: E402


def install_fusions(model, names):
    done = {}
    for f in names:
        if f == "router":
            from sglang.srt.hardware_backend.mlx.moe.fused_router import (
                patch_moe_blocks_with_fused_router as patch,
            )
        elif f == "front":
            from sglang.srt.hardware_backend.mlx.gated_delta_front import (
                patch_gated_delta_front as patch,
            )
        elif f == "linears":
            from sglang.srt.hardware_backend.mlx.fused_linears import (
                patch_fused_decode_linears as patch,
            )
        else:
            raise SystemExit(f"unknown fusion {f!r}")
        done[f] = patch(model)
    return done


def cycles(eng, y, n):
    """n verify cycles of y.shape[1] rows; returns the fed-back tokens and per-cycle ms."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = mx.argmax(eng.logits(y), axis=-1).astype(mx.int32)
        mx.eval(tok)
        out.append((time.perf_counter() - t0) * 1e3)
        y = tok
    return y, out


def batch_depth(eng, model_dir, depth, rows, a, rk):
    """One decode step of k separate sequences, each on its own ``depth``-token prompt."""
    row = {"depth": depth, "rows": {}}
    print(f"\n== depth {depth}, batched sequences", flush=True)
    pooled = {k: [] for k in rows}
    for k in rows + rows[::-1]:
        eng.new_cache()
        prompts = mx.array([census.prompt_ids(model_dir, depth, seed=7 + i) for i in range(k)], dtype=mx.int32)
        for i in range(0, depth - 1, a.chunk):
            part = prompts[:, i : min(i + a.chunk, depth - 1)]
            if eng.vlm:
                eng.lm(part, cache=eng.cache, skip_logits=True)
            elif eng.trunk_module is not None:
                eng.trunk_module(part, cache=eng.cache)
            else:
                eng.lm(part, cache=eng.cache)
            mx.eval(eng.cache_arrays())
            mx.clear_cache()
        y = prompts[:, depth - 1 :]
        if rk is not None:
            rk.calls.update(kernel=0, stock=0)
        y, _ = cycles(eng, y, a.warmup)
        y, ms = cycles(eng, y, a.iters)
        pooled[k].extend(ms)
        if rk is not None:
            row.setdefault("linear_calls", {})[str(k)] = dict(rk.calls)
    base = census.med(pooled[rows[0]]) / rows[0]
    for k in rows:
        s = census.summarize(pooled[k])
        s["ms_per_row"] = round(s["median_ms"] / k, 3)
        s["against_k_single_cycles"] = round(s["median_ms"] / (k * base), 3)
        s["aggregate_tok_s"] = round(1e3 * k / s["median_ms"], 1)
        row["rows"][str(k)] = s
        via = f", linear calls {row['linear_calls'][str(k)]}" if rk is not None else ""
        print(f"rows {k:3d}: {s['median_ms']:8.2f} ms per step, {s['aggregate_tok_s']:7.1f} tok/s aggregate, "
              f"x{s['against_k_single_cycles']:.3f} of {k} one-row steps{via}", flush=True)
    mx.clear_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="24,8192")
    ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=24, help="timed cycles per row count per pass")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--shape", default="seq", choices=["seq", "batch"],
                    help="seq: k new tokens of one sequence, a verify block or a prompt tail; "
                    "batch: one new token of k separate sequences, a batched decode step")
    ap.add_argument("--fuse", default="", help="comma list of router,front,linears")
    ap.add_argument("--rows-kernel", default="", help="min,max rows sent through the multi-row matvec kernel "
                    "of mlx_qmm_rows_kernel_probe.py, e.g. 3,14")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    mx.set_cache_limit(int(a.cache_limit_gb * 2**30))
    model_dir = census.resolve_model_dir(a.model)
    eng = census.Engine(model_dir)
    fused = install_fusions(eng.model, [f for f in a.fuse.split(",") if f])
    rk = None
    if a.rows_kernel:
        import mlx_qmm_rows_kernel_probe as rk
        lo, hi = (int(v) for v in a.rows_kernel.split(","))
        rk.install(lo, hi)
    print(f"loaded {a.model} vlm={eng.vlm} layers={len(eng.layers)} "
          f"active={mx.get_active_memory() / 2**30:.2f} GB fusions={fused}", flush=True)

    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    ids = census.prompt_ids(model_dir, max(depths) + max(rows))
    result = {"model": a.model, "mlx": mx.__version__, "fusions": {k: str(v) for k, v in fused.items()},
              "shape": a.shape, "rows_kernel": a.rows_kernel, "iters": a.iters, "warmup": a.warmup, "depths": []}

    for depth in depths:
        if a.shape == "batch":
            result["depths"].append(batch_depth(eng, model_dir, depth, rows, a, rk))
            continue
        pre_s, _ = eng.prefill(ids[:depth], a.chunk)
        row = {"depth": depth, "prefill_s": round(pre_s, 2), "rows": {}}
        print(f"\n== depth {depth}: prefill {pre_s:.1f}s", flush=True)

        y1 = mx.array([ids[depth - 1]], dtype=mx.int32)
        y1, _ = census.loop_pipelined(eng.step_full, y1, 16)
        y1, piped = census.loop_pipelined(eng.step_full, y1, 96)
        row["one_row_pipelined_ms"] = census.med(piped)
        fed = depth + 16 + 96 + 2

        pooled = {k: [] for k in rows}
        for k in rows + rows[::-1]:
            y = mx.array(ids[depth : depth + k], dtype=mx.int32)[None]
            y, _ = cycles(eng, y, a.warmup)
            y, ms = cycles(eng, y, a.iters)
            pooled[k].extend(ms)
            if rk is not None:
                row.setdefault("linear_calls", {})[str(k)] = dict(rk.calls)
                rk.calls.update(kernel=0, stock=0)
            fed += k * (a.warmup + a.iters)
        mx.eval(eng.cache_arrays())
        row["cache_tokens_at_end"] = fed

        base = census.med(pooled[rows[0]]) / rows[0]
        for k in rows:
            s = census.summarize(pooled[k])
            s["ms_per_row"] = round(s["median_ms"] / k, 3)
            s["against_k_single_cycles"] = round(s["median_ms"] / (k * base), 3)
            row["rows"][str(k)] = s
            via = f", linear calls {row['linear_calls'][str(k)]}" if rk is not None else ""
            print(f"rows {k:3d}: {s['median_ms']:8.2f} ms per cycle, {s['ms_per_row']:7.3f} ms per row, "
                  f"x{s['against_k_single_cycles']:.3f} of {k} one-row cycles{via}", flush=True)
        print(f"one-row pipelined step {row['one_row_pipelined_ms']:.2f} ms; cache ends at {fed} tokens", flush=True)
        result["depths"].append(row)
        mx.clear_cache()

    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
