#!/usr/bin/env python3
"""Raw MLX: decode attention against a deep KV cache, in the shape a serving
step runs it (synthetic tensors, no model).

mlx_sdpa_kv_microbench.py builds K/V of exactly n keys and evaluates all
layers in one mx.eval. A serving step differs in two ways, and this probe
times each:

  layout   the attention cache is a pre-sized buffer (1, kv_heads, capacity,
           head_dim) and the kernel reads the slice [:, :, :n, :], so the
           stride between heads is capacity * head_dim, not n * head_dim.
           Capacity 0 means exact-size tensors.
  SPLIT=1  one mx.eval per layer. A step meets its attention layers one at a
           time between other work; several two-pass kernels queued back to
           back lose more on large blocks than one kernel alone.

Each row is the median over REPS (default 7) of the time for LAYERS (default
10) layers at one block count; block count 0 is MLX's own ladder, any other
value is set through MLX_SDPA_BLOCKS, which MLX reads on every kernel call.

Shapes follow Qwen3.6-35B-A3B: 16 query heads, 2 KV heads, head dim 256.

usage: mlx_sdpa_layout_probe.py <depths csv> <out jsonl> <blocks csv> <capacities csv>
  e.g. SPLIT=1 mlx_sdpa_layout_probe.py 196608,251475 out.jsonl 0,1536,2048 262208
"""
import json, os, statistics, sys, time
import mlx.core as mx

QH, KVH, D = 16, 2, 256
LAYERS = int(os.environ.get("LAYERS", "10"))
REPS = int(os.environ.get("REPS", "7"))
SPLIT = os.environ.get("SPLIT") == "1"
SCALE = D ** -0.5
BF = mx.bfloat16
depths = [int(x) for x in sys.argv[1].split(",")]
out_path = sys.argv[2]
blocks_list = [int(x) for x in sys.argv[3].split(",")]
caps = [int(x) for x in sys.argv[4].split(",")]
mx.random.seed(11)


def timed(layers, blocks):
    if blocks:
        os.environ["MLX_SDPA_BLOCKS"] = str(blocks)
    else:
        os.environ.pop("MLX_SDPA_BLOCKS", None)
    outs = [mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=None) for q, k, v in layers]
    t0 = time.perf_counter()
    if SPLIT:
        for o in outs:
            mx.eval(o)
    else:
        mx.eval(outs)
    return (time.perf_counter() - t0) * 1e3


for cap in caps:
    top = cap or max(depths)
    bufs = []
    for _ in range(LAYERS):
        q = (mx.random.normal((1, QH, 1, D)) * 1.5).astype(BF)
        kb = mx.random.normal((1, KVH, top, D)).astype(BF)
        vb = mx.random.normal((1, KVH, top, D)).astype(BF)
        mx.eval(q, kb, vb)
        bufs.append((q, kb, vb))
    for n in depths:
        if n > top:
            continue
        if cap:
            layers = [(q, kb[:, :, :n, :], vb[:, :, :n, :]) for q, kb, vb in bufs]
        else:
            layers = [(q, mx.contiguous(kb[:, :, :n, :]), mx.contiguous(vb[:, :, :n, :])) for q, kb, vb in bufs]
        mx.eval([t for L in layers for t in L])
        for b in blocks_list:  # compile and touch every tensor once
            timed(layers, b)
        times = {b: [] for b in blocks_list}
        for rep in range(REPS):
            for b in (blocks_list if rep % 2 == 0 else blocks_list[::-1]):
                times[b].append(timed(layers, b))
        gb = KVH * D * 4 * n * LAYERS / 1e9
        for b in blocks_list:
            ms = statistics.median(times[b])
            row = {"eval": "per_layer" if SPLIT else "joint", "layers": LAYERS,
                   "layout": f"cap{cap}" if cap else "exact", "capacity": cap or n, "depth": n,
                   "blocks": b or "stock", "ms_all_layers": round(ms, 3), "ms_min": round(min(times[b]), 3),
                   "kv_gb_read": round(gb, 3), "gb_per_s": round(gb / (ms / 1e3), 1),
                   "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_") and k != "MLX_SDPA_BLOCKS"}}
            print(json.dumps(row), flush=True)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        del layers
    del bufs
    mx.clear_cache()
