#!/usr/bin/env python3
"""One raw-MLX decode measurement under the caller's MLX_* environment.

MLX reads its command-buffer limits (MLX_MAX_OPS_PER_BUFFER,
MLX_MAX_MB_PER_BUFFER) and MLX_METAL_FAST_SYNCH once per process, so every
arm is a fresh process; mlx_decode_knob_sweep.sh interleaves the arms.

usage: mlx_decode_knobs.py <tag> <depths csv> <out jsonl> [model]
Per depth: prefill in 2048-token chunks (peak memory recorded), 32 warm
steps, three 96-step pipelined segments and one 96-step sync segment.
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx_decode_census as c

tag, depths, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "mlx-community/Qwen3.6-35B-A3B-4bit"
mx.set_cache_limit(4 * 2**30)
d = c.resolve_model_dir(model)
eng = c.Engine(d)
env = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
for depth in depths:
    ids = c.prompt_ids(d, depth)
    mx.reset_peak_memory()
    pre, _ = eng.prefill(ids, 2048)
    prefill_peak = mx.get_peak_memory() / 2**30
    y = mx.array([ids[-1]], dtype=mx.int32)
    y, _ = c.loop_pipelined(eng.step_full, y, 32)
    segs = []
    for _ in range(3):
        y, t = c.loop_pipelined(eng.step_full, y, 96); segs.append(statistics.median(t))
    y, t, _ = c.loop_sync(eng.step_full, y, 96); sync = statistics.median(t)
    row = {"tag": tag, "env": env, "depth": depth, "prefill_s": round(pre, 2), "prefill_tok_s": round(depth / pre, 1),
           "pipelined_ms": round(statistics.median(segs), 3), "pipelined_segs": [round(s, 3) for s in segs],
           "sync_ms": round(sync, 3), "prefill_peak_gb": round(prefill_peak, 2)}
    print(json.dumps(row), flush=True)
    with open(out, "a") as f: f.write(json.dumps(row) + "\n")
