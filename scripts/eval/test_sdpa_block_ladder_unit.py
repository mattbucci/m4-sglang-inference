#!/usr/bin/env python3
"""Unit test for patch 029: the depth-aware block count for MLX's two-pass
decode attention kernel.

Covers the ladder itself, the holder that keeps MLX_SDPA_BLOCKS in step with
the decode depth (writes only on a change, leaves an operator pin alone), the
runner asking for a block count on both decode paths (serial for mlx-vlm
attention, batched for mlx-lm), and the one property of MLX the patch depends
on: the variable is read on every kernel call, so a value set after import
takes effect and any value gives the same attention output up to rounding.
That last part runs a small kernel on the GPU; pass --no-gpu to skip it.

    python scripts/eval/test_sdpa_block_ladder_unit.py
"""
import os
import sys

os.environ.setdefault("SGLANG_USE_MLX", "1")
os.environ.pop("MLX_SDPA_BLOCKS", None)

from sglang.srt.hardware_backend.mlx import sdpa_blocks
from sglang.srt.hardware_backend.mlx.sdpa_blocks import ENV, SdpaBlockLadder, blocks_for

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# --- the ladder ---------------------------------------------------------------
check(all(blocks_for(n) == 0 for n in (1, 1024, 65_536, 131_072, 139_264)),
      "up to 139,264 keys MLX's own ladder is kept")
want = {139_265: 1536, 163_840: 1536, 196_608: 1536, 196_609: 2048, 251_659: 2048,
        262_144: 2048, 262_145: 2560, 327_680: 2560}
got = {n: blocks_for(n) for n in want}
check(got == want, f"512 blocks per 65,536 keys past the threshold ({got})")
deep = range(sdpa_blocks.STOCK_MAX_KEYS + 1, 1_048_577, 4099)
check(all(blocks_for(n) % 32 == 0 for n in deep), "every count is a multiple of 32 (MLX rounds up otherwise)")
check(all(n / blocks_for(n) <= 128 for n in deep), "a slice never holds more than 128 keys past the threshold")
check(all(blocks_for(a) <= blocks_for(b) for a, b in zip(deep, deep[1:])), "the count never falls as the context grows")

# --- the holder ---------------------------------------------------------------
writes = []


class Recorder(dict):
    """os.environ stand-in that records writes."""

    def __setitem__(self, k, v):
        writes.append(("set", k, v))
        super().__setitem__(k, v)

    def pop(self, k, *default):
        writes.append(("pop", k))
        return super().pop(k, *default)


real_environ = sdpa_blocks.os.environ
try:
    sdpa_blocks.os.environ = env = Recorder()
    off = SdpaBlockLadder(False)
    check(off.apply(250_000) == 0 and not writes and not off.enabled, "disabled: nothing is written at any depth")

    ladder = SdpaBlockLadder(True)
    check(ladder.enabled and ladder.apply(8_192) == 0 and not writes, "enabled, shallow: nothing is written")
    check(ladder.apply(150_000) == 1536 and env.get(ENV) == "1536", "crossing the threshold sets 1536")
    n = len(writes)
    for depth in range(150_001, 150_300):
        ladder.apply(depth)
    check(len(writes) == n, "steps inside one rung write nothing")
    check(ladder.apply(200_000) == 2048 and env.get(ENV) == "2048", "the next rung sets 2048")
    check(ladder.apply(4_096) == 0 and ENV not in env, "a shallow request after a deep one clears the variable")
    check(ladder.apply(262_145) == 2560 and env.get(ENV) == "2560", "past 262,144 keys the ladder keeps going")

    writes.clear()
    sdpa_blocks.os.environ = env = Recorder({ENV: "768"})
    writes.clear()
    pinned = SdpaBlockLadder(True)
    check(not pinned.enabled and pinned.apply(250_000) == 0 and env[ENV] == "768" and not writes,
          "a value set before launch is a pin: left alone")
finally:
    sdpa_blocks.os.environ = real_environ

# --- the runner asks on both decode paths ---------------------------------------
import inspect
import types

import mlx.core as mx

import sglang.srt.hardware_backend.mlx.model_runner as mr


class Spy:
    def __init__(self, enabled):
        self.enabled = enabled
        self.seen = []

    def apply(self, n_keys):
        self.seen.append(n_keys)
        return 0


def serial_step(spy):
    r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    r._cache_layout = types.SimpleNamespace(first_attention_layer_index=1)
    r._sdpa_block_ladder = spy
    r.model = lambda ids, cache: mx.zeros((1, 1, 4))
    caches = [[None, types.SimpleNamespace(offset=200_000)], [None, types.SimpleNamespace(offset=9)]]
    return r._decode_with_native_cache(caches, [mx.array([[1]]), mx.array([[2]])])


spy = Spy(True)
out = serial_step(spy)
check(spy.seen == [200_001] and out.shape == (2, 4),
      f"serial decode (mlx-vlm attention) asks for the deepest request's key count ({spy.seen})")
spy = Spy(False)
serial_step(spy)
check(not spy.seen, "serial decode with the ladder off does not read the caches")
check("_sdpa_block_ladder.apply(ctx.max_len)" in inspect.getsource(mr.MlxModelRunner._build_batched_decode_context),
      "batched decode asks with the step's context length")

# --- MLX reads the variable on every call --------------------------------------
if "--no-gpu" in sys.argv:
    print("SKIP kernel checks (--no-gpu)")
else:
    mx.random.seed(3)
    n_keys = 20_000
    q = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, n_keys, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, n_keys, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v)

    def attend(blocks):
        if blocks:
            os.environ[ENV] = str(blocks)
        else:
            os.environ.pop(ENV, None)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask=None)
        mx.eval(out)
        os.environ.pop(ENV, None)
        return out.astype(mx.float32)

    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=256 ** -0.5, mask=None)
    stock = attend(0)
    outs = {b: attend(b) for b in (512, 1536, 2048, 2560)}

    def rel(a, b):
        return float((mx.sqrt(((a - b) ** 2).sum()) / mx.sqrt((b ** 2).sum())).item())

    errs = {b: round(rel(o, ref), 5) for b, o in outs.items()}
    check(rel(stock, ref) < 0.01 and all(e < 0.01 for e in errs.values()),
          f"every block count matches a float32 reference (stock {rel(stock, ref):.5f}, {errs})")
    check(any(not bool(mx.array_equal(o, stock).item()) for o in outs.values()),
          "a value set after import changes the reduction order: MLX reads it per call")
    check(bool(mx.array_equal(attend(0), stock).item()), "clearing the variable restores the stock result bit for bit")
    check(bool(mx.array_equal(attend(2048), outs[2048]).item()), "one block count gives one result")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
