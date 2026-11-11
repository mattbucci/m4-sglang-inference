#!/usr/bin/env python3
"""CPU unit test for decode submission in parts: the boundary layers of N even
parts hand their output to ``mx.async_eval`` on a one-token step, and nothing
else about the forward changes.

Covers the boundary indices, the class swap and its idempotence, which steps
submit (one token per request) and which do not (a prefill chunk), tuple and
non-array layer outputs, argument pass-through, bit-equal outputs, and the
flag's default.

    python scripts/eval/test_decode_submit_unit.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

import sglang.srt.hardware_backend.mlx.decode_submit as ds
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.decode_submit import (
    patch_decode_submit_layers,
    submit_layer_indices,
)

fails = 0
DIM = 8


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# --- boundary indices --------------------------------------------------------
check(submit_layer_indices(num_layers=49, parts=1) == [], "one part: no layer submits early")
check(submit_layer_indices(num_layers=49, parts=2) == [23], "49 layers in two parts: layer 23")
check(submit_layer_indices(num_layers=49, parts=5) == [8, 18, 28, 38], "49 layers in five parts")
check(submit_layer_indices(num_layers=40, parts=4) == [9, 19, 29], "40 layers in four parts")
check(submit_layer_indices(num_layers=4, parts=9) == [0, 1, 2], "more parts than layers: one layer per part")
check(submit_layer_indices(num_layers=1, parts=4) == [], "a single layer: nothing to split")
ok = True
for n in range(1, 81):
    for parts in range(1, 13):
        idx = submit_layer_indices(num_layers=n, parts=parts)
        ok &= idx == sorted(set(idx)) and all(0 <= i < n - 1 for i in idx)
        ok &= len(idx) == min(parts, n) - 1
check(ok, "indices are distinct, ascending, never the last layer, one per boundary (1-80 layers, 1-12 parts)")


# --- the class swap ----------------------------------------------------------
class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM)
        self.seen = []

    def __call__(self, x, mask=None, cache=None):
        self.seen.append((mask, cache))
        return x + mx.tanh(self.proj(x))


class TupleLayer(Layer):
    def __call__(self, x, mask=None, cache=None):
        return super().__call__(x, mask=mask, cache=cache), "state"


class OddLayer(nn.Module):
    def __call__(self, x):
        return None


def forward(layers, x):
    for layer in layers:
        out = layer(x, mask="m", cache="c")
        x = out[0] if isinstance(out, tuple) else out
    return x


mx.random.seed(7)
layers = [Layer() for _ in range(8)]
mx.eval([layer.parameters() for layer in layers])
step = mx.random.normal((2, 1, DIM))
chunk = mx.random.normal((1, 5, DIM))
want_step, want_chunk = forward(layers, step), forward(layers, chunk)
mx.eval(want_step, want_chunk)

n = patch_decode_submit_layers(layers=layers, parts=4)
marked = [i for i, layer in enumerate(layers) if type(layer) is not Layer]
check(n == 3 and marked == [1, 3, 5], f"eight layers in four parts: layers 1, 3, 5 submit ({n}, {marked})")
check(all(type(layers[i]).__name__ == "SubmittingLayer" and isinstance(layers[i], Layer) for i in marked),
      "a marked layer is a subclass of its own class")
check(len({type(layers[i]) for i in marked}) == 1, "layers of one class share one submitting class")
before = [type(layer) for layer in layers]
check(patch_decode_submit_layers(layers=layers, parts=2) == 3 and [type(layer) for layer in layers] == before,
      "a second install changes nothing and reports the layers already marked")

# --- which forwards submit ---------------------------------------------------
calls = []
real_mx = ds.mx
ds.mx = SimpleNamespace(array=mx.array, async_eval=lambda *arrays: calls.append(arrays))
for layer in layers:
    layer.seen.clear()
got = forward(layers, step)
check(len(calls) == 3 and all(len(c) == 1 and c[0].shape == (2, 1, DIM) for c in calls),
      f"a one-token step submits once per marked layer ({len(calls)})")
check(all(layer.seen == [("m", "c")] for layer in layers), "arguments reach the layer's own call unchanged")
calls.clear()
forward(layers, chunk)
check(calls == [], "a prefill chunk (five tokens) submits nothing early")

tup = [TupleLayer() for _ in range(4)]
patch_decode_submit_layers(layers=tup, parts=2)
calls.clear()
out = tup[1](step)
check(isinstance(out, tuple) and out[1] == "state" and len(calls) == 1 and calls[0][0] is out[0],
      "a tuple output: the first entry is submitted, the tuple is returned whole")
odd = [OddLayer() for _ in range(2)]
patch_decode_submit_layers(layers=odd, parts=2)
calls.clear()
check(odd[0](step) is None and calls == [], "an output that is not an array: nothing submitted, value returned")
ds.mx = real_mx

# --- same results ------------------------------------------------------------
got_step, got_chunk = forward(layers, step), forward(layers, chunk)
mx.eval(got_step, got_chunk)
check(mx.array_equal(got_step, want_step).item(), "one-token step: output bit-equal to the unmarked layers")
check(mx.array_equal(got_chunk, want_chunk).item(), "prefill chunk: output bit-equal to the unmarked layers")
chained = forward(layers, forward(layers, step))
mx.async_eval(chained)
check(mx.array_equal(chained, forward(layers, want_step)).item(),
      "a step built on a step still in flight evaluates to the same values")

# --- the flag ----------------------------------------------------------------
check(envs.SGLANG_MLX_DECODE_SUBMIT_PARTS.get() == 1, "flag default: one part (off)")
with envs.SGLANG_MLX_DECODE_SUBMIT_PARTS.override(4):
    check(envs.SGLANG_MLX_DECODE_SUBMIT_PARTS.get() == 4, "flag override reads back")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
