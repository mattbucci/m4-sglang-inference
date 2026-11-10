#!/usr/bin/env python3
"""Unit test for the one-launch GatedDeltaNet decode front (patch 031).

Runs small arrays on the Metal device; no server and no model download.

  1. fused_front: for bf16, fp16 and float32 inputs the shifted conv state is
     the stock state bit for bit, and q, k and v match a float64 reference
     over the same (already rounded) inputs to the dtype's precision, for one
     row and for several.
  2. Block swap: a small mlx-vlm Qwen3_5GatedDeltaNet gives the stock output
     and the stock cache contents over chained decode steps after the swap
     (float32 weights, so the stock chain has no intermediate rounding to
     differ from). With bf16 weights the two stay within rounding of each
     other and the conv state stays bit-equal.
  3. Scope: prefill, masked, length-tracked, target-verify, cache-less and
     first-step (empty cache) calls keep the stock chain bit for bit; the
     swap is idempotent and leaves other modules alone.
  4. Install site: the flag is read in the helper both load paths of the
     runner call (mlx-vlm checkpoints return before the text-path tail).

  python scripts/eval/test_gdn_front_unit.py
"""
import os
import sys
import types

import numpy as np

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as lang  # noqa: E402

from sglang.srt.hardware_backend.mlx import gated_delta_front as gf  # noqa: E402

FAILS = []
TOL = {mx.bfloat16: 2**-7, mx.float16: 2**-9, mx.float32: 1e-5}


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def rel(got, want):
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


def test_kernel():
    rng = np.random.default_rng(0)
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        for rows, taps, kh, d, vdim in ((1, 4, 4, 32, 256), (3, 4, 2, 64, 128), (5, 3, 3, 16, 96), (2, 2, 1, 128, 128)):
            kd = kh * d
            c = 2 * kd + vdim
            cstate = mx.array(rng.normal(0, 1, (rows, taps - 1, c)).astype(np.float32)).astype(dtype)
            x = mx.array(rng.normal(0, 1, (rows, 1, c)).astype(np.float32)).astype(dtype)
            w = mx.array(rng.normal(0, 0.5, (taps, c)).astype(np.float32))
            state, q, k, v = gf.fused_front(cstate, x, w, kd, d)
            mx.eval(state, q, k, v)
            conv_in = np.concatenate([f64(cstate), f64(x)], axis=1)
            acc = (conv_in * f64(w)[None]).sum(axis=1)
            o = acc / (1 + np.exp(-acc))
            qr = o[:, :kd].reshape(rows, kh, d)
            kr = o[:, kd : 2 * kd].reshape(rows, kh, d)
            qr = qr / np.sqrt((qr * qr).mean(-1, keepdims=True) + 1e-6) / d
            kr = kr / np.sqrt((kr * kr).mean(-1, keepdims=True) + 1e-6) * d**-0.5
            want_state = mx.concatenate([cstate, x], axis=1)[:, 1:, :]
            name = f"front {str(dtype).split('.')[-1]} rows={rows} taps={taps} heads={kh}x{d} v={vdim}"
            errs = (
                rel(f64(q).reshape(rows, kh, d), qr),
                rel(f64(k).reshape(rows, kh, d), kr),
                rel(f64(v).reshape(rows, vdim), o[:, 2 * kd :]),
            )
            shapes = q.shape == (rows, 1, kd) and k.shape == (rows, 1, kd) and v.shape == (rows, 1, vdim)
            check(
                name,
                shapes
                and state.dtype == dtype
                and bool(mx.array_equal(state, want_state).item())
                and max(errs) < TOL[dtype],
                "q/k/v err " + "/".join(f"{e:.1e}" for e in errs),
            )


def make_block(dtype, seed=3):
    cfg = types.SimpleNamespace(
        hidden_size=96,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(seed)
    b = lang.Qwen3_5GatedDeltaNet(cfg)
    b.conv1d.weight = mx.random.normal(b.conv1d.weight.shape) * 0.5
    b.A_log = mx.log(mx.random.uniform(0.1, 16, (8,)))
    b.dt_bias = mx.random.normal((8,))
    b.norm.weight = 1 + 0.2 * mx.random.normal((32,))
    b.set_dtype(dtype)
    b.eval()
    mx.eval(b.parameters())
    return b


def primed_cache(b, dtype, rows, seed):
    mx.random.seed(seed)
    cache = ArraysCache(size=2)
    cache[0] = mx.random.normal((rows, b.conv_kernel_size - 1, b.conv_dim)).astype(dtype)
    cache[1] = (mx.random.normal((rows, b.num_v_heads, b.head_v_dim, b.head_k_dim)) * 0.1).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def run_steps(call, b, cache, xs):
    outs = []
    for x in xs:
        y = call(b, x, None, cache)
        mx.eval(y, cache[0], cache[1])
        outs.append(y)
    return outs


def test_block_swap():
    stock_cls = lang.Qwen3_5GatedDeltaNet
    for dtype, tol in ((mx.float32, 2e-5), (mx.bfloat16, 3e-2)):
        tag = str(dtype).split(".")[-1]
        for rows in (1, 3):
            b = make_block(dtype)
            mx.random.seed(11)
            xs = [mx.random.normal((rows, 1, 96)).astype(dtype) for _ in range(6)]
            ref_cache = primed_cache(b, dtype, rows, 5)
            want = run_steps(stock_cls.__call__, b, ref_cache, xs)
            holder = nn.Module()
            holder.layers = [b, nn.Linear(4, 4)]
            n = gf.patch_gated_delta_front(holder)
            again = gf.patch_gated_delta_front(holder)
            check(
                f"swap {tag} rows={rows}: one layer swapped, idempotent, others untouched",
                n == 1 and again == 1 and type(b).__name__ == "FusedQwen3_5GatedDeltaNet"
                and isinstance(b, stock_cls) and type(holder.layers[1]) is nn.Linear,
            )
            cache = primed_cache(b, dtype, rows, 5)
            got = run_steps(type(b).__call__, b, cache, xs)
            err = max(rel(f64(g), f64(w)) for g, w in zip(got, want))
            state_err = rel(f64(cache[1]), f64(ref_cache[1]))
            conv_equal = bool(mx.array_equal(cache[0], ref_cache[0]).item())
            check(
                f"swap {tag} rows={rows}: six chained decode steps match stock",
                err < tol and state_err < tol and conv_equal and cache[0].dtype == ref_cache[0].dtype
                and cache[1].dtype == ref_cache[1].dtype and cache[1].shape == ref_cache[1].shape,
                f"out err {err:.1e}, state err {state_err:.1e}, conv state equal {conv_equal}",
            )


def test_scope():
    stock_cls = lang.Qwen3_5GatedDeltaNet
    dtype = mx.bfloat16
    b = make_block(dtype)
    holder = nn.Module()
    holder.layers = [b]
    gf.patch_gated_delta_front(holder)
    fused = type(b).__call__
    mx.random.seed(21)
    x1 = mx.random.normal((2, 1, 96)).astype(dtype)
    x5 = mx.random.normal((2, 5, 96)).astype(dtype)

    def same(name, x, make, **kw):
        ca, cb = make(), make()
        ya = stock_cls.__call__(b, x, kw.get("mask"), ca, target_verify=kw.get("target_verify", False))
        yb = fused(b, x, kw.get("mask"), cb, target_verify=kw.get("target_verify", False))
        mx.eval(ya, yb)
        ok = bool(mx.array_equal(ya, yb).item())
        if ca is not None:
            ok = ok and all(bool(mx.array_equal(ca[i], cb[i]).item()) for i in range(2))
        check(f"scope: {name} keeps the stock chain", ok)

    def lengths_cache():
        c = primed_cache(b, dtype, 2, 5)
        c.lengths = mx.array([1, 1])
        return c

    same("prefill of five tokens", x5, lambda: primed_cache(b, dtype, 2, 5))
    same("masked decode step", x1, lambda: primed_cache(b, dtype, 2, 5), mask=mx.array([[True], [True]]))
    same("length-tracked cache", x1, lengths_cache)
    same("target-verify call", x1, lambda: primed_cache(b, dtype, 2, 5), target_verify=True)
    same("first step on an empty cache", x1, lambda: ArraysCache(size=2))
    same("call without a cache", x1, lambda: None)
    same("cache of another batch size", x1, lambda: primed_cache(b, dtype, 3, 5))

    # a decode step with a live cache does take the kernel: not bit-equal on
    # bf16, but the offsets the cache tracks advance as stock's do
    ca, cb = primed_cache(b, dtype, 2, 5), primed_cache(b, dtype, 2, 5)
    for c in (ca, cb):
        c.left_padding = mx.array([3, 0])
    stock_cls.__call__(b, x1, None, ca)
    fused(b, x1, None, cb)
    check(
        "scope: live decode step advances the cache bookkeeping as stock does",
        np.array_equal(np.array(ca.left_padding), np.array(cb.left_padding))
        and np.array(cb.left_padding).tolist() == [2, -1],
    )


def test_install_site():
    import ast

    runner = os.path.join(os.path.dirname(gf.__file__), "model_runner.py")
    tree = ast.parse(open(runner).read())
    helper = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_install_decode_fusions"
    ]
    check(
        "install site: the helper both load paths call reads the flag",
        len(helper) == 1 and "SGLANG_MLX_FUSE_GDN_FRONT" in ast.dump(helper[0]),
    )


if __name__ == "__main__":
    print("module:", gf.__file__)
    test_kernel()
    test_block_swap()
    test_scope()
    test_install_site()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("ALL PASS")
