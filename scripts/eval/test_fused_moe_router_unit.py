#!/usr/bin/env python3
"""Unit test for the one-launch MoE routing and combine kernels (patch 030).

Runs small arrays on the Metal device; no server and no model download.

  1. fused_route: for bf16, fp16 and float32 logits the kernel picks the same
     expert set as a float64 reference over the same (already rounded) logits,
     scores match the reference to the dtype's precision, sum to one, ascend
     within a row, and ties go to the lower expert index. Without top-k
     renormalisation the scores are the full softmax probabilities.
  2. fused_combine: the weighted sum, with and without the shared-expert term,
     matches a float64 reference to the dtype's precision.
  3. Block swap: a small mlx-lm Qwen3 MoE block and a small mlx-vlm Qwen3.5 MoE
     block give the stock output after the swap (float32 weights, so the stock
     chain has no intermediate rounding to differ from), for one decode row
     and for a batch of prefill rows; the swap is idempotent, leaves other
     modules alone, and the target-verify call keeps the stock chain.

  python scripts/eval/test_fused_moe_router_unit.py
"""
import os
import sys
import types

import numpy as np

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from sglang.srt.hardware_backend.mlx.moe import fused_router as fr  # noqa: E402

FAILS = []
TOL = {mx.bfloat16: 2**-7, mx.float16: 2**-9, mx.float32: 1e-5}


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def ref_route(logits64, k, norm):
    order = np.lexsort((np.arange(logits64.shape[-1])[None, :].repeat(logits64.shape[0], 0), -logits64), axis=-1)
    top = order[:, :k]
    ex = np.exp(logits64 - logits64.max(axis=-1, keepdims=True))
    picked = np.take_along_axis(ex, top, axis=-1)
    denom = picked.sum(axis=-1, keepdims=True) if norm else ex.sum(axis=-1, keepdims=True)
    return top, picked / denom


def test_route():
    rng = np.random.default_rng(0)
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        for rows, e, k, norm in ((1, 256, 8, True), (7, 256, 8, True), (5, 128, 8, True), (3, 16, 4, False), (4, 60, 6, False)):
            logits = mx.array(rng.normal(0, 3, (rows, e)).astype(np.float32)).astype(dtype)
            inds, scores = fr.fused_route(logits, k, norm)
            mx.eval(inds, scores)
            top, want = ref_route(f64(logits), k, norm)
            got_i, got_s = np.array(inds), f64(scores)
            name = f"route {str(dtype).split('.')[-1]} rows={rows} e={e} k={k} norm={norm}"
            same_set = all(set(got_i[r]) == set(top[r]) for r in range(rows))
            # compare scores expert by expert
            err = 0.0
            for r in range(rows):
                w = dict(zip(top[r].tolist(), want[r].tolist()))
                for i, s in zip(got_i[r].tolist(), got_s[r].tolist()):
                    err = max(err, abs(s - w.get(i, 0.0)) / max(w.get(i, 1e-9), 1e-9))
            ascending = bool(np.all(np.diff(got_s, axis=-1) >= 0))
            sums = bool(np.all(np.abs(got_s.sum(-1) - 1) < 4 * TOL[dtype])) if norm else True
            check(name, same_set and err < TOL[dtype] and ascending and sums,
                  f"set={same_set} rel_err={err:.2e} ascending={ascending} sum1={sums}")
    flat = mx.zeros((2, 32), dtype=mx.bfloat16)
    inds, scores = fr.fused_route(flat, 4)
    mx.eval(inds, scores)
    check("route ties go to the lower expert index",
          sorted(np.array(inds)[0].tolist()) == [0, 1, 2, 3] and np.allclose(f64(scores), 0.25),
          f"inds={np.array(inds)[0].tolist()}")
    shaped = fr.fused_route(mx.zeros((2, 5, 32), dtype=mx.bfloat16), 4)
    check("route keeps leading dims", shaped[0].shape == (2, 5, 4) and shaped[1].shape == (2, 5, 4)
          and shaped[0].dtype == mx.uint32 and shaped[1].dtype == mx.bfloat16)


def test_combine():
    rng = np.random.default_rng(1)
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        for lead, k, h in (((1, 1), 8, 2048), ((2, 5), 8, 256), ((3,), 4, 96)):
            y = mx.array(rng.normal(0, 1, lead + (k, h)).astype(np.float32)).astype(dtype)
            s = mx.array(rng.dirichlet(np.ones(k), lead).astype(np.float32)).astype(dtype)
            shared = mx.array(rng.normal(0, 1, lead + (h,)).astype(np.float32)).astype(dtype)
            gate = mx.array(rng.normal(0, 2, lead + (1,)).astype(np.float32)).astype(dtype)
            base = (f64(y) * f64(s)[..., None]).sum(-2)
            full = base + 1 / (1 + np.exp(-f64(gate))) * f64(shared)
            a = fr.fused_combine(y, s)
            b = fr.fused_combine(y, s, shared, gate)
            mx.eval(a, b)
            tol = TOL[dtype] * (np.abs(full).max() + 1)
            ea, eb = np.abs(f64(a) - base).max(), np.abs(f64(b) - full).max()
            check(f"combine {str(dtype).split('.')[-1]} lead={lead} k={k} h={h}",
                  a.shape == lead + (h,) and b.shape == lead + (h,) and a.dtype == dtype
                  and ea < tol and eb < tol, f"err plain {ea:.2e} shared {eb:.2e} tol {tol:.2e}")


def test_swap():
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    mx.random.seed(3)
    args = types.SimpleNamespace(hidden_size=64, moe_intermediate_size=32,
                                 shared_expert_intermediate_size=48, num_experts=16,
                                 num_experts_per_tok=4, norm_topk_prob=True)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = Qwen3MoeSparseMoeBlock(args)
            self.shared = Qwen3_5MoeSparseMoeBlock(args)
            self.other = nn.Linear(64, 64)

    holder = Holder()
    holder.eval()
    mx.eval(holder.parameters())
    xs = {"decode row": mx.random.normal((1, 1, 64)), "prefill rows": mx.random.normal((2, 37, 64))}
    stock = {}
    for name, x in xs.items():
        stock[name] = (holder.plain(x), holder.shared(x))
        mx.eval(*stock[name])
    n = fr.patch_moe_blocks_with_fused_router(holder)
    check("swap covers both block shapes and nothing else", n == 2
          and type(holder.plain).__name__ == "FusedQwen3MoeSparseMoeBlock"
          and type(holder.shared).__name__ == "FusedQwen3_5MoeSparseMoeBlock"
          and type(holder.other) is nn.Linear and isinstance(holder.plain, Qwen3MoeSparseMoeBlock),
          f"swapped {n}")
    check("swap is idempotent", fr.patch_moe_blocks_with_fused_router(holder) == 2
          and type(holder.plain).__mro__[1] is Qwen3MoeSparseMoeBlock)
    for name, x in xs.items():
        a, b = holder.plain(x), holder.shared(x)
        mx.eval(a, b)
        ea = float(mx.abs(a - stock[name][0]).max())
        eb = float(mx.abs(b - stock[name][1]).max())
        scale = float(mx.abs(stock[name][1]).max())
        check(f"swapped blocks match stock, {name}", a.shape == stock[name][0].shape
              and b.shape == stock[name][1].shape and ea < 1e-5 * (scale + 1) and eb < 1e-5 * (scale + 1),
              f"max err plain {ea:.2e} shared {eb:.2e} at scale {scale:.2f}")
    x = xs["prefill rows"]
    tv = holder.shared(x, True)
    want = Qwen3_5MoeSparseMoeBlock.__call__(holder.shared, x, True)
    mx.eval(tv, want)
    check("target-verify call keeps the stock chain", bool(mx.array_equal(tv, want)))
    args2 = types.SimpleNamespace(**{**vars(args), "norm_topk_prob": False})
    blk = Qwen3MoeSparseMoeBlock(args2)
    blk.eval()
    mx.eval(blk.parameters())
    x = mx.random.normal((1, 3, 64))
    want = blk(x)
    mx.eval(want)
    h2 = nn.Module()
    h2.blk = blk
    fr.patch_moe_blocks_with_fused_router(h2)
    got = blk(x)
    mx.eval(got)
    check("unnormalised top-k scores match stock", float(mx.abs(got - want).max()) < 1e-5,
          f"max err {float(mx.abs(got - want).max()):.2e}")


if __name__ == "__main__":
    test_route()
    test_combine()
    test_swap()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)
