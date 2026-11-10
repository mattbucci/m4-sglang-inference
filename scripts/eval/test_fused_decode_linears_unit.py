#!/usr/bin/env python3
"""Unit test for grouped quantized linears on a decode step (patch 032).

Runs small arrays on the Metal device; no server and no model download.

  1. fused_decode_linears: sets of two, three and four quantized linears, of
     one bit width and mixed, give the separate calls' outputs bit for bit, for
     one row and for several; a set with nothing to fold (single member per
     bit width, a prefill-shaped input, a group over the byte budget) returns
     None; non-quantized members and members with a bias run on their own.
  2. Hook: once installed in mlx-vlm's dispatch namespace, a two-member and a
     three-member set go through the grouped fold, the four-member set of one
     bit width stays with mlx-vlm's own fold, target-verify dispatch is
     untouched, and a second install does not wrap twice. A real mlx-vlm
     Qwen3_5MLP gives its stock output bit for bit.
  3. MoE block: the fused shared-expert block (patch 030) folds its two 8-bit
     gates when the fold is enabled and returns the same output bit for bit.
  4. Install site: the flag is read in the helper both load paths of the
     runner call (mlx-vlm checkpoints return before the text-path tail).

  python scripts/eval/test_fused_decode_linears_unit.py
"""
import os
import sys
import types

import numpy as np

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as lang  # noqa: E402

from sglang.srt.hardware_backend.mlx import fused_linears as fl  # noqa: E402
from sglang.srt.hardware_backend.mlx.moe import fused_router as fr  # noqa: E402

FAILS = []
DIM = 128


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.all(a == b).item())


def qlin(out_dim, bits, dtype=mx.bfloat16, bias=False):
    lin = nn.Linear(DIM, out_dim, bias=bias)
    lin.set_dtype(dtype)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    mx.eval(q.parameters())
    return q


def folded(linear):
    return getattr(linear, fl._CACHE_ATTR, None) is not None


def test_fold():
    mx.random.seed(0)
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        for rows in (1, 3):
            x = mx.random.normal((rows, 1, DIM)).astype(dtype)
            for bits in ([4, 4], [8, 8, 8], [4, 4, 4, 4], [8, 4, 4, 8], [4, 8, 4]):
                lins = [qlin(16 * (i + 1) + 1, b, dtype) for i, b in enumerate(bits)]
                got = fl.fused_decode_linears(lins, x)
                ok = got is not None and all(
                    same(g, lin(x)) for g, lin in zip(got, lins)
                )
                n_groups = sum(folded(lin) for lin in lins)
                want_groups = len({b for b in bits if bits.count(b) > 1})
                check(f"{dtype} rows={rows} bits={bits} bit-equal to separate calls",
                      ok and n_groups == want_groups, f"groups {n_groups}")
    x = mx.random.normal((1, 1, DIM)).astype(mx.bfloat16)
    a, b = qlin(8, 4), qlin(8, 8)
    check("one member per bit width folds nothing",
          fl.fused_decode_linears([a, b], x) is None and not folded(a))
    a, b = qlin(8, 4), qlin(8, 4)
    check("a single linear folds nothing", fl.fused_decode_linears([a], x) is None)
    check("a prefill-shaped input folds nothing",
          fl.fused_decode_linears([a, b], mx.random.normal((1, 5, DIM)).astype(mx.bfloat16)) is None
          and fl.fused_decode_linears([a, b], x[0]) is None and not folded(a))
    check("an input of another dtype folds nothing",
          fl.fused_decode_linears([a, b], x.astype(mx.float32)) is None)
    budget = fl.MAX_GROUP_BYTES
    fl.MAX_GROUP_BYTES = a.weight.nbytes + b.weight.nbytes - 1
    over = fl.fused_decode_linears([a, b], x)
    fl.MAX_GROUP_BYTES = a.weight.nbytes + b.weight.nbytes
    at = fl.fused_decode_linears([a, b], x)
    fl.MAX_GROUP_BYTES = budget
    check("a group over the byte budget stays separate", over is None and at is not None)
    dense = nn.Linear(DIM, 8, bias=False)
    dense.set_dtype(mx.bfloat16)
    biased = qlin(8, 4, bias=True)
    c, d = qlin(24, 4), qlin(40, 4)
    got = fl.fused_decode_linears([c, dense, biased, d], x)
    check("dense and biased members run on their own, the rest fold",
          got is not None and folded(c) and not folded(biased)
          and all(same(g, lin(x)) for g, lin in zip(got, [c, dense, biased, d])))
    off = fl.decode_linears([a, b], x)
    check("decode_linears without the flag is the separate calls",
          not fl._enabled and same(off[0], a(x)) and same(off[1], b(x)))


def test_hook():
    mx.random.seed(1)
    stock_fold = lang._decode_quantized_linears_fused
    x = mx.random.normal((1, 1, DIM)).astype(mx.bfloat16)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = lang.Qwen3_5MLP(DIM, 192)

    holder = Holder()
    holder.set_dtype(mx.bfloat16)
    nn.quantize(holder, group_size=64, bits=4)
    holder.eval()
    mx.eval(holder.parameters())
    want_mlp = holder.mlp(x)
    mx.eval(want_mlp)
    try:
        n = fl.patch_fused_decode_linears(holder)
        hooked = lang._decode_quantized_linears_fused
        n2 = fl.patch_fused_decode_linears(holder)
        check("hook installs once in the dispatch namespace",
              n == 1 and n2 == 1 and hooked is not stock_fold
              and lang._decode_quantized_linears_fused is hooked
              and hooked._sglang_stock is stock_fold and fl._enabled)
        got_mlp = holder.mlp(x)
        check("Qwen3_5MLP folds its two input projections, output bit-equal",
              folded(holder.mlp.gate_proj) and same(got_mlp, want_mlp))
        tv = holder.mlp(mx.random.normal((2, 6, DIM)).astype(mx.bfloat16), True)
        mx.eval(tv)
        check("prefill and target-verify calls leave the fold alone", tv.shape == (2, 6, DIM))
        three = [qlin(24, 4), qlin(8, 4), qlin(8, 4)]
        got = lang._target_verify_linears(three, x, False)
        check("a three-member set folds through the hook",
              folded(three[0]) and all(same(g, lin(x)) for g, lin in zip(got, three)))
        four = [qlin(24, 4), qlin(8, 4), qlin(8, 4), qlin(16, 4)]
        got = lang._target_verify_linears(four, x, False)
        check("four members of one bit width stay with the stock fold",
              not folded(four[0]) and getattr(four[0], "_qwen3_5_fused_decode_linears", None) is not None
              and all(same(g, lin(x)) for g, lin in zip(got, four)))
        mixed = [qlin(24, 8), qlin(8, 4), qlin(8, 4), qlin(1, 8)]
        got = lang._target_verify_linears(mixed, x, False)
        check("four members of two bit widths fold as two pairs",
              folded(mixed[0]) and folded(mixed[1])
              and all(same(g, lin(x)) for g, lin in zip(got, mixed)))
    finally:
        lang._decode_quantized_linears_fused = stock_fold
        fl._enabled = False


def test_moe_block():
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    mx.random.seed(2)
    stock_fold = lang._decode_quantized_linears_fused
    args = types.SimpleNamespace(hidden_size=DIM, moe_intermediate_size=64,
                                 shared_expert_intermediate_size=64, num_experts=16,
                                 num_experts_per_tok=4)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Qwen3_5MoeSparseMoeBlock(args)

    holder = Holder()
    holder.set_dtype(mx.bfloat16)

    def recipe(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith("gate"):
            return {"bits": 8, "group_size": 64}
        return True

    nn.quantize(holder, group_size=64, bits=4, class_predicate=recipe)
    holder.eval()
    mx.eval(holder.parameters())
    block = holder.mlp
    check("test block mirrors the community recipe (8-bit gates, 4-bit rest)",
          block.gate.bits == 8 and block.shared_expert_gate.bits == 8
          and block.shared_expert.gate_proj.bits == 4)
    fr.patch_moe_blocks_with_fused_router(holder)
    x = mx.random.normal((2, 1, DIM)).astype(mx.bfloat16)
    want = block(x)
    mx.eval(want)
    check("fused block without the flag keeps separate gates",
          not folded(block.gate) and not folded(block.shared_expert.gate_proj))
    try:
        fl.patch_fused_decode_linears(holder)
        got = block(x)
        check("fused block folds both gates and the shared expert's projections, output bit-equal",
              folded(block.gate) and folded(block.shared_expert.gate_proj) and same(got, want))
        pre = mx.random.normal((1, 9, DIM)).astype(mx.bfloat16)
        check("prefill rows give the stock fused-block output",
              same(block(pre), fr.fused_combine(
                  block.switch_mlp(pre, fr.fused_route(block.gate(pre), block.top_k)[0]),
                  fr.fused_route(block.gate(pre), block.top_k)[1],
                  block.shared_expert(pre, False), block.shared_expert_gate(pre))))
    finally:
        lang._decode_quantized_linears_fused = stock_fold
        fl._enabled = False


def test_install_site():
    import ast

    runner = os.path.join(os.path.dirname(fl.__file__), "model_runner.py")
    tree = ast.parse(open(runner).read())
    helper = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_install_decode_fusions"
    ]
    check(
        "install site: the helper both load paths call reads the flag",
        len(helper) == 1 and "SGLANG_MLX_FUSE_DECODE_LINEARS" in ast.dump(helper[0]),
    )


if __name__ == "__main__":
    for t in (test_fold, test_hook, test_moe_block, test_install_site):
        print(t.__name__)
        t()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
