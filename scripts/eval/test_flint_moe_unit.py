#!/usr/bin/env python3
"""Unit test for the fused Flint MoE block (SGLANG_MLX_FUSE_FLINT_MOE).

Runs small arrays and tiny random Flint models on the Metal device; no server
and no model download.

  1. Kernels: the combine kernel equals the stock op chain bit for bit in
     bf16, fp16 and float32, for one to four rows, three widths, two expert
     counts and three magnitudes; the router kernel equals the stock matmul
     on a 256 x 2048 router weight.
  2. Model: after the patch, decode steps under a batched decode context give
     the stock logits and the stock cache contents bit for bit, for one to
     five requests of unequal length, in bf16 and float32; so do prefill and
     mlx-lm's own decode loop. One model has the checkpoint's 256 x 2048
     router, so its steps run the router kernel.
  3. Scope: a block in training mode keeps the stock call; the install is
     idempotent, leaves the global random state alone, returns 0 on another
     model, and the flag is read in the helper both load paths of the runner
     call.

  python scripts/eval/test_flint_moe_unit.py
"""
import ast
import os
import sys

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models import flint as fl  # noqa: E402

from sglang.srt.hardware_backend.mlx import model_runner  # noqa: E402
from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (  # noqa: E402
    get_layer_window_sizes,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (  # noqa: E402
    AttentionOffsetCache,
    ContiguousAttentionKVCache,
    WindowedAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (  # noqa: E402
    BatchedDecodeContext,
    clear_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (  # noqa: E402
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.moe import fused_flint_block as fb  # noqa: E402

FAILS = []
DTYPES = (mx.bfloat16, mx.float16, mx.float32)
VOCAB = 512
WINDOW = 16
PROMPT_LENS = (9, 40, 23, 31, 12)
STEPS = 24


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def name_of(dtype):
    return str(dtype).split(".")[-1]


def test_kernels():
    for dtype in DTYPES:
        bad = []
        for dim in (128, 512, 2048):
            for rows in range(1, fb.MAX_ROWS + 1):
                for top_k in (3, 6):
                    for scale in (0.02, 1.0, 40.0):
                        b = (scale * mx.random.normal((rows, 1, dim))).astype(dtype)
                        y = (scale * mx.random.normal((rows, 1, top_k, dim))).astype(dtype)
                        w = mx.softmax(mx.random.normal((rows, 1, top_k)), axis=-1)
                        want = (y * w[..., None]).sum(axis=-2).astype(dtype) + b
                        if not same(fb.combine(y=y, weights=w, shared=b), want):
                            bad.append((dim, rows, top_k, scale))
        check(f"kernels: combine equals the stock ops, {name_of(dtype)}",
              not bad, f"{len(bad)} differ, first {bad[:3]}" if bad else "")
    for wdtype in (mx.bfloat16, mx.float32):
        gate = nn.Linear(2048, 256, bias=False)
        gate.weight = mx.random.normal((256, 2048)).astype(wdtype)
        bad = 0
        for xdtype in (mx.bfloat16, mx.float32):
            for _ in range(8):
                x = (2 * mx.random.normal((1, 1, 2048))).astype(xdtype)
                got = fb.router_logits(weight=gate.weight, x=x)
                bad += not same(got, gate(x.astype(mx.float32)))
        check(f"kernels: router logits equal the stock matmul, 256 x 2048 {name_of(wdtype)}",
              bad == 0, f"{bad} of 16 differ" if bad else "")


def build(*, hidden, heads, kv, experts, topk, dtype, seed, layers=6):
    mx.random.seed(seed)
    args = fl.ModelArgs(
        model_type="flint", hidden_size=hidden, num_hidden_layers=layers,
        intermediate_size=hidden, num_attention_heads=heads, num_key_value_heads=kv,
        head_dim=64, num_experts=experts, num_experts_per_tok=topk,
        moe_intermediate_size=64, shared_expert_intermediate_size=128,
        mlp_only_layers=[0],
        layer_types=["sliding_attention", "full_attention"] * (layers // 2),
        sliding_window=WINDOW, rms_norm_eps=1e-6, vocab_size=VOCAB,
    )
    model = fl.Model(args)
    for _, m in model.named_modules():
        if isinstance(m, nn.RMSNorm):
            m.weight = 1 + 0.3 * mx.random.normal(m.weight.shape)
        if isinstance(m, fl.Attention):
            m.softmax_offset = 0.5 * mx.random.normal(m.softmax_offset.shape)
        if isinstance(m, nn.Linear):
            m.weight = m.weight * 3
    nn.quantize(
        model, bits=4, group_size=64,
        class_predicate=lambda path, m: (
            isinstance(m, (nn.Linear, nn.Embedding)) or hasattr(m, "to_quantized")
        ) and model.quant_predicate(path, m),
    )
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def new_caches(model):
    windows = get_layer_window_sizes(model)
    return [
        ContiguousAttentionKVCache(max_seq_len=256) if windows.get(i) is None
        else WindowedAttentionKVCache(windows[i])
        for i in range(len(model.model.layers))
    ]


def run_requests(model, prompts, steps):
    """Prefill each request on its own, then decode them as one batched step
    at a time the way the runner does. Returns prefill logits, the logits of
    every step and the cache arrays at the end."""
    caches, prefill = [], []
    for prompt in prompts:
        cache = new_caches(model)
        out = model(prompt[None], cache=cache)
        mx.eval(out, [c.state for c in cache])
        caches.append(cache)
        prefill.append(out)
    layers = len(model.model.layers)
    outs = []
    for tokens in steps:
        seq_lens = [cache[0].offset for cache in caches]
        ctx = BatchedDecodeContext(
            batch_size=len(caches),
            seq_lens=seq_lens,
            attention_layer_caches=[[c[i] for c in caches] for i in range(layers)],
        )
        set_context(ctx)
        try:
            shim = [AttentionOffsetCache(offset=max(seq_lens)) for _ in range(layers)]
            out = model(tokens[:, None], cache=shim)[:, -1, :]
            mx.eval(out)
        finally:
            clear_context()
        outs.append(out)
    state = [a for cache in caches for c in cache for a in c.state]
    mx.eval(state)
    return prefill, outs, state


def native_steps(model, prompt, tokens):
    """mlx-lm's own caches and forward, no decode context."""
    cache = model.make_cache()
    outs = [model(prompt[None], cache=cache)[:, -1, :]]
    for token in tokens:
        outs.append(model(token.reshape(1, 1), cache=cache)[:, -1, :])
    mx.eval(outs)
    return outs


def inputs(rows):
    keys = mx.random.split(mx.random.key(100 + rows), rows + 1)
    prompts = [
        mx.random.randint(0, VOCAB, (PROMPT_LENS[i],), key=keys[i]) for i in range(rows)
    ]
    steps = mx.random.randint(0, VOCAB, (STEPS, rows), key=keys[rows])
    return prompts, [steps[i] for i in range(STEPS)]


def all_same(got, want):
    return len(got) == len(want) and all(same(g, w) for g, w in zip(got, want))


def fused_blocks(core):
    return [l.mlp for l in core.layers if type(l.mlp) in fb._fused_classes.values()]


def test_model(tag, model, expect_routers):
    core = model.model
    patch_model_attention(model)
    moe = sum(isinstance(l.mlp, fl.SparseMoeBlock) for l in core.layers)
    stock = {rows: run_requests(model, *inputs(rows)) for rows in range(1, 6)}
    prompts, steps = inputs(1)
    native = native_steps(model, prompts[0], [s[0] for s in steps])

    # The harness itself: the wrapper's batched path against mlx-lm's forward.
    spread = max(mx.abs(o).max().item() for o in stock[1][1])
    drift = max(
        mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        for a, b in zip(stock[1][1], native[1:])
    )
    limit = 1e-3 if core.norm.weight.dtype == mx.float32 else 0.25
    check(f"{tag}: the context harness tracks mlx-lm's own decode",
          drift <= limit * max(spread, 1.0), f"max abs diff {drift:.2e}, logits to {spread:.1f}")

    mx.random.seed(7)
    before = mx.random.normal((4,))
    mx.random.seed(7)
    patched = fb.patch_flint_moe_blocks(model)
    after = mx.random.normal((4,))
    blocks = fused_blocks(core)
    routers = sum(b._sglang_router_kernel for b in blocks)
    rows_served = min((b._sglang_fused_rows for b in blocks), default=0)
    print(f"     {tag}: {patched} of {moe} MoE blocks, {routers} router kernels, "
          f"steps of up to {rows_served}")
    check(f"{tag}: every MoE block is patched", patched == moe and len(blocks) == moe)
    check(f"{tag}: the install leaves the global random state alone", same(before, after))
    check(f"{tag}: steps of up to {fb.MAX_ROWS} requests", rows_served == fb.MAX_ROWS)
    if expect_routers:
        check(f"{tag}: a router kernel on every MoE block", routers == moe)

    for rows in range(1, 6):
        prefill, outs, state = run_requests(model, *inputs(rows))
        want_prefill, want_outs, want_state = stock[rows]
        diff = sum(not same(g, w) for g, w in zip(outs, want_outs))
        check(f"{tag}: {rows} request(s), {STEPS} steps equal the stock logits",
              diff == 0 and len(outs) == len(want_outs), f"{diff} steps differ" if diff else "")
        check(f"{tag}: {rows} request(s), cache contents equal", all_same(state, want_state))
        check(f"{tag}: {rows} request(s), prefill equals the stock forward",
              all_same(prefill, want_prefill))
    check(f"{tag}: mlx-lm's own decode loop equals the stock logits",
          all_same(native_steps(model, prompts[0], [s[0] for s in steps]), native))

    block = blocks[0]
    x = (2 * mx.random.normal((1, 1, core.args.hidden_size))).astype(core.norm.weight.dtype)
    want = block(x)
    block.train()
    kept = same(block(x), block._sglang_stock_call(x))
    block.eval()
    check(f"{tag}: a block in training mode keeps the stock call",
          kept and same(block(x), want))
    cls = type(block)
    check(f"{tag}: idempotent",
          fb.patch_flint_moe_blocks(model) == patched and type(block) is cls)


def test_scope():
    check("scope: another model is left alone", fb.patch_flint_moe_blocks(nn.Linear(8, 8)) == 0)
    tree = ast.parse(open(model_runner.__file__).read())
    helper = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_install_decode_fusions"
    ]
    check("install site: the helper both load paths call reads the flag",
          len(helper) == 1 and "SGLANG_MLX_FUSE_FLINT_MOE" in ast.dump(helper[0]))


if __name__ == "__main__":
    print("module:", fb.__file__)
    test_kernels()
    for tag, expect_routers, kwargs in (
        ("hidden 128 bf16", False,
         dict(hidden=128, heads=2, kv=1, experts=8, topk=3, dtype=mx.bfloat16, seed=3)),
        ("hidden 512 bf16", False,
         dict(hidden=512, heads=8, kv=2, experts=16, topk=6, dtype=mx.bfloat16, seed=4)),
        ("hidden 512 float32", False,
         dict(hidden=512, heads=8, kv=2, experts=16, topk=6, dtype=mx.float32, seed=5)),
        ("hidden 2048, 256 experts, bf16", True,
         dict(hidden=2048, heads=32, kv=4, experts=256, topk=6, dtype=mx.bfloat16, seed=6,
              layers=4)),
    ):
        test_model(tag, build(**kwargs), expect_routers)
    test_scope()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("ALL PASS")
