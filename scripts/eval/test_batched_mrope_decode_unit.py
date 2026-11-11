#!/usr/bin/env python3
"""Unit test for batched decode on mRoPE hybrids (patch 035).

A small random mlx-vlm Qwen3.5 language model on the Metal device; no server
and no model download.

mlx-vlm's Qwen-family attention rotates through ``rotary_emb.apply_rotary``
from explicit positions, so the runner's batched decode (which looks for
``rope``) left those models decoding one request per forward. With
``SGLANG_MLX_BATCH_MROPE_DECODE`` a step of two or more requests runs as one
forward: every request's token is numbered from its own cache offset plus its
own mRoPE delta, and attends over its own cache.

  1. Batched steps against each request's solo serial decode, for 2, 3 and 4
     requests of unequal length, text and image prompts mixed.
  2. Membership changes: requests that decode batched and then alone, and a
     request that decodes alone and is then joined, keep their solo logits.
  3. Positions: the step context carries offset plus delta per request.
  4. Routing: only steps of two or more requests batch, only with the flag,
     and models with ``rope`` are untouched.
  5. Bookkeeping: the int delta is read once per request and dropped with it.

  python scripts/eval/test_batched_mrope_decode_unit.py
"""
import ast
import sys

import numpy as np

import tiny_mrope_model as tiny
from tiny_mrope_model import check, mr, mx

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    MLXAttentionWrapper,
)

STEPS = 6


def serial_step(r, prompts, caches, toks):
    inputs = [mx.array([[t]], dtype=mx.int32) for t in toks]
    return tiny.rows(
        r._decode_with_native_cache(caches, inputs, [p.rid for p in prompts]), caches
    )


def batched_step(r, prompts, caches, toks):
    batch = mx.array(toks, dtype=mx.int32)[:, None]
    return tiny.rows(
        r._decode_with_hybrid_batching(caches, batch, [p.rid for p in prompts]), caches
    )


def step(r, prompts, caches, toks):
    """One decode step the way decode_batch_start routes it."""
    fn = batched_step if r._batches_decode(len(prompts)) else serial_step
    return fn(r, prompts, caches, toks)


def solo(r, prompt, steps=STEPS):
    """Serial decode of one request alone: per-step logits and the tokens fed."""
    cache = r._new_native_cache()
    tok = tiny.forward_prompt(r, prompt, cache)
    logits, fed = [], []
    for _ in range(steps):
        fed.append(tok)
        row = serial_step(r, [prompt], [cache], [tok])[0]
        logits.append(row)
        tok = int(row.argmax())
    return np.stack(logits), fed


def main():
    rng = np.random.default_rng(11)
    lm = tiny.make_lm()
    r = tiny.make_runner(lm)
    print("module:", mr.__file__)

    prompts = [
        tiny.text_prompt("t0", rng, 23),
        tiny.image_prompt("i0", rng, head=5, grid=(1, 8, 12), tail=9),
        tiny.text_prompt("t1", rng, 61),
        tiny.image_prompt("i1", rng, head=14, grid=(1, 4, 6), tail=3),
    ]
    refs = {p.rid: solo(r, p) for p in prompts}
    deltas = {p.rid: int(r._req_rope_delta[p.rid].reshape(-1)[0].item()) for p in prompts}
    scale = float(max(np.abs(ref).max() for ref, _ in refs.values()))
    check(
        "prompts: text deltas are zero, image deltas negative and distinct",
        deltas["t0"] == deltas["t1"] == 0 and deltas["i0"] < 0 and deltas["i1"] < 0 and deltas["i0"] != deltas["i1"],
        str(deltas),
    )

    def worst(got, want):
        return float(np.abs(got - want).max())

    def same(got, want):
        return worst(got, want) <= 1e-4 * scale and bool((got.argmax(-1) == want.argmax(-1)).all())

    r._mrope_batched_decode = True

    # 1. Batched steps against solo serial decode.
    for members in (["t0", "i0"], ["i0", "i1"], ["t1", "i1", "t0"], ["t0", "i0", "t1", "i1"]):
        group = [p for p in prompts if p.rid in members]
        group.sort(key=lambda p: members.index(p.rid))
        caches = [r._new_native_cache() for _ in group]
        for p, cache in zip(group, caches):
            tiny.forward_prompt(r, p, cache)
        got = np.stack(
            [batched_step(r, group, caches, [refs[p.rid][1][s] for p in group]) for s in range(STEPS)]
        )
        want = np.stack([refs[p.rid][0] for p in group], axis=1)
        check(
            f"batched {'+'.join(members)}: {STEPS} steps match solo decode",
            same(got, want),
            f"max abs diff {worst(got, want):.2e} of {scale:.2f}",
        )

    # The control: the same batch with every delta forced to zero moves the
    # image requests and leaves the text request alone.
    group = [prompts[0], prompts[1]]
    caches = [r._new_native_cache() for _ in group]
    for p, cache in zip(group, caches):
        tiny.forward_prompt(r, p, cache)
    r._req_position_delta.update({p.rid: 0 for p in group})
    got = batched_step(r, group, caches, [refs[p.rid][1][0] for p in group])
    r._req_position_delta.clear()
    check("control, deltas zeroed: the text request still matches", worst(got[0], refs["t0"][0][0]) <= 1e-4 * scale)
    moved = worst(got[1], refs["i0"][0][0])
    check("control, deltas zeroed: the image request moves", moved > 1e-2 * scale, f"max abs diff {moved:.2e} of {scale:.2f}")

    # 2. Membership changes, routed the way decode_batch_start does.
    a, b, c = prompts[1], prompts[2], prompts[0]
    ca, cb, cc = (r._new_native_cache() for _ in range(3))
    tiny.forward_prompt(r, a, ca)
    got_a = [step(r, [a], [ca], [refs["i0"][1][s]])[0] for s in range(2)]  # alone
    tiny.forward_prompt(r, b, cb)
    got_b = []
    for s in range(2, 4):  # joined by b
        out = step(r, [a, b], [ca, cb], [refs["i0"][1][s], refs["t1"][1][s - 2]])
        got_a.append(out[0])
        got_b.append(out[1])
    tiny.forward_prompt(r, c, cc)
    got_c = []
    for s in range(4, 6):  # c joins, a finishes after this
        out = step(r, [a, b, c], [ca, cb, cc], [refs["i0"][1][s], refs["t1"][1][s - 2], refs["t0"][1][s - 4]])
        got_a.append(out[0])
        got_b.append(out[1])
        got_c.append(out[2])
    for s in range(4, 6):  # b alone again
        got_b.append(step(r, [b], [cb], [refs["t1"][1][s]])[0])
    check("alone, joined, joined again: the image request matches", same(np.stack(got_a), refs["i0"][0]), f"max abs diff {worst(np.stack(got_a), refs['i0'][0]):.2e}")
    check("joins, batched, then alone: the text request matches", same(np.stack(got_b), refs["t1"][0]), f"max abs diff {worst(np.stack(got_b), refs['t1'][0]):.2e}")
    check("joins a running batch: the late request matches", same(np.stack(got_c), refs["t0"][0][:2]), f"max abs diff {worst(np.stack(got_c), refs['t0'][0][:2]):.2e}")
    check(
        "alone again: the serial path finds its own delta on the model",
        lm._rope_deltas is r._req_rope_delta["t1"],
    )

    # 3. Positions in the step context.
    ctx = r._build_batched_decode_context([ca, cb], ["i0", "t1"])
    ids = np.array(ctx.mrope_position_ids())
    offsets = np.array(ctx.offsets)
    check("context: positions are (3, B, 1)", ids.shape == (3, 2, 1))
    check(
        "context: each position is the cache offset plus the request's delta",
        bool((ids[:, :, 0] == (offsets + np.array([deltas["i0"], deltas["t1"]]))[None, :]).all()),
        f"offsets {offsets.tolist()} deltas {[deltas['i0'], deltas['t1']]} positions {ids[0, :, 0].tolist()}",
    )
    check("context: one array per step", ctx.mrope_position_ids() is ctx.mrope_position_ids())

    # 4. Routing.
    check("routing: one request keeps the model's own forward", r._batches_decode(1) is False)
    check("routing: two or more requests batch", r._batches_decode(2) and r._batches_decode(5))
    r._mrope_batched_decode = False
    check("routing: flag off, every step is serial", not r._batches_decode(1) and not r._batches_decode(4))
    check(
        "routing: flag off, the context carries no deltas",
        r._build_batched_decode_context([ca, cb], ["i0", "t1"]).position_deltas is None,
    )
    r._attention_wrapper_batchable = True
    check("routing: rope models batch every step as before", r._batches_decode(1) and r._batches_decode(3))
    r._attention_wrapper_batchable = False
    bare = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    check("routing: a bare runner defaults to serial", bare._mrope_batched_decode is False)
    check("flag: off unless set", envs.SGLANG_MLX_BATCH_MROPE_DECODE.get() is False)
    wrappers = [m for _, m in lm.named_modules() if isinstance(m, MLXAttentionWrapper)]
    check("wrapper: mlx_vlm attention resolves its rotary embedding", len(wrappers) == 2 and all(w._mrope is not None for w in wrappers))

    # 5. Bookkeeping.
    r._mrope_batched_decode = True
    r._req_position_delta.clear()
    first = r._position_delta("i0")
    check("delta: the int equals the request's kept delta", first == deltas["i0"] and r._req_position_delta == {"i0": deltas["i0"]})
    check("delta: a request with no record reads zero", r._position_delta("never-prefilled") == 0)
    tiny.forward_prompt(r, prompts[0], r._new_native_cache())
    r._position_delta("t0")
    src = open(mr.__file__).read()
    fns = {
        n.name: ast.get_source_segment(src, n)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef)
    }
    for fn in ("prefill_start", "remove_request"):
        check(f"{fn} drops the int delta", "self._req_position_delta.pop(req_id, None)" in fns[fn])
    check("clear drops every int delta", "self._req_position_delta.clear()" in fns["clear"])

    print(f"\n{'ALL PASS' if not tiny.FAILS else 'FAILED: ' + ', '.join(tiny.FAILS)}")
    sys.exit(1 if tiny.FAILS else 0)


if __name__ == "__main__":
    main()
