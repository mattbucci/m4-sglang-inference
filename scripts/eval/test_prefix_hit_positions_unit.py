#!/usr/bin/env python3
"""Unit test for prefix-hit token positions on mlx-vlm checkpoints (patch 033).

A small random mlx-vlm Qwen3.5 language model on the Metal device; no server
and no model download.

mlx-vlm's Qwen language models keep two pieces of position state between
calls. The runner clears both at every new prefill so one request cannot leak
positions into the next. On a radix prefix hit the forward then starts on a
cache whose offset is the prefix length, and with the state cleared mlx-vlm
numbers the forwarded tokens from zero. The runner seeds the text-only delta
on that path so the positions come off the cache offset.

  1. Cold reference: a prefix and a tail fed as two calls on one cache (what
     chunked prefill does) give the logits of the whole prompt in one call.
  2. Prefix hit: a restored prefix cache, then reset + seed, gives the two-call
     logits bit for bit, for the tail in one call and split in two, and the
     decode steps after it stay bit-equal.
  3. Control: reset without the seed moves the tail logits and the decode
     step after it, so the test sees the defect it guards.
  4. Seed state: it is the state a cold text prefill leaves (shape, dtype,
     value) and it reuses no position tensor of the previous request.
  5. Call site: prefill_start seeds only on the pool-backed prefix path, after
     the full-prompt fallback has returned.

  python scripts/eval/test_prefix_hit_positions_unit.py
"""
import ast
import os
import sys
import types

import numpy as np

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as lang  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402

from sglang.srt.hardware_backend.mlx import model_runner as mr  # noqa: E402

FAILS = []
PREFIX, TAIL, STEPS = 23, 9, 3


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_lm(seed=7):
    args = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=96,
        intermediate_size=192,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=8,
        num_attention_heads=3,
        rms_norm_eps=1e-6,
        vocab_size=211,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 100.0,
            "partial_rotary_factor": 0.5,
        },
        full_attention_interval=4,
    )
    config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2),
        image_token_id=208,
        video_token_id=209,
        vision_start_token_id=210,
    )
    mx.random.seed(seed)
    lm = lang.LanguageModel(args, config)
    # Default inits leave the attention path too faint to tell positions
    # apart; widen every weight matrix so the logits depend on them.
    def widen(_, value):
        return value * 4.0 if value.ndim >= 2 else value

    from mlx.utils import tree_map_with_path

    lm.update(tree_map_with_path(widen, lm.parameters()))
    lm.set_dtype(mx.float32)
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def runner_for(lm):
    """The two runner helpers only touch ``self.model.language_model``."""
    return types.SimpleNamespace(model=types.SimpleNamespace(language_model=lm))


def reset(lm):
    mr.MlxModelRunner._reset_mlx_vlm_position_cache(runner_for(lm))


def seed(lm):
    mr.MlxModelRunner._seed_mlx_vlm_prefix_positions(runner_for(lm))


def call(lm, ids, cache):
    out = lm(mx.array([ids], dtype=mx.int32), cache=cache)
    logits = out.logits
    mx.eval(logits, [c.state for c in cache])
    return np.array(logits[0].astype(mx.float32))


def snapshot(cache):
    def copy(v):
        if isinstance(v, (list, tuple)):
            return [copy(x) for x in v]
        return None if v is None else mx.array(np.array(v))

    return [copy(c.state) for c in cache]


def restored(lm, snap):
    cache = lm.make_cache()
    for c, s in zip(cache, snap):
        c.state = [None if x is None else mx.array(np.array(x)) for x in s]
    return cache


def decode(lm, cache, first):
    """Greedy steps from ``first``; returns the per-step logits."""
    rows, tok = [], int(first)
    for _ in range(STEPS):
        row = call(lm, [tok], cache)[-1]
        rows.append(row)
        tok = int(row.argmax())
    return np.stack(rows)


def main():
    rng = np.random.default_rng(11)
    ids = [int(t) for t in rng.integers(0, 200, PREFIX + TAIL)]
    head, tail = ids[:PREFIX], ids[PREFIX:]
    lm = make_lm()
    print("module:", mr.__file__)

    # 1. Cold reference.
    reset(lm)
    one = call(lm, ids, lm.make_cache())[PREFIX:]
    reset(lm)
    cache = lm.make_cache()
    call(lm, head, cache)
    cold_deltas = lm._rope_deltas
    snap = snapshot(cache)
    two = call(lm, tail, cache)
    nxt = int(two[-1].argmax())
    two_dec = decode(lm, cache, nxt)
    scale = float(np.abs(one).max())
    check(
        "cold: prefix then tail on one cache equals the whole prompt",
        float(np.abs(two - one).max()) <= 1e-4 * scale,
        f"max abs diff {np.abs(two - one).max():.2e} of {scale:.2f}",
    )

    # 2. Prefix hit with the seed.
    reset(lm)
    seed(lm)
    cache = restored(lm, snap)
    hit = call(lm, tail, cache)
    hit_dec = decode(lm, cache, nxt)
    check("hit: seeded tail is bit-equal to the cold two-call tail", np.array_equal(hit, two))
    check("hit: decode steps after a seeded tail are bit-equal", np.array_equal(hit_dec, two_dec))

    reset(lm)
    seed(lm)
    cache = restored(lm, snap)
    cut = TAIL // 2
    call(lm, tail[:cut], cache)
    split = call(lm, tail[cut:], cache)
    check(
        "hit: a tail split in two (tracked-state split) ends on the same logits",
        float(np.abs(split - two[cut:]).max()) <= 1e-4 * scale,
        f"max abs diff {np.abs(split - two[cut:]).max():.2e}",
    )
    check(
        "hit: no position tensor is stored by a seeded hit",
        lm._position_ids is None,
    )

    # 3. Control: the reset alone numbers the tail from zero.
    reset(lm)
    cache = restored(lm, snap)
    bad = call(lm, tail, cache)
    bad_dec = decode(lm, cache, nxt)
    moved = float(np.abs(bad - two).max())
    check(
        "control: reset without the seed moves the tail logits",
        moved > 1e-2 * scale,
        f"max abs diff {moved:.2e} of {scale:.2f}",
    )
    check(
        "control: the decode steps after an unseeded tail move too",
        float(np.abs(bad_dec - two_dec).max()) > 1e-3 * scale,
        f"max abs diff {np.abs(bad_dec - two_dec).max():.2e}",
    )

    # 4. Seed state equals what a cold text prefill leaves.
    reset(lm)
    check("reset: both position fields are cleared", lm._position_ids is None and lm._rope_deltas is None)
    seed(lm)
    s = lm._rope_deltas
    check(
        "seed: the delta a cold text prefill leaves (shape, dtype, value)",
        s is not None
        and s.shape == cold_deltas.shape
        and s.dtype == cold_deltas.dtype
        and bool(mx.array_equal(s, cold_deltas)),
        f"seed {None if s is None else (s.shape, s.dtype)} cold {(cold_deltas.shape, cold_deltas.dtype)}",
    )
    check("seed: the previous request's positions are not reused", lm._position_ids is None)
    plain = types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace()))
    mr.MlxModelRunner._seed_mlx_vlm_prefix_positions(plain)
    check(
        "seed: a language model without the field is left alone",
        not hasattr(plain.model.language_model, "_rope_deltas"),
    )
    mr.MlxModelRunner._seed_mlx_vlm_prefix_positions(types.SimpleNamespace(model=types.SimpleNamespace()))
    check("seed: a model without a language_model is left alone", True)

    # 5. Call site.
    tree = ast.parse(open(mr.__file__).read())
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prefill_start"]
    hits = [
        n
        for n in ast.walk(fn[0])
        if isinstance(n, ast.If) and ast.unparse(n.test) == "trusted_prefix_len > 0"
    ] if len(fn) == 1 else []
    ok = len(hits) == 1
    if ok:
        body, orelse = hits[0].body, hits[0].orelse
        last = ast.unparse(body[-1])
        ok = (
            last.startswith("if self.is_vlm:")
            and "self._seed_mlx_vlm_prefix_positions()" in last
            and "_seed_mlx_vlm_prefix_positions" not in "".join(ast.unparse(n) for n in body[:-1])
            and "_seed_mlx_vlm_prefix_positions" not in "".join(ast.unparse(n) for n in orelse)
            and any(isinstance(n, ast.Return) for n in ast.walk(body[-2]))
        )
    check("call site: seeded last on the pool-backed prefix path, after the full-prompt fallback", ok)
    src = ast.unparse(fn[0]) if fn else ""
    check(
        "call site: the reset still runs before the prefix decision",
        0 <= src.find("self._reset_mlx_vlm_position_cache()") < src.find("trusted_prefix_len > 0"),
    )

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: " + "; ".join(FAILS))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
