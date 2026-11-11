#!/usr/bin/env python3
"""Unit test for per-request mRoPE deltas on mlx-vlm checkpoints (patch 034).

A small random mlx-vlm Qwen3.5 language model on the Metal device; no server
and no model download.

mlx-vlm's Qwen language models number every token forwarded on a cache with
history as cache offset plus ``_rope_deltas``, one field on the model that
each prefill overwrites: zero after a text prompt, negative after an image.
With two live requests the one that prefilled first would decode its later
tokens under the other's delta. The runner keeps the delta per request and
puts it back before each of the request's forwards.

  1. Solo references: an image request and a text request, each decoded alone.
  2. Image first, text arrives mid-decode: with the restore both requests
     match their solo logits; without it the image request moves.
  3. Text first, image arrives mid-decode: same, the text request moves
     without the restore.
  4. Chunked prefill: a text prompt whose second chunk runs after an image
     request prefilled matches the solo chunked prefill only with the restore.
  5. One live request: the restore never reassigns the field.
  6. Call sites: prefill_start drops a stale delta, extend_start restores
     before its first forward, every prompt forward keeps the delta, both
     serial decode call sites pass the request ids, remove_request and clear
     drop the record, and a runner built without __init__ needs none of it.

  python scripts/eval/test_request_position_delta_unit.py
"""
import ast
import sys

import numpy as np

import tiny_mrope_model as tiny
from tiny_mrope_model import check, mr, mx

STEPS, EARLY = 8, 2


def decode(r, prompts, caches, toks, steps, restore=True):
    """Serial decode steps; returns per-step logits [steps, B, vocab]."""
    out = []
    toks = list(toks)
    for _ in range(steps):
        inputs = [mx.array([[t]], dtype=mx.int32) for t in toks]
        ids = [p.rid for p in prompts] if restore else None
        step = tiny.rows(r._decode_with_native_cache(caches, inputs, ids), caches)
        out.append(step)
        toks = [int(row.argmax()) for row in step]
    return np.stack(out), toks


def solo(r, prompt):
    cache = r._new_native_cache()
    tok = tiny.forward_prompt(r, prompt, cache)
    logits, _ = decode(r, [prompt], [cache], [tok], STEPS)
    return logits[:, 0]


def interleaved(r, first, second, restore):
    """``first`` decodes EARLY steps alone, ``second`` prefills, both decode on."""
    c1 = r._new_native_cache()
    t1 = tiny.forward_prompt(r, first, c1)
    head, (t1,) = decode(r, [first], [c1], [t1], EARLY, restore)
    c2 = r._new_native_cache()
    t2 = tiny.forward_prompt(r, second, c2)
    both, _ = decode(r, [first, second], [c1, c2], [t1, t2], STEPS - EARLY, restore)
    return np.concatenate([head[:, 0], both[:, 0]]), both[:, 1]


def main():
    rng = np.random.default_rng(5)
    lm = tiny.make_lm()
    r = tiny.make_runner(lm)
    print("module:", mr.__file__)
    image = tiny.image_prompt("img", rng, head=9, grid=(1, 8, 12), tail=7)
    text = tiny.text_prompt("txt", rng, 31)

    # 1. Solo references.
    ref_image = solo(r, image)
    image_delta = int(r._req_rope_delta["img"].reshape(-1)[0].item())
    ref_text = solo(r, text)
    text_delta = int(r._req_rope_delta["txt"].reshape(-1)[0].item())
    scale = float(max(np.abs(ref_image).max(), np.abs(ref_text).max()))
    check("solo: the image prompt leaves a negative delta", image_delta < 0, f"delta {image_delta}")
    check("solo: the text prompt leaves a zero delta", text_delta == 0)

    def close(a, b):
        return float(np.abs(a - b).max()) <= 1e-4 * scale

    def diff(a, b):
        return f"max abs diff {np.abs(a - b).max():.2e} of {scale:.2f}"

    # 2. Image request first, text request arrives while it decodes.
    got_image, got_text = interleaved(r, image, text, restore=True)
    check("image first: the image request matches its solo decode", close(got_image, ref_image), diff(got_image, ref_image))
    check("image first: the text request matches its solo decode", close(got_text, ref_text[: STEPS - EARLY]), diff(got_text, ref_text[: STEPS - EARLY]))
    bad_image, _ = interleaved(r, image, text, restore=False)
    check("image first, no restore: steps before the arrival still match", close(bad_image[:EARLY], ref_image[:EARLY]))
    moved = float(np.abs(bad_image[EARLY:] - ref_image[EARLY:]).max())
    check("image first, no restore: steps after the arrival move", moved > 1e-2 * scale, f"max abs diff {moved:.2e} of {scale:.2f}")

    # 3. Text request first, image request arrives while it decodes.
    got_text, got_image = interleaved(r, text, image, restore=True)
    check("text first: the text request matches its solo decode", close(got_text, ref_text), diff(got_text, ref_text))
    check("text first: the image request matches its solo decode", close(got_image, ref_image[: STEPS - EARLY]), diff(got_image, ref_image[: STEPS - EARLY]))
    bad_text, _ = interleaved(r, text, image, restore=False)
    moved = float(np.abs(bad_text[EARLY:] - ref_text[EARLY:]).max())
    check("text first, no restore: steps after the arrival move", moved > 1e-2 * scale, f"max abs diff {moved:.2e} of {scale:.2f}")

    # 4. A chunked text prefill with an image prefill between its chunks.
    cut = 17

    def chunked(between, restore):
        cache = r._new_native_cache()
        tiny.forward_prompt(r, text, cache, ids=text.ids[:cut])
        if between:
            tiny.forward_prompt(r, image, r._new_native_cache())
        if restore:
            r._restore_position_delta("txt")
        logits = r.model(mx.array([text.ids[cut:]], dtype=mx.int32), cache=cache)
        return tiny.rows(logits, [cache])[0]

    ref_chunks = chunked(between=False, restore=True)
    got = chunked(between=True, restore=True)
    check("chunked: the second chunk after another prefill matches", close(got, ref_chunks), diff(got, ref_chunks))
    moved = float(np.abs(chunked(between=True, restore=False) - ref_chunks).max())
    check("chunked, no restore: the second chunk moves", moved > 1e-2 * scale, f"max abs diff {moved:.2e} of {scale:.2f}")

    # 5. One live request: the field is never reassigned.
    cache = r._new_native_cache()
    tok = tiny.forward_prompt(r, text, cache)
    kept = r._req_rope_delta["txt"]
    check("single: the kept delta is the array the model holds", lm._rope_deltas is kept)
    decode(r, [text], [cache], [tok], 3)
    check("single: decode steps leave the model's field alone", lm._rope_deltas is kept)
    # A runner built without __init__ (upstream unit tests do this) is a
    # text-path runner by class default and needs no delta bookkeeping.
    bare = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    bare._keep_position_delta("x")
    bare._restore_position_delta("x")
    check(
        "bare runner: text path by default, nothing kept or restored",
        bare.is_vlm is False and not hasattr(bare, "_req_rope_delta"),
    )

    # 6. Call sites.
    src = open(mr.__file__).read()
    fns = {n.name: n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}

    def calls(fn, name):
        return [
            n
            for n in ast.walk(fns[fn])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name
        ]

    def text_of(fn):
        return ast.get_source_segment(src, fns[fn])

    check("prefill_start drops the request's stale delta", "self._req_rope_delta.pop(req_id, None)" in text_of("prefill_start"))
    restore = calls("extend_start", "_restore_position_delta")
    forwards = calls("extend_start", "_forward_lazy_token") + calls("extend_start", "model") + calls("extend_start", "_trunk_forward")
    check(
        "extend_start restores before its first forward",
        len(restore) == 1 and all(restore[0].lineno < f.lineno for f in forwards),
    )
    check("_forward_lazy_token keeps the delta after the forward", len(calls("_forward_lazy_token", "_keep_position_delta")) == 1)
    check(
        "prefill_start and extend_start forward through _forward_lazy_token",
        len(calls("prefill_start", "_forward_lazy_token")) >= 3 and len(calls("extend_start", "_forward_lazy_token")) == 1,
    )
    for fn, ids in (("decode_batch_start", "req_ids"), ("decode_batch_start_chained", "prev.req_ids")):
        native = calls(fn, "_decode_with_native_cache")
        check(
            f"{fn} passes the request ids to the serial decode",
            len(native) == 1 and len(native[0].args) == 3 and ast.unparse(native[0].args[2]) == ids,
        )
    check("remove_request drops the delta", "self._req_rope_delta.pop(req_id, None)" in text_of("remove_request"))
    check("clear drops every delta", "self._req_rope_delta.clear()" in text_of("clear"))
    check(
        "the cleanup reads the delta record only on VLM runners",
        all(
            "if self.is_vlm:" in text_of(fn)
            and text_of(fn).index("if self.is_vlm:") < text_of(fn).index("self._req_rope_delta.")
            for fn in ("prefill_start", "remove_request", "clear")
        ),
    )

    print(f"\n{'ALL PASS' if not tiny.FAILS else 'FAILED: ' + ', '.join(tiny.FAILS)}")
    sys.exit(1 if tiny.FAILS else 0)


if __name__ == "__main__":
    main()
