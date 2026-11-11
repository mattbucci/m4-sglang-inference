#!/usr/bin/env python3
"""Does a batched decode step depend on a request's row or its batch-mates?

Raw MLX plus the runner's own decode paths, no server. One target request is
teacher-forced through the same tokens in several arrangements and its logits
are compared bit for bit:

  native [A]         the model's own forward on one request (the serial path)
  batched [A]        patch 035's layer walk with a batch of one
  batched [A,B]      first of two         batched [B,A]      last of two
  batched [C,A]      another batch-mate   batched [A,B,C,D]  first of four
  batched [B,C,D,A]  last of four         batched [B,A,C,D]  in the middle

A served request's greedy text under ``SGLANG_MLX_BATCH_MROPE_DECODE=1`` can
leave its single-stream text (``probe_thinking_under_load.py``). This probe
says what kind of difference that is. Logits that are bit-equal across every
batched arrangement of two or more, and differ from ``native [A]`` by the
noise ``batched_decode_fidelity.py`` measures between two correct paths, mean
one thing: a row is computed the same whatever its index, the batch size and
the neighbours, and the only difference is between the one-row and the
several-row kernels. A row whose logits moved with its index or its
neighbours would be a defect in the batched path.

    python scripts/eval/batched_decode_row_symmetry.py --model <dir> --out <json>

Needs the GPU to itself: do not run it beside a serving arm.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))

import tiny_mrope_model as tiny  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx_decode_census as census  # noqa: E402
from batched_decode_fidelity import ROOT, corpus_ids, logprobs  # noqa: E402

ARRANGEMENTS = [
    ("native [A]", False, [0]),
    ("batched [A]", True, [0]),
    ("batched [A,B]", True, [0, 1]),
    ("batched [B,A]", True, [1, 0]),
    ("batched [C,A]", True, [2, 0]),
    ("batched [A,B,C,D]", True, [0, 1, 2, 3]),
    ("batched [B,C,D,A]", True, [1, 2, 3, 0]),
    ("batched [B,A,C,D]", True, [1, 0, 2, 3]),
]


def differ(ref, got):
    """Per-step comparison of two [steps, vocab] logit stacks."""
    a, b = ref.astype(mx.float32), got.astype(mx.float32)
    gap = mx.abs(a - b).max(axis=-1)
    la, lb = logprobs(ref), logprobs(got)
    kl = (mx.exp(la) * (la - lb)).sum(axis=-1)
    same = mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)
    mx.eval(gap, kl, same)
    gap = [float(v) for v in gap.tolist()]
    kl = [float(v) for v in kl.tolist()]
    return {
        "steps": len(gap),
        "bit_equal_steps": sum(1 for v in gap if v == 0.0),
        "max_abs_logit_gap": max(gap),
        "kl_mean": sum(kl) / len(kl),
        "kl_max": max(kl),
        "top1_same": int(same.sum().item()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="742,311,1530,405", help="prompt tokens for A,B,C,D; A is the target")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--root", default=ROOT, help="repo root; its README and docs are the forced text")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model_dir = census.resolve_model_dir(a.model)
    eng = census.Engine(model_dir)
    lengths = [int(x) for x in a.lengths.split(",")]
    if len(lengths) != 4:
        raise SystemExit("--lengths takes four values")
    S = a.steps
    ids = corpus_ids(model_dir, a.root)
    reqs, at = [], 0
    for i, n in enumerate(lengths):
        chunk = ids[at : at + n + S + 1]
        at += n + S + 1
        reqs.append({"prompt": tiny.Prompt(f"r{i}", chunk[:n], None), "forced": chunk[n : n + S + 1]})
    r = tiny.make_runner(eng.lm, max_seq_len=max(lengths) + S + 64)
    print(f"module: {tiny.mr.__file__}", flush=True)
    t0 = time.time()

    def run(batched, order):
        r._mrope_batched_decode = batched
        qs = [reqs[i] for i in order]
        caches = [r._new_native_cache() for _ in qs]
        for q, cache in zip(qs, caches):
            tiny.forward_prompt(r, q["prompt"], cache)
        rids = [q["prompt"].rid for q in qs]
        k = order.index(0)
        rows = []
        for j in range(S):
            toks = [q["forced"][j] for q in qs]
            if batched:
                out = r._decode_with_hybrid_batching(caches, mx.array(toks, dtype=mx.int32)[:, None], rids)
            else:
                out = r._decode_with_native_cache(caches, [mx.array([[t]], dtype=mx.int32) for t in toks], rids)
            row = out[k]
            mx.eval(row, [c.state for cache in caches for c in cache])
            rows.append(row)
        out = mx.stack(rows)
        mx.eval(out)
        mx.clear_cache()
        return out

    logits = {}
    for name, batched, order in ARRANGEMENTS:
        logits[name] = run(batched, order)
        print(f"{name:<18} done {time.time() - t0:.0f}s  dtype {logits[name].dtype}", flush=True)
    # The serial path again: it has to repeat itself bit for bit, or bit
    # equality between arrangements would say nothing.
    logits["native [A] again"] = run(False, [0])

    result = {"model": a.model, "steps": S, "lengths": dict(zip("ABCD", lengths)), "arrangements": []}
    print("\narrangement         against native [A]: bit-equal  max gap   KL mean   KL max  top-1   against batched [A,B]: bit-equal  max gap")
    for name in list(logits):
        n = differ(logits["native [A]"], logits[name])
        b = differ(logits["batched [A,B]"], logits[name])
        result["arrangements"].append({"name": name, "against_native": n, "against_batched_AB": b})
        print(
            f"{name:<18}  {n['bit_equal_steps']:>20}/{S:<3} {n['max_abs_logit_gap']:8.4f}  {n['kl_mean']:.5f}  {n['kl_max']:.5f}  "
            f"{n['top1_same']:>3}/{S:<3} {b['bit_equal_steps']:>24}/{S:<3} {b['max_abs_logit_gap']:8.4f}",
            flush=True,
        )
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
