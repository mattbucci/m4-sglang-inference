#!/usr/bin/env python3
"""Teacher-forced fidelity of batched mRoPE decode on a real checkpoint.

Raw MLX plus the runner's own decode paths, no server. Takes B stretches of
real text of unequal length and recomputes the next-token distribution after
each forced token on three paths that are the same function in exact
arithmetic:

  prefill  one forward over prompt + forced tokens (the yardstick)
  solo     prefill the prompt, then one-token steps, one request per forward
           (the serial decode the server runs without the flag)
  batched  prefill each prompt, then one-token steps of all B requests in one
           forward (``SGLANG_MLX_BATCH_MROPE_DECODE``, patch 035)

``solo`` against ``prefill`` is the rounding cost of stepping at all, the
noise two correct paths show in the model dtype. ``batched`` against ``solo``
is the gate: a batched step that computed a different function (a wrong
position, a neighbour's cache) would sit far above that noise. A served A/B
cannot give this, because greedy arms part ways at the first flipped token.

    python scripts/eval/batched_decode_fidelity.py --model <dir> --out <json>

``--fuse router,front,linears`` installs the decode fusions of patches 030 to
032 first, so both stepping paths are the fused ones. ``--image-grid t,h,w``
puts an image block (placeholder tokens, no vision tower) into the second
request, so its tokens after the image carry a negative mRoPE delta while the
others carry none: the batched step has to rotate each row from its own
positions. A control run repeats the batched steps with the deltas swapped
(the image request numbered as text, the text requests under the image's
delta), which is what one shared delta field hands out.

Needs the GPU to itself: do not run it beside a serving arm.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))

import tiny_mrope_model as tiny  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx_decode_census as census  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def corpus_ids(model_dir, root):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    paths = [os.path.join(root, "README.md"), os.path.join(root, "rules-for-agents.md")]
    paths += sorted(glob.glob(os.path.join(root, "docs", "*.md")))
    text = "\n\n".join(open(p).read() for p in paths if os.path.exists(p))
    return tok.encode(text, add_special_tokens=False).ids


def logprobs(logits):
    x = logits.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def compare(ref, got, targets):
    """Per-step KL(ref || got), argmax agreement and both NLLs on ``targets``."""
    kl = (mx.exp(ref) * (ref - got)).sum(axis=-1)
    same = mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1)
    idx = mx.array(targets, dtype=mx.int32)[:, None]
    nll_ref = -mx.take_along_axis(ref, idx, axis=-1)[:, 0]
    nll_got = -mx.take_along_axis(got, idx, axis=-1)[:, 0]
    mx.eval(kl, same, nll_ref, nll_got)
    kl = [float(v) for v in kl.tolist()]
    return {
        "kl_mean": statistics.mean(kl),
        "kl_max": max(kl),
        "top1_same": int(same.sum().item()),
        "steps": len(kl),
        "nll_ref": float(nll_ref.mean().item()),
        "nll_got": float(nll_got.mean().item()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="311,742,1530,2617", help="prompt tokens per request")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--fuse", default="", help="comma list of router, front, linears")
    ap.add_argument("--image-grid", default="1,56,56", help="t,h,w patches for request 1's image block; empty for text only")
    ap.add_argument("--root", default=ROOT, help="repo root; its README and docs are the forced text")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model_dir = census.resolve_model_dir(a.model)
    eng = census.Engine(model_dir)
    fused = {}
    for f in [f for f in a.fuse.split(",") if f]:
        if f == "router":
            from sglang.srt.hardware_backend.mlx.moe.fused_router import patch_moe_blocks_with_fused_router as patch
        elif f == "front":
            from sglang.srt.hardware_backend.mlx.gated_delta_front import patch_gated_delta_front as patch
        elif f == "linears":
            from sglang.srt.hardware_backend.mlx.fused_linears import patch_fused_decode_linears as patch
        else:
            raise SystemExit(f"unknown fusion {f}")
        fused[f] = patch(eng.model)
    print(f"module: {tiny.mr.__file__}\nfusions installed: {fused}", flush=True)

    lengths = [int(x) for x in a.lengths.split(",")]
    S = a.steps
    ids = corpus_ids(model_dir, a.root)
    need = sum(lengths) + len(lengths) * (S + 1)
    if len(ids) < need:
        raise SystemExit(f"corpus has {len(ids)} tokens, need {need}")
    reqs, at = [], 0
    for i, n in enumerate(lengths):
        chunk = ids[at : at + n + S + 1]
        at += n + S + 1
        # forced[j] is fed at step j; the step predicts forced[j + 1].
        prompt, grid = chunk[:n], None
        if i == 1 and a.image_grid:
            grid = tuple(int(x) for x in a.image_grid.split(","))
            cfg = eng.lm.config
            merge = cfg.vision_config.spatial_merge_size
            block = [cfg.vision_start_token_id] + [cfg.image_token_id] * (
                grid[0] * (grid[1] // merge) * (grid[2] // merge)
            )
            prompt = prompt[:40] + block + prompt[40:]
        reqs.append(
            {
                "prompt": tiny.Prompt(f"r{i}", prompt, grid),
                "forced": chunk[n : n + S + 1],
                "all": prompt + chunk[n:],
            }
        )

    r = tiny.make_runner(eng.lm, max_seq_len=max(len(q["all"]) for q in reqs) + 64)
    t0 = time.time()

    # prefill: one forward over prompt + forced tokens, rows for steps 0..S-1.
    yard = []
    for q in reqs:
        r._reset_mlx_vlm_position_cache()
        n = len(q["prompt"].ids)
        kwargs = {}
        if q["prompt"].grid is not None:
            kwargs["image_grid_thw"] = mx.array([list(q["prompt"].grid)])
        out = r.model(mx.array([q["all"][:-1]], dtype=mx.int32), cache=r._new_native_cache(), **kwargs)
        lp = logprobs(out[0, n : n + S])
        mx.eval(lp)
        yard.append(lp)
        mx.clear_cache()
    print(f"prefill path done {time.time() - t0:.0f}s", flush=True)

    def stepped(batched, wrong_deltas=None):
        r._mrope_batched_decode = batched
        caches = [r._new_native_cache() for _ in reqs]
        for k, (q, cache) in enumerate(zip(reqs, caches)):
            tiny.forward_prompt(r, q["prompt"], cache)
            if wrong_deltas is not None:
                r._req_position_delta[q["prompt"].rid] = wrong_deltas[k]
        prompts = [q["prompt"] for q in reqs]
        rows = [[] for _ in reqs]
        for j in range(S):
            toks = [q["forced"][j] for q in reqs]
            if batched:
                out = r._decode_with_hybrid_batching(
                    caches, mx.array(toks, dtype=mx.int32)[:, None], [p.rid for p in prompts]
                )
            else:
                out = r._decode_with_native_cache(
                    caches, [mx.array([[t]], dtype=mx.int32) for t in toks], [p.rid for p in prompts]
                )
            lp = logprobs(out)
            mx.eval(lp, [c.state for cache in caches for c in cache])
            for i in range(len(reqs)):
                rows[i].append(lp[i])
        out = [mx.stack(x) for x in rows]
        mx.eval(out)
        return out

    solo = stepped(False)
    deltas = [r._position_delta(q["prompt"].rid) for q in reqs]
    print(f"position deltas: {deltas}", flush=True)
    print(f"solo path done {time.time() - t0:.0f}s", flush=True)
    batched = stepped(True)
    print(f"batched path done {time.time() - t0:.0f}s", flush=True)
    # Control: the same batched steps with the deltas the shared model field
    # would have handed out: the image request numbered as text, the text
    # requests numbered with the image's delta. It has to stand out of the
    # noise, or the comparison above could not see a wrong position.
    wrong = [0 if d else min(deltas) for d in deltas]
    control = stepped(True, wrong_deltas=wrong) if any(deltas) else None

    result = {
        "model": a.model, "fused": {k: str(v) for k, v in fused.items()}, "steps": S,
        "lengths": [len(q["prompt"].ids) for q in reqs], "position_deltas": deltas, "requests": [],
    }
    lengths = result["lengths"]
    print("\nrequest  prompt  pair               KL mean    KL max   top-1 same   NLL ref   NLL got")
    for i, q in enumerate(reqs):
        targets = q["forced"][1:]
        row = {"prompt_tokens": lengths[i]}
        for name, ref, got in (
            ("solo vs prefill", yard[i], solo[i]),
            ("batched vs prefill", yard[i], batched[i]),
            ("batched vs solo", solo[i], batched[i]),
        ):
            c = compare(ref, got, targets)
            row[name] = c
            print(
                f"{i:>7}  {lengths[i]:>6}  {name:<18} {c['kl_mean']:.5f}  {c['kl_max']:.5f}  "
                f"{c['top1_same']:>4}/{c['steps']:<4}   {c['nll_ref']:.4f}    {c['nll_got']:.4f}",
                flush=True,
            )
        if control is not None:
            c = compare(solo[i], control[i], targets)
            c["delta"] = wrong[i]
            row["control: wrong delta vs solo"] = c
            print(
                f"{i:>7}  {lengths[i]:>6}  {'CONTROL delta ' + str(wrong[i]):<18} {c['kl_mean']:.5f}  {c['kl_max']:.5f}  "
                f"{c['top1_same']:>4}/{c['steps']:<4}   {c['nll_ref']:.4f}    {c['nll_got']:.4f}",
                flush=True,
            )
        result["requests"].append(row)
    for name in ("solo vs prefill", "batched vs prefill", "batched vs solo"):
        rows = [q[name] for q in result["requests"]]
        total = sum(c["steps"] for c in rows)
        result[name] = {
            "kl_mean": sum(c["kl_mean"] * c["steps"] for c in rows) / total,
            "kl_max": max(c["kl_max"] for c in rows),
            "top1_same": sum(c["top1_same"] for c in rows),
            "steps": total,
            "nll_ref": sum(c["nll_ref"] * c["steps"] for c in rows) / total,
            "nll_got": sum(c["nll_got"] * c["steps"] for c in rows) / total,
        }
        c = result[name]
        print(
            f"    all          {name:<18} {c['kl_mean']:.5f}  {c['kl_max']:.5f}  "
            f"{c['top1_same']:>4}/{c['steps']:<4}   {c['nll_ref']:.4f}    {c['nll_got']:.4f}"
        )
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
