#!/usr/bin/env python3
"""Which path is closer to the checkpoint's own function: prefill or decode?

Raw MLX, no server. Takes a reference dump from `decode_fidelity.py` (a
prompt plus the tokens a served greedy decode produced) and recomputes the
next-token distribution at every decoded position on four paths that are
the same function in exact arithmetic:

  batch32  one forward over prompt + decoded tokens, float32 activations
  step32   prefill the prompt, then one-token steps, float32 activations
  batch16  one forward in the model dtype (the prefill path)
  step16   prefill the prompt, then one-token steps in the model dtype
           (the decode path)

`batch32` is the yardstick: the quantized weights are unchanged and only
the activations carry a 24-bit mantissa instead of bf16's 8. `step32`
against it is the logic check (a decode path that computed a different
function would show here), `batch16` and `step16` against it are the
rounding cost of each path, and the served dump's top-k against it says
how far the tokens the server emitted are from the checkpoint's function.

    python scripts/eval/decode_path_precision.py --model <dir> \
        --ref-dump <decode_fidelity dump> --out <json>

`--fuse router,front,linears` installs the decode fusions of patches 030
to 032 before the run, so `step16` is the fused decode path on the same
tokens: a teacher-forced fidelity gate for a decode-only kernel, which a
served A/B cannot give because its arms part ways at the first flipped
token. Use `--paths step16` with it; the yardstick is a prefill and
keeps the stock calls.

Needs the GPU to itself: do not run it beside a serving arm.
"""
import argparse, json, math, os, statistics, sys, time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import mlx_decode_census as census  # noqa: E402

FAR = 0.5  # nats of KL from the yardstick that count as a different answer


def forward(eng, ids, f32):
    ids = mx.array(ids, dtype=mx.int32)[None]
    if f32:
        emb = eng.lm.model.embed_tokens(ids).astype(mx.float32)
        out = eng.lm(ids, inputs_embeds=emb, cache=eng.cache)
    else:
        out = eng.lm(ids, cache=eng.cache)
    return getattr(out, "logits", out)


def logprobs(logits):
    x = logits.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def batch_path(eng, prompt, toks, f32):
    """Log-probs [steps, vocab] for the token after the prompt and after each decoded token but the last."""
    eng.new_cache()
    logits = forward(eng, prompt + toks[:-1], f32)
    lp = logprobs(logits[0, len(prompt) - 1 :, :])
    mx.eval(lp)
    return lp, str(logits.dtype)


def step_path(eng, prompt, toks, f32, truth):
    """Per step: KL(truth || path), the path's argmax and its top-1 log-prob."""
    eng.new_cache()
    logits = forward(eng, prompt, f32)[0, -1, :]
    rows = []
    for j in range(len(toks)):
        lp = logprobs(logits)
        t = truth[j]
        kl = mx.sum(mx.exp(t) * (t - lp))
        am = mx.argmax(lp)
        mx.eval(kl, am, eng.cache_arrays())
        rows.append((kl.item(), am.item()))
        if j + 1 < len(toks):
            logits = forward(eng, [toks[j]], f32)[0, -1, :]
    return rows


def restricted_kl(truth_row, top):
    """KL(truth || served) over the served top-k, both renormalised on it."""
    ids = [t for t, _ in top]
    t = [truth_row[i] for i in ids]
    s = [lp for _, lp in top]
    tz = math.log(sum(math.exp(a) for a in t))
    sz = math.log(sum(math.exp(a) for a in s))
    return sum(math.exp(a - tz) * ((a - tz) - (b - sz)) for a, b in zip(t, s))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(rows, name):
    kl = [r[f"kl_{name}"] for r in rows]
    wrong = [r for r in rows if r[f"am_{name}"] != r["truth"]]
    return {
        "steps": len(rows),
        "kl_mean": statistics.mean(kl),
        "kl_median": statistics.median(kl),
        "kl_p99": pct(kl, 0.99),
        "kl_max": max(kl),
        "far_steps_per_1000": 1000 * sum(1 for k in kl if k > FAR) / len(kl),
        "argmax_differs_per_1000": 1000 * len(wrong) / len(rows),
        # the path's choice has under 1 % of the yardstick's mass: another answer, not a near tie
        "argmax_under_1pct_per_1000": 1000 * sum(1 for r in wrong if r[f"plp_{name}"] < math.log(0.01)) / len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref-dump", required=True)
    ap.add_argument("--docs", type=int, default=0, help="0 = every document of the dump")
    ap.add_argument("--paths", default="batch16,step16,step32", help="paths to score against batch32")
    ap.add_argument("--fuse", default="", help="comma list of router, front, linears (patches 030, 031, 032)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dump = json.load(open(a.ref_dump))
    docs = dump["docs"][: a.docs] if a.docs else dump["docs"]
    eng = census.Engine(census.resolve_model_dir(a.model))
    names = [n for n in a.paths.split(",") if n]
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
    print(f"fusions installed: {fused}", flush=True)
    out = {"model": a.model, "ref_dump": os.path.basename(a.ref_dump), "far_kl": FAR, "fused": fused,
           "paths": names, "truth_top1_logprob_sum": 0.0, "docs": []}
    t0 = time.time()
    for di, d in enumerate(docs):
        prompt, steps = d["ids"], d["steps"]
        toks = [s["tid"] for s in steps]
        truth, dt32 = batch_path(eng, prompt, toks, True)
        out.setdefault("logit_dtypes", {"batch32": dt32})
        am_t = mx.argmax(truth, axis=-1)
        srt = mx.sort(truth, axis=-1)[:, -2:]
        mx.eval(am_t, srt)
        out["truth_top1_logprob_sum"] += mx.sum(srt[:, 1]).item()
        paths = {}
        if "batch16" in names:
            b16, dt16 = batch_path(eng, prompt, toks, False)
            out["logit_dtypes"]["batch16"] = dt16
            kl_b16 = mx.sum(mx.exp(truth) * (truth - b16), axis=-1)
            am_b16 = mx.argmax(b16, axis=-1)
            mx.eval(kl_b16, am_b16)
            del b16
            paths["batch16"] = list(zip(kl_b16.tolist(), am_b16.tolist()))
        if "step16" in names:
            paths["step16"] = step_path(eng, prompt, toks, False, truth)
        if "step32" in names:
            paths["step32"] = step_path(eng, prompt, toks, True, truth)
        rows = []
        for j, s in enumerate(steps):
            trow = truth[j]
            pick = [s["tid"]] + [paths[n][j][1] for n in names] + [t for t, _ in s["top"]]
            vals = trow[mx.array(pick)].tolist()
            row = {
                "s": j,
                "truth": am_t[j].item(),
                "truth_margin": (srt[j, 1] - srt[j, 0]).item(),
                "served": s["tid"],
                "am_served": s["tid"],
                "plp_served": vals[0],
                "kl_served": restricted_kl(dict(zip(pick[1 + len(names):], vals[1 + len(names):])), s["top"]),
            }
            for k, n in enumerate(names):
                row[f"kl_{n}"], row[f"am_{n}"] = paths[n][j]
                row[f"plp_{n}"] = vals[1 + k]
            rows.append(row)
        out["docs"].append({"domain": d["domain"], "rows": rows})
        far = {n: sum(1 for r in rows if r[f"kl_{n}"] > FAR) for n in names + ["served"]}
        print(f"doc {di} {d['domain']:5s} {len(rows)} steps, far steps {far}, {time.time() - t0:.0f}s elapsed", flush=True)
        mx.clear_cache()
    allrows = [r for d in out["docs"] for r in d["rows"]]
    out["summary"] = {n: summarize(allrows, n) for n in names + ["served"]}
    json.dump(out, open(a.out, "w"))
    print(json.dumps({"logit_dtypes": out["logit_dtypes"], "fused": fused,
                      "truth_top1_logprob_sum": out["truth_top1_logprob_sum"], **out["summary"]}, indent=1))


if __name__ == "__main__":
    main()
