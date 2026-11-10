#!/usr/bin/env python3
"""Raw-MLX fidelity of an int8 KV cache, simulated: after a real-text prefill
the cached keys and values of every attention layer are rounded to int8 and
written back as bf16, then the same continuation is teacher-forced and each
step's distribution is compared with the bf16 run.

The prompt is this repository's SGLang source (a coding-agent context), not
the census word salad, so the attention patterns are the ones a deep coding
turn produces. Tokens appended during the continuation keep bf16 K/V, the
shape a real int8 cache would have with a bf16 tail.

Arms:
  again     bf16 a second time (the noise floor of the comparison)
  kv_row    int8 keys and values, one scale per head-token row of 256
  k_row     int8 keys only
  v_row     int8 values only
  kv_g64    int8 keys and values, one scale per 64 dims
  kv_g32    int8 keys and values, one scale per 32 dims

usage: mlx_kv_int8_fidelity.py <depths csv> <out jsonl> [steps] [arms csv]
"""
import glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx_decode_census as c

depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 128
arms = (sys.argv[4] if len(sys.argv) > 4 else "again,kv_row,k_row,v_row,kv_g64,kv_g32").split(",")
model = os.environ.get("MODEL", "mlx-community/Qwen3.6-35B-A3B-4bit")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.environ.get("CORPUS", os.path.join(REPO, "components/sglang/python/sglang/srt"))
SPEC = {"kv_row": (256, 256), "k_row": (256, 0), "v_row": (0, 256), "kv_g64": (64, 64), "kv_g32": (32, 32)}
mx.set_cache_limit(4 * 2**30)
d = c.resolve_model_dir(model)
eng = c.Engine(d)


def corpus_ids(n):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    ids = []
    for path in sorted(glob.glob(os.path.join(CORPUS, "**", "*.py"), recursive=True)):
        text = f"# file: {os.path.relpath(path, CORPUS)}\n" + open(path, errors="ignore").read() + "\n"
        ids.extend(tok.encode(text, add_special_tokens=False).ids)
        if len(ids) >= n:
            return ids[:n]
    raise SystemExit(f"corpus under {CORPUS} has only {len(ids)} tokens, need {n}")


def fake_int8(x, group):
    f = x.astype(mx.float32)
    sh = f.shape
    g = f.reshape(*sh[:-1], sh[-1] // group, group)
    s = mx.maximum(mx.abs(g).max(axis=-1, keepdims=True), 1e-20) / 127.0
    return (mx.clip(mx.round(g / s), -127, 127) * s).reshape(sh).astype(x.dtype)


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.sqrt(((a - b) ** 2).sum()) / mx.sqrt((b ** 2).sum())).item())


def quantize_cache(kg, vg):
    errs = []
    for cc in eng.cache:
        if getattr(cc, "keys", None) is None or not hasattr(cc, "values"):
            continue
        k, v = cc.keys[..., : cc.offset, :], cc.values[..., : cc.offset, :]
        kq = fake_int8(k, kg) if kg else k
        vq = fake_int8(v, vg) if vg else v
        mx.eval(kq, vq)
        errs.append((rel(kq, k) if kg else 0.0, rel(vq, v) if vg else 0.0))
        cc.keys, cc.values = kq, vq
    return errs


def run(ids, depth, arm):
    eng.prefill(ids[:depth], 2048)
    errs = quantize_cache(*SPEC[arm]) if arm in SPEC else []
    lps = []
    for t in range(steps):
        lg = eng.logits(mx.array([[ids[depth + t]]], dtype=mx.int32))[:, -1, :].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp)
        lps.append(lp[0])
    return mx.stack(lps), errs


for depth in depths:
    ids = corpus_ids(depth + steps + 1)
    target = mx.array(ids[depth + 1 : depth + steps + 1])
    base, _ = run(ids, depth, "base")
    pb = mx.exp(base)
    base_nll = float((-mx.take_along_axis(base, target[:, None], axis=-1)).mean().item())
    base_top = mx.argmax(base, axis=-1)
    for arm in arms:
        lp, errs = run(ids, depth, arm)
        kl = (pb * (base - lp)).sum(axis=-1)
        mx.eval(kl)
        kls = sorted(kl.tolist())
        nll = float((-mx.take_along_axis(lp, target[:, None], axis=-1)).mean().item())
        row = {"arm": arm, "depth": depth, "steps": steps,
               "kl_mean": round(statistics.fmean(kls), 6), "kl_p95": round(kls[int(0.95 * (len(kls) - 1))], 6),
               "kl_max": round(kls[-1], 6),
               "top1_agree": round(float((mx.argmax(lp, axis=-1) == base_top).mean().item()), 4),
               "nll_base": round(base_nll, 5), "nll_arm": round(nll, 5), "nll_delta": round(nll - base_nll, 5),
               "key_rel_err": round(statistics.fmean(e[0] for e in errs), 5) if errs else None,
               "value_rel_err": round(statistics.fmean(e[1] for e in errs), 5) if errs else None}
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
