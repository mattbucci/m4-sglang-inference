#!/usr/bin/env python3
"""Kernel microbench at decode shapes (raw MLX, synthetic tensors, no model).

What one GPU launch costs when it reads the previous launch's output (chain)
and when it does not (fan), and what byte rate the quantized matmuls reach
when every launch reads weights the GPU has not touched recently.

Run it with both command-buffer limits very large, so MLX encodes the whole
graph before the GPU starts:

    MLX_MAX_OPS_PER_BUFFER=1000000 MLX_MAX_MB_PER_BUFFER=1000000 \
        python scripts/bench/mlx_launch_microbench.py out.jsonl

`mx.async_eval` then returns after pure CPU encode and the wait that follows
is pure GPU time. Each case runs at n and 2n launches; the per-launch cost is
the difference divided by n, which drops the fixed commit and wake-up cost.

Shapes follow Qwen3.6-35B-A3B: hidden 2048, 256 experts of 512, top-8, 4-bit
group 64, bf16 activations.
"""
import json, os, sys, time
import mlx.core as mx

out_path = sys.argv[1] if len(sys.argv) > 1 else None
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
H, E, K, R = 2048, 512, 8, 256
DT = mx.bfloat16
mx.random.seed(7)
x0 = mx.random.normal((1, 1, H)).astype(DT)
w0 = mx.ones((H,), dtype=DT)
consts = [mx.random.normal((1, 1, H)).astype(DT) * 0.01 for _ in range(8)]
mx.eval(x0, w0, consts)


def quantized(shape, scale):
    w = mx.random.normal(shape) * scale
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    q = (wq, scales.astype(DT), biases.astype(DT))  # the checkpoint stores both as 16-bit
    mx.eval(q)
    return q


def nbytes(arrs):
    return sum(a.nbytes for a in arrs)


CASES = {}


def case(name, launches=1, fan=False):
    def reg(fn):
        CASES[name] = (fn, launches, fan)
        return fn
    return reg


@case("add chain")
def _(n):
    x = x0
    for i in range(n):
        x = x + consts[i % 8]
    return [x], 0


@case("add fan", fan=True)
def _(n):
    return [x0 + consts[i % 8] * (1 + i) for i in range(n)], 0


@case("multiply-add fan (2 launches)", launches=2, fan=True)
def _(n):
    return [x0 * consts[i % 8] + consts[(i + 1) % 8] for i in range(n)], 0


@case("rms_norm chain")
def _(n):
    x = x0
    for _ in range(n):
        x = mx.fast.rms_norm(x, w0, 1e-6)
    return [x], 0


@case("add + rms_norm chain (2 launches)", launches=2)
def _(n):
    x = x0
    for i in range(n):
        x = mx.fast.rms_norm(x + consts[i % 8], w0, 1e-6)
    return [x], 0


@case("sigmoid * x + y chain (3 launches)", launches=3)
def _(n):
    x = x0
    for i in range(n):
        x = mx.sigmoid(x) * consts[i % 8] + consts[(i + 1) % 8]
    return [x], 0


g0 = mx.random.normal((1, 1, R)).astype(DT)
mx.eval(g0)


@case("softmax 256 chain")
def _(n):
    g = g0
    for _ in range(n):
        g = mx.softmax(g, axis=-1, precise=True)
    return [g], 0


@case("partition 256 chain (the argpartition sort kernel)")
def _(n):
    g = g0
    for _ in range(n):
        g = mx.partition(g, kth=-K, axis=-1)
    return [g], 0


@case("argpartition 256 fan", fan=True)
def _(n):
    return [mx.argpartition(g0 + (i % 8), kth=-K, axis=-1)[..., -K:] for i in range(n)], 0


@case("router tail chain: softmax, argpartition, gather, sum, divide, concatenate (6 launches)", launches=6)
def _(n):
    g = g0
    for _ in range(n):
        p = mx.softmax(g, axis=-1, precise=True)
        i = mx.argpartition(p, kth=-K, axis=-1)[..., -K:]
        s = mx.take_along_axis(p, i, axis=-1)
        s = s / s.sum(axis=-1, keepdims=True)
        g = mx.concatenate([s] * (R // K), axis=-1)
    return [g], 0


# Dense 4-bit linears, 2048 -> 2048, distinct weights per launch (2.25 MB each).
N_DENSE = int(os.environ.get("N_DENSE", "384"))
dense = None


def dense_pool():
    global dense
    if dense is None:
        dense = [quantized((H, H), H ** -0.5) for _ in range(N_DENSE)]
    return dense


@case("qmm 2048x2048 chain, distinct weights")
def _(n):
    pool = dense_pool()
    x = x0
    used = 0
    for i in range(n):
        w, s, b = pool[i % len(pool)]
        x = mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True, group_size=64, bits=4)
        used += nbytes((w, s, b))
    return [x], used


@case("qmm 2048x2048 fan, distinct weights", fan=True)
def _(n):
    pool = dense_pool()
    outs, used = [], 0
    for i in range(n):
        w, s, b = pool[i % len(pool)]
        outs.append(mx.quantized_matmul(x0, w, scales=s, biases=b, transpose=True, group_size=64, bits=4))
        used += nbytes((w, s, b))
    return outs, used


# Routed experts: stacked (256, 512, 2048) and (256, 2048, 512) tensors, top-8.
N_STACK = int(os.environ.get("N_STACK", "4"))
stacks = None


def stack_pool():
    global stacks
    if stacks is None:
        stacks = [(quantized((R, E, H), H ** -0.5), quantized((R, E, H), H ** -0.5),
                   quantized((R, H, E), E ** -0.5)) for _ in range(N_STACK)]
    return stacks


def pick(i):
    idx = mx.array([[[(37 * i + 29 * j) % R for j in range(K)]]], dtype=mx.uint32)
    return idx


@case("routed experts chain: 3 gather_qmm, 3 activation, sum, scale (8 launches)", launches=8)
def _(n):
    pool = stack_pool()
    idxs = [pick(i) for i in range(64)]
    mx.eval(idxs)
    x = x0
    used = 0
    for i in range(n):
        (uw, us, ub), (gw, gs, gb), (dw, ds, db) = pool[i % len(pool)]
        idx = idxs[i % 64]
        h = mx.expand_dims(x, (-2, -3))
        up = mx.gather_qmm(h, uw, us, ub, rhs_indices=idx, transpose=True, group_size=64, bits=4)
        gt = mx.gather_qmm(h, gw, gs, gb, rhs_indices=idx, transpose=True, group_size=64, bits=4)
        y = mx.gather_qmm(mx.sigmoid(gt) * gt * up, dw, ds, db, rhs_indices=idx, transpose=True,
                          group_size=64, bits=4)
        x = y.squeeze(-2).sum(axis=-2) * 0.125
        used += K * (nbytes((uw, us, ub)) + nbytes((gw, gs, gb)) + nbytes((dw, ds, db))) // R
    return [x], used


# The token-selection tail over the 248,320-wide vocabulary. Each iteration
# feeds its token back into the next logits, so the tails run in sequence the
# way decode steps do; read these rows per iteration, not per launch.
V, TOPK, BLK = 248320, 20, 512
lg0 = None


def logits_after(tok):
    global lg0
    if lg0 is None:
        lg0 = (mx.random.normal((1, V)) * 3).astype(DT)
        mx.eval(lg0)
    if tok is None:
        return lg0
    return lg0 + (tok.astype(mx.float32) * 1e-12).astype(DT)[:, None]


def filter_and_pick(p_sort, cand):
    cum = mx.cumsum(p_sort, axis=-1)
    logw = mx.log(mx.where((cum - p_sort) > 0.8, 0.0, p_sort))
    pick = mx.argmax(logw + mx.random.gumbel(logw.shape), axis=-1)
    return mx.take_along_axis(cand, pick[:, None], axis=-1).squeeze(-1)


@case("token tail per iteration: greedy argmax over 248,320")
def _(n):
    tok = None
    for _ in range(n):
        tok = mx.argmax(logits_after(tok).astype(mx.float32), axis=-1)
    return [tok], 0


@case("token tail per iteration: sampled, full argsort over 248,320 (the serving chain)")
def _(n):
    tok = None
    for _ in range(n):
        probs = mx.softmax(logits_after(tok).astype(mx.float32) / 0.7, axis=-1)
        idx = mx.argsort(-probs, axis=-1)[:, :TOPK]
        tok = filter_and_pick(mx.take_along_axis(probs, idx, axis=-1), idx)
    return [tok], 0


@case("token tail per iteration: sampled, top-20 from the 20 best blocks of 512 (same candidates)")
def _(n):
    tok = None
    for _ in range(n):
        probs = mx.softmax(logits_after(tok).astype(mx.float32) / 0.7, axis=-1)
        blocks = probs.reshape(1, V // BLK, BLK)
        bidx = mx.argsort(-blocks.max(axis=-1), axis=-1)[:, :TOPK]
        flat = mx.take_along_axis(blocks, bidx[..., None], axis=1).reshape(1, TOPK * BLK)
        loc = mx.argsort(-flat, axis=-1)[:, :TOPK]
        idx = mx.take_along_axis(bidx, loc // BLK, axis=-1) * BLK + loc % BLK
        tok = filter_and_pick(mx.take_along_axis(flat, loc, axis=-1), idx)
    return [tok], 0


def measure(fn, n):
    best = None
    for _ in range(4):
        outs, used = fn(n)
        t0 = time.perf_counter()
        mx.async_eval(outs)
        t1 = time.perf_counter()
        mx.eval(outs)
        t2 = time.perf_counter()
        r = (t1 - t0, t2 - t1, used)
        if best is None or r[0] + r[1] < best[0] + best[1]:
            best = r
    return best


single = (int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "50")) >= 100000
          and int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "50")) >= 100000)
print(f"# single-buffer mode: {single} (encode and gpu columns are separable only when True)")
print("| case | launches | encode us/launch | gpu us/launch | weights MB/iter | weight GB/s |")
print("|---|---|---|---|---|---|")
for name, (fn, launches, fan) in CASES.items():
    if only and not any(o in name for o in only):
        continue
    n = int(os.environ.get("N", "512"))
    if "distinct" in name:
        n = min(n, N_DENSE // 2)
    e1, g1, u1 = measure(fn, n)
    e2, g2, u2 = measure(fn, 2 * n)
    enc = (e2 - e1) / n / launches * 1e6
    gpu = (g2 - g1) / n / launches * 1e6
    mb = (u2 - u1) / n / 2**20
    rate = (u2 - u1) / (g2 - g1) / 1e9 if mb and g2 > g1 else None
    row = {"case": name, "launches_per_iter": launches, "n": n, "single_buffer": single,
           "encode_us_per_launch": round(enc, 2), "gpu_us_per_launch": round(gpu, 2),
           "weight_mb_per_iter": round(mb, 2), "weight_gb_per_s": rate and round(rate, 1),
           "raw": {"n": [round(e1 * 1e3, 3), round(g1 * 1e3, 3)], "2n": [round(e2 * 1e3, 3), round(g2 * 1e3, 3)]},
           "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")}}
    print(f"| {name} | {launches} | {enc:.2f} | {gpu:.2f} | {mb:.2f} | {rate and round(rate, 1)} |", flush=True)
    if out_path:
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
