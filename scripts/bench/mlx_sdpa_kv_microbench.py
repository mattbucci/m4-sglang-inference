#!/usr/bin/env python3
"""Kernel microbench: one decode query against a deep KV cache, bf16 K/V
against int8 K/V with one float32 scale per head-token row (raw MLX,
synthetic tensors, no model).

Decode attention at depth reads every cached key and value once per step, so
the question is whether the step is bound by those bytes. Arms:

  core_bf16   mx.fast.scaled_dot_product_attention (what serving calls)
  core_b<N>   the same call with MLX_SDPA_BLOCKS=<N> set for that evaluation;
              MLX reads the variable on every kernel call, so one process can
              sweep the first pass's block count (its own ladder stops at 1024)
  vlm_bf16    mlx-vlm's two-pass Metal kernel for Qwen3.5/3.6
  own_bf16    this file's two-pass kernel on bf16 K/V (control for the next two)
  i8k         int8 keys, bf16 values
  i8kv        int8 keys and int8 values

Each arm evaluates LAYERS independent layers with distinct tensors, the way a
decode step touches every attention layer's cache once. Speed only: the int8
arms are checked against a float32 reference on the same dequantized tensors
(kernel correctness), and what the quantization costs in model output is
measured by mlx_kv_int8_fidelity.py.

Shapes follow Qwen3.6-35B-A3B: 16 query heads, 2 KV heads, head dim 256.

usage: mlx_sdpa_kv_microbench.py <depths csv> <out jsonl> [arms csv]
"""
import json, os, statistics, sys, time
import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as lang

depths, out_path = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
arms = (sys.argv[3] if len(sys.argv) > 3 else "core_bf16,vlm_bf16,own_bf16,i8k,i8kv").split(",")
QH, KVH, D = 16, 2, 256
GQA = QH // KVH
LAYERS = int(os.environ.get("LAYERS", "10"))
REPS = int(os.environ.get("REPS", "7"))
SCALE = D ** -0.5
BF = mx.bfloat16


def pass1_source(k_int8, v_int8):
    kt, vt = ("int8_t" if k_int8 else "T"), ("int8_t" if v_int8 else "T")
    ksp = "const device float* ksp = kscale + int(kv_head_idx) * K_SIZE + int(block_idx);" if k_int8 else ""
    vsp = "const device float* vsp = vscale + int(kv_head_idx) * K_SIZE + int(block_idx);" if v_int8 else ""
    adv = (" ksp += BLOCKS;" if k_int8 else "") + (" vsp += BLOCKS;" if v_int8 else "")
    return f"""
    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint block_idx = threadgroup_position_in_grid.z;
    uint gqa_idx = thread_position_in_threadgroup.y;

    constexpr int BD = 32;
    constexpr int per_thread = D_SIZE / BD;

    typedef float U;
    thread U q[per_thread];
    thread U o[per_thread] = {{0}};

    int K_SIZE = int(k_size[0]);
    int q_head_idx = int(GQA_FACTOR * kv_head_idx + gqa_idx);

    const device T* qptr = queries + q_head_idx * D_SIZE + int(simd_lid) * per_thread;
    const device {kt}* kptr = keys + int(kv_head_idx) * K_SIZE * D_SIZE
        + int(block_idx) * D_SIZE + int(simd_lid) * per_thread;
    const device {vt}* vptr = values + int(kv_head_idx) * K_SIZE * D_SIZE
        + int(block_idx) * D_SIZE + int(simd_lid) * per_thread;
    {ksp}
    {vsp}
    device T* optr = partials + q_head_idx * BLOCKS * D_SIZE
        + int(block_idx) * D_SIZE + int(simd_lid) * per_thread;
    device float* sump = sums + q_head_idx * BLOCKS + int(block_idx);
    device float* maxp = maxs + q_head_idx * BLOCKS + int(block_idx);

    U s = U(scale[0]);
    for (int i = 0; i < per_thread; i++) {{
        q[i] = s * U(qptr[i]);
    }}

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(block_idx); i < K_SIZE; i += BLOCKS) {{
        U score = 0;
        for (int j = 0; j < per_thread; j++) {{
            score += q[j] * U(kptr[j]);
        }}
        score = simd_sum(score){" * ksp[0]" if k_int8 else ""};

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        U ev = exp_score{" * vsp[0]" if v_int8 else ""};
        for (int j = 0; j < per_thread; j++) {{
            o[j] = o[j] * factor + ev * U(vptr[j]);
        }}

        kptr += BLOCKS * D_SIZE;
        vptr += BLOCKS * D_SIZE;{adv}
    }}

    if (simd_lid == 0) {{
        sump[0] = sum_exp_score;
        maxp[0] = max_score;
    }}
    for (int i = 0; i < per_thread; i++) {{
        optr[i] = static_cast<T>(o[i]);
    }}
"""


_K1 = {}


def own_sdpa(q, k, v, ks=None, vs=None, blocks=None):
    n = k.shape[2]
    blocks = blocks or lang._qwen3_5_sdpa_vector_blocks(n, GQA)
    key = (ks is not None, vs is not None, blocks)
    if key not in _K1:
        names = ["queries", "keys", "values"] + (["kscale"] if key[0] else []) + (["vscale"] if key[1] else [])
        _K1[key] = mx.fast.metal_kernel(
            name=f"sdpa_kv_bench_k{int(key[0])}_v{int(key[1])}_b{blocks}",
            input_names=names + ["scale", "k_size"],
            output_names=["partials", "sums", "maxs"],
            header="#include <metal_simdgroup>\nusing namespace metal;\n",
            source=pass1_source(key[0], key[1]))
    scale_a, ksize_a = lang._qwen3_5_cached_sdpa_scalars(float(SCALE), int(n))
    ins = [q, k, v] + ([ks] if ks is not None else []) + ([vs] if vs is not None else []) + [scale_a, ksize_a]
    partials, sums, maxs = _K1[key](
        inputs=ins,
        template=[("T", BF), ("D_SIZE", D), ("GQA_FACTOR", GQA), ("BLOCKS", int(blocks))],
        grid=(32 * KVH, GQA, blocks), threadgroup=(32, GQA, 1),
        output_shapes=[(1, QH, 1, blocks, D), (1, QH, 1, blocks), (1, QH, 1, blocks)],
        output_dtypes=[BF, mx.float32, mx.float32])
    k2 = lang._qwen3_5_ragged_sdpa_two_pass_2_kernel(BF, D, blocks)
    return k2(inputs=[partials, sums, maxs],
              template=[("T", BF), ("D_SIZE", D), ("BLOCKS", int(blocks))],
              grid=(1024, QH, 1), threadgroup=(1024, 1, 1),
              output_shapes=[(1, QH, 1, D)], output_dtypes=[BF])[0]


def quant8(x):
    f = x.astype(mx.float32)
    s = mx.maximum(mx.abs(f).max(axis=-1), 1e-20) / 127.0
    q8 = mx.clip(mx.round(f / s[..., None]), -127, 127).astype(mx.int8)
    return q8, s[0]


def make_layer(n):
    q = (mx.random.normal((1, QH, 1, D)) * 1.5).astype(BF)
    k = mx.random.normal((1, KVH, n, D)).astype(BF)
    v = mx.random.normal((1, KVH, n, D)).astype(BF)
    mx.eval(q, k, v)
    k8, ks = quant8(k)
    v8, vs = quant8(v)
    mx.eval(k8, ks, v8, vs)
    return {"q": q, "k": k, "v": v, "k8": k8, "ks": ks, "v8": v8, "vs": vs}


def blocks_of(arm):
    return int(arm[6:]) if arm.startswith("core_b") and arm[6:].isdigit() else None


_ENV_BLOCKS = os.environ.get("MLX_SDPA_BLOCKS")


def evaluate(arm, outs):
    """mx.eval under the arm's block count; the caller's setting otherwise."""
    want = str(blocks_of(arm)) if blocks_of(arm) else _ENV_BLOCKS
    if want is None:
        os.environ.pop("MLX_SDPA_BLOCKS", None)
    else:
        os.environ["MLX_SDPA_BLOCKS"] = want
    mx.eval(outs)


def arm_fn(arm):
    return ARM["core_bf16" if blocks_of(arm) else arm]


ARM = {
    "core_bf16": lambda L: mx.fast.scaled_dot_product_attention(L["q"], L["k"], L["v"], scale=SCALE, mask=None),
    "vlm_bf16": lambda L: lang._qwen3_5_ragged_decode_attention(L["q"], L["k"], L["v"], [0], SCALE),
    "own_bf16": lambda L: own_sdpa(L["q"], L["k"], L["v"]),
    "i8k": lambda L: own_sdpa(L["q"], L["k8"], L["v"], ks=L["ks"]),
    "i8kv": lambda L: own_sdpa(L["q"], L["k8"], L["v8"], ks=L["ks"], vs=L["vs"]),
}
BYTES = {  # K/V bytes one layer reads, per cached token
    "core_bf16": KVH * D * 4, "vlm_bf16": KVH * D * 4, "own_bf16": KVH * D * 4,
    "i8k": KVH * (D * 3 + 4), "i8kv": KVH * (D * 2 + 8),
}


def rel_err(a, ref):
    a, ref = a.astype(mx.float32), ref.astype(mx.float32)
    return float((mx.sqrt(((a - ref) ** 2).sum()) / mx.sqrt((ref ** 2).sum())).item())


def reference(L, k_int8, v_int8):
    kf = (L["k8"].astype(mx.float32) * L["ks"][None, ..., None]) if k_int8 else L["k"].astype(mx.float32)
    vf = (L["v8"].astype(mx.float32) * L["vs"][None, ..., None]) if v_int8 else L["v"].astype(mx.float32)
    return mx.fast.scaled_dot_product_attention(L["q"].astype(mx.float32), kf, vf, scale=SCALE, mask=None)


mx.random.seed(11)
env = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
for n in depths:
    layers = [make_layer(n) for _ in range(LAYERS)]
    blocks = lang._qwen3_5_sdpa_vector_blocks(n, GQA)
    check = {}
    for arm in arms:
        got = arm_fn(arm)(layers[0])
        ref = reference(layers[0], arm in ("i8k", "i8kv"), arm == "i8kv")
        evaluate(arm, [got])
        evaluate("reference", [ref])
        check[arm] = rel_err(got, ref)
    times = {a: [] for a in arms}
    for arm in arms:  # compile and touch every tensor once
        evaluate(arm, [arm_fn(arm)(L) for L in layers])
    for rep in range(REPS):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            outs = [arm_fn(arm)(L) for L in layers]
            t0 = time.perf_counter()
            evaluate(arm, outs)
            times[arm].append((time.perf_counter() - t0) * 1e3)
    base = statistics.median(times[arms[0]])
    for arm in arms:
        ms = statistics.median(times[arm])
        gb = BYTES.get(arm, BYTES["core_bf16"]) * n * LAYERS / 1e9
        row = {"arm": arm, "depth": n, "layers": LAYERS, "blocks": int(blocks_of(arm) or blocks),
               "ms_all_layers": round(ms, 3), "ms_min": round(min(times[arm]), 3),
               "kv_gb_read": round(gb, 3), "gb_per_s": round(gb / (ms / 1e3), 1),
               "speedup_vs_first_arm": round(base / ms, 3),
               "rel_err_vs_f32_reference": round(check[arm], 5), "env": env}
        print(json.dumps(row), flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    del layers
    mx.clear_cache()
