#!/usr/bin/env python3
"""Raw-MLX A/B inside one process: where the MoE block's time goes beyond the
routed-expert weights, and what the GatedDeltaNet block's small launches cost
(Qwen3.5/3.6 MoE hybrids on mlx-vlm). Every arm
re-prefills, takes 64 sync steps that double as the greedy identity sample
against the base arm, then three 96-step pipelined segments.

Arms (combine with "+"):
  base            stock block
  lhs             gather_qmm gets a cached constant lhs_indices instead of the
                  arange MLX builds per call (same values: bit-exact)
  route_const     expert indices and scores frozen per layer: removes router
                  linear, softmax, argpartition, gather, sum, divide (ceiling)
  inds_const      indices frozen, scores computed (argpartition removed)
  scores_const    scores frozen, indices computed (gather, sum, divide removed)
  argpart_logits  argpartition reads the router logits, so the softmax leaves
                  the path the experts wait on (ties can pick another expert)
  sm8             argpartition on logits, softmax over the eight selected
                  logits (no 256-wide softmax, sum or divide; not bit-exact)
  fused           one Metal launch after the router linear: every expert ranks
                  itself, the top k write index and renormalised score (float32
                  maths, one rounding to the model dtype; not bit-exact)
  comb            one Metal launch for the block's tail: score-weighted sum of
                  the routed experts plus sigmoid(gate) * shared expert (stock:
                  multiply, sum, sigmoid, multiply, add; not bit-exact)
  fused+comb      both kernels
  lin4            the four linears that read the block input (router, shared
                  gate_proj and up_proj, shared-expert gate) as one quantized
                  matmul over concatenated weights (mlx-vlm's own helper). It
                  needs equal bits on all four: the 4-bit qwen36 checkpoint
                  keeps both gates at 8 bits, so there the arm is the stock
                  path (`fused_linear_calls` stays 0)
  pair            the same four linears as two quantized matmuls, grouped by
                  bit width (router + shared-expert gate, gate_proj + up_proj)
  front           GatedDeltaNet: one Metal launch for the conv-state shift, the
                  depthwise conv, silu and the q/k RMS norms with their scales
                  (stock: concatenate, contiguous, multiply, sum, astype, silu,
                  two rms_norm, two multiplies; float32 maths, one rounding)
  back            GatedDeltaNet: g and beta computed inside the delta kernel,
                  and one launch for the gated output norm (stock: compiled g,
                  sigmoid, delta kernel, rms_norm, compiled swiglu)

The frozen arms change the routing, so their text differs by construction;
they are ceilings, not candidates.

usage: mlx_decode_router_probe.py <depths csv> <out jsonl> [arms csv]
"""
import json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx_decode_census as c
from mlx_vlm.models import switch_layers as sl
from mlx_vlm.models.qwen3_5 import gated_delta as gd
from mlx_vlm.models.qwen3_5 import language as lang

depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
order = (sys.argv[3] if len(sys.argv) > 3 else
         "base,lhs,route_const,inds_const,scores_const,argpart_logits,sm8,sm8,argpart_logits,"
         "scores_const,inds_const,route_const,lhs,base").split(",")
model = os.environ.get("MODEL", "mlx-community/Qwen3.6-35B-A3B-4bit")
mx.set_cache_limit(4 * 2**30)
d = c.resolve_model_dir(model)
eng = c.Engine(d)
MODE = set()
CALLS = {"delta": 0, "lin": 0}
_FUSED_LINEARS = {}


def fused_linears(linears, x):
    """One quantized matmul over the concatenated weights of linears that share
    bits, group size and mode; None when they do not."""
    import mlx.nn as nn

    first = linears[0]
    if not all(isinstance(l, nn.QuantizedLinear) and l.bits == first.bits
               and l.group_size == first.group_size and l.mode == first.mode
               and l.biases is not None and "bias" not in l for l in linears):
        return None
    cached = _FUSED_LINEARS.get(id(first))
    if cached is None:
        w = mx.concatenate([l.weight for l in linears], axis=0)
        sc = mx.concatenate([l.scales for l in linears], axis=0)
        bi = mx.concatenate([l.biases for l in linears], axis=0)
        splits, off = [], 0
        for l in linears[:-1]:
            off += l.weight.shape[0]
            splits.append(off)
        mx.eval(w, sc, bi)
        cached = _FUSED_LINEARS[id(first)] = (w, sc, bi, splits)
    w, sc, bi, splits = cached
    out = mx.quantized_matmul(x, w, scales=sc, biases=bi, transpose=True,
                              group_size=first.group_size, bits=first.bits, mode=first.mode)
    return mx.split(out, splits, axis=-1)
_LHS = {}
_orig_qsl = sl.QuantizedSwitchLinear.__call__


def _qsl_call(self, x, indices, sorted_indices=False):
    if "lhs" not in MODE or sorted_indices:
        return _orig_qsl(self, x, indices, sorted_indices)
    shape = tuple(x.shape[:-2])
    lhs = _LHS.get(shape)
    if lhs is None:
        lhs = mx.arange(math.prod(shape), dtype=mx.uint32).reshape(shape)
        mx.eval(lhs)
        _LHS[shape] = lhs
    x = mx.gather_qmm(x, self["weight"], self["scales"], self.get("biases"), lhs_indices=lhs,
                      rhs_indices=indices, transpose=True, group_size=self.group_size,
                      bits=self.bits, mode=self.mode, sorted_indices=sorted_indices)
    if "bias" in self:
        x = x + mx.expand_dims(self["bias"][indices], -2)
    return x


sl.QuantizedSwitchLinear.__call__ = _qsl_call

# One threadgroup per row, one thread per expert. A thread counts the experts
# that beat it (ties go to the lower index), so ranks are a permutation; the top
# K_SIZE threads share their exp through threadgroup memory to renormalise.
# Slots are ascending by score, the order argpartition's last-k slice yields.
_ROUTE_SRC = """
    uint e = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    threadgroup float top_exp[K_SIZE];
    uint g = row * E_SIZE;
    float mine = float(gates[g + e]);
    float best = mine;
    uint rank = 0;
    for (uint j = 0; j < E_SIZE; j++) {
        float v = float(gates[g + j]);
        best = max(best, v);
        rank += uint((v > mine) || (v == mine && j < e));
    }
    float ex = 0;
    if (rank < K_SIZE) {
        ex = exp(mine - best);
        top_exp[rank] = ex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rank < K_SIZE) {
        float total = 0;
        for (uint i = 0; i < K_SIZE; i++) {
            total += top_exp[i];
        }
        uint slot = K_SIZE - 1 - rank;
        inds[row * K_SIZE + slot] = e;
        scores[row * K_SIZE + slot] = static_cast<T>(ex / total);
    }
"""
_ROUTE_K = {}


def fused_route(logits, k):
    e = logits.shape[-1]
    rows = math.prod(logits.shape[:-1])
    key = (logits.dtype, e, k)
    if key not in _ROUTE_K:
        _ROUTE_K[key] = mx.fast.metal_kernel(
            name=f"moe_route_e{e}_k{k}", input_names=["gates"], output_names=["inds", "scores"],
            source=_ROUTE_SRC)
    inds, scores = _ROUTE_K[key](
        inputs=[logits], template=[("T", logits.dtype), ("E_SIZE", e), ("K_SIZE", k)],
        grid=(e, rows, 1), threadgroup=(e, 1, 1),
        output_shapes=[(rows, k), (rows, k)], output_dtypes=[mx.uint32, logits.dtype])
    shape = tuple(logits.shape[:-1]) + (k,)
    return inds.reshape(shape), scores.reshape(shape)


_COMB_SRC = """
    uint h = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint y0 = row * K_SIZE * H_SIZE + h;
    uint s0 = row * K_SIZE;
    float acc = 0;
    for (uint k = 0; k < K_SIZE; k++) {
        acc += float(y[y0 + k * H_SIZE]) * float(scores[s0 + k]);
    }
    float g = 1.0f / (1.0f + exp(-float(sgate[row])));
    out[row * H_SIZE + h] = static_cast<T>(acc + g * float(shared[row * H_SIZE + h]));
"""
_COMB_K = {}


def fused_combine(y, scores, shared, sgate):
    k, h = y.shape[-2], y.shape[-1]
    rows = math.prod(y.shape[:-2])
    key = (y.dtype, k, h)
    if key not in _COMB_K:
        _COMB_K[key] = mx.fast.metal_kernel(
            name=f"moe_combine_k{k}_h{h}", input_names=["y", "scores", "shared", "sgate"],
            output_names=["out"], source=_COMB_SRC)
    out = _COMB_K[key](
        inputs=[y, scores, shared, sgate], template=[("T", y.dtype), ("K_SIZE", k), ("H_SIZE", h)],
        grid=(h, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, h)], output_dtypes=[y.dtype])[0]
    return out.reshape(shared.shape)


# GatedDeltaNet front. One threadgroup per head of D_SIZE channels, one thread
# per channel: depthwise conv over the kept state plus the new input, silu, and
# for the q and k heads an RMS norm through threadgroup memory. The conv state
# shifts by one in the same launch.
_FRONT_SRC = """
    uint c = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;
    threadgroup float sq[D_SIZE];
    uint s0 = row * KEEP * C_SIZE + c;
    float acc = 0;
    for (uint j = 0; j < KEEP; j++) {
        T sv = cstate[s0 + j * C_SIZE];
        acc += float(sv) * w[j * C_SIZE + c];
        if (j > 0) {
            new_state[s0 + (j - 1) * C_SIZE] = sv;
        }
    }
    T xv = x[row * C_SIZE + c];
    acc += float(xv) * w[KEEP * C_SIZE + c];
    new_state[s0 + (KEEP - 1) * C_SIZE] = xv;
    float o = acc / (1.0f + exp(-acc));
    sq[t] = o * o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c >= 2 * KD_SIZE) {
        v[row * (C_SIZE - 2 * KD_SIZE) + c - 2 * KD_SIZE] = static_cast<T>(o);
    } else {
        float ss = 0;
        for (uint i = 0; i < D_SIZE; i++) {
            ss += sq[i];
        }
        float r = o * rsqrt(ss / float(D_SIZE) + 1e-6f);
        if (c < KD_SIZE) {
            q[row * KD_SIZE + c] = static_cast<T>(r / float(D_SIZE));
        } else {
            k[row * KD_SIZE + c - KD_SIZE] = static_cast<T>(r * rsqrt(float(D_SIZE)));
        }
    }
"""
_FRONT_K = {}


def delta_front(b, conv_state, mixed):
    rows, keep, cdim = conv_state.shape
    kd, d = b.key_dim, b.head_k_dim
    w = b._qwen3_5_decode_conv_weight[1]
    if "front" not in _FRONT_K:
        _FRONT_K["front"] = mx.fast.metal_kernel(
            name="gdn_front", input_names=["cstate", "x", "w"],
            output_names=["new_state", "q", "k", "v"], source=_FRONT_SRC)
    dt = mixed.dtype
    new_state, q, k, v = _FRONT_K["front"](
        inputs=[conv_state, mixed, w],
        template=[("T", dt), ("C_SIZE", cdim), ("KD_SIZE", kd), ("D_SIZE", d), ("KEEP", keep)],
        grid=(cdim, rows, 1), threadgroup=(d, 1, 1),
        output_shapes=[(rows, keep, cdim), (rows, 1, b.num_k_heads, d), (rows, 1, b.num_k_heads, d),
                       (rows, 1, b.num_v_heads, b.head_v_dim)],
        output_dtypes=[dt, dt, dt, dt])
    return new_state, q, k, v


# The stock single-step delta kernel with g = exp(-exp(A_log) * softplus(a +
# dt_bias)) and beta = sigmoid(b) computed per head inside it.
_DELTA_SRC = """
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    uint q0 = (b_idx * Hk + hk_idx) * Dk;
    uint v0 = (b_idx * Hv + hv_idx) * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    uint st0 = (n * Dv + dv_idx) * Dk;

    float av = float(a[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
    float sp = max(av, 0.0f) + log(1.0f + exp(-abs(av)));
    float gv = exp(-exp(float(a_log[hv_idx])) * sp);
    float bv = 1.0f / (1.0f + exp(-float(bgate[b_idx * Hv + hv_idx])));

    float state[n_per_t];
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(state_in[st0 + s_idx]) * gv;
        kv_mem += state[i] * float(k[q0 + s_idx]);
    }
    kv_mem = simd_sum(kv_mem);
    float delta = (float(v[v0 + dv_idx]) - kv_mem) * bv;
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + float(k[q0 + s_idx]) * delta;
        out += state[i] * float(q[q0 + s_idx]);
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
        y[v0 + dv_idx] = static_cast<InT>(out);
    }
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state_out[st0 + s_idx] = static_cast<StT>(state[i]);
    }
"""
# Gated output norm: RMS norm over a value head times its weight, times
# silu(z), float32 throughout.
_GNORM_SRC = """
    uint c = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;
    threadgroup float sq[D_SIZE];
    float yv = float(y[row * N_SIZE + c]);
    sq[t] = yv * yv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0;
    for (uint i = 0; i < D_SIZE; i++) {
        ss += sq[i];
    }
    float xn = yv * rsqrt(ss / float(D_SIZE) + float(eps[0])) * float(weight[t]);
    float zv = float(z[row * N_SIZE + c]);
    out[row * N_SIZE + c] = static_cast<T>(xn * zv / (1.0f + exp(-zv)));
"""
_BACK_K = {}
_EPS = {}


def delta_back(b, q, k, v, a, bg, z, state):
    rows = q.shape[0]
    hk, dk, hv, dv = b.num_k_heads, b.head_k_dim, b.num_v_heads, b.head_v_dim
    if not _BACK_K:
        _BACK_K["delta"] = mx.fast.metal_kernel(
            name="gdn_delta_gb", input_names=["q", "k", "v", "a", "bgate", "a_log", "dt_bias", "state_in"],
            output_names=["y", "state_out"], source=_DELTA_SRC)
        _BACK_K["norm"] = mx.fast.metal_kernel(
            name="gdn_gated_norm", input_names=["y", "z", "weight", "eps"], output_names=["out"],
            source=_GNORM_SRC)
    y, state = _BACK_K["delta"](
        inputs=[q, k, v, a, bg, b.A_log, b.dt_bias, state],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
        grid=(32, dv, rows * hv), threadgroup=(32, 4, 1),
        output_shapes=[(rows, 1, hv, dv), state.shape], output_dtypes=[q.dtype, state.dtype])
    eps = _EPS.get(b.norm.eps)
    if eps is None:
        eps = _EPS[b.norm.eps] = mx.array([b.norm.eps], dtype=mx.float32)
    out = _BACK_K["norm"](
        inputs=[y, z, b.norm.weight, eps],
        template=[("T", z.dtype), ("N_SIZE", hv * dv), ("D_SIZE", dv)],
        grid=(hv * dv, rows, 1), threadgroup=(dv, 1, 1),
        output_shapes=[(rows, 1, hv * dv)], output_dtypes=[z.dtype])[0]
    return out, state


class DeltaVariant:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False):
        m = MODE & {"front", "back"}
        b = self.inner
        if (not m or inputs.shape[1] != 1 or mask is not None or gdn_sink is not None or target_verify
                or cache is None or cache[0] is None or cache[1] is None
                or getattr(cache, "lengths", None) is not None
                or getattr(b, "_qwen3_5_decode_conv_weight", None) is None):
            return b(inputs, mask, cache, gdn_sink=gdn_sink, target_verify=target_verify)
        CALLS["delta"] += 1
        rows = inputs.shape[0]
        mixed, z, bg, a = lang._decode_quantized_linears_fused(
            (b.in_proj_qkv, b.in_proj_z, b.in_proj_b, b.in_proj_a), inputs)
        if "front" in m:
            cache[0], q, k, v = delta_front(b, cache[0], mixed)
        else:
            conv_input = mx.concatenate([cache[0], mixed], axis=1)
            cache[0] = mx.contiguous(conv_input[:, -(b.conv_kernel_size - 1):, :])
            conv_out = lang.nn.silu(b._causal_conv1d_decode(conv_input))
            q, k, v = [t.reshape(rows, 1, h, d) for t, h, d in zip(
                mx.split(conv_out, [b.key_dim, 2 * b.key_dim], -1),
                [b.num_k_heads, b.num_k_heads, b.num_v_heads],
                [b.head_k_dim, b.head_k_dim, b.head_v_dim])]
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        if "back" in m:
            out, state = delta_back(b, q, k, v, a, bg, z, cache[1])
        else:
            out, state = gd.gated_delta_update(q, k, v, a, bg, b.A_log, b.dt_bias, cache[1], None,
                                               use_kernel=True)
            out = b.norm(out, z.reshape(rows, 1, -1, b.head_v_dim)).reshape(rows, 1, -1)
        cache[1] = state
        if hasattr(cache, "advance"):
            cache.advance(1)
            lang._qwen3_5_advance_left_padding_info(cache, 1)
            lang._qwen3_5_advance_lengths_info(cache, 1)
        return b.out_proj(out)


class MoeVariant:
    def __init__(self, inner):
        self.inner = inner
        self.const = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, x, target_verify=False):
        m = MODE - {"lhs", "front", "back"}
        if not m or target_verify or x.shape[1] != 1:
            return self.inner(x, target_verify)
        b, k = self.inner, self.inner.top_k
        lin4 = None
        if "lin4" in m:
            se = b.shared_expert
            lin4 = lang._decode_quantized_linears_fused(
                (b.gate, se.gate_proj, se.up_proj, b.shared_expert_gate), x)
        elif "pair" in m:
            se = b.shared_expert
            gates2 = fused_linears((b.gate, b.shared_expert_gate), x)
            proj2 = fused_linears((se.gate_proj, se.up_proj), x)
            if gates2 and proj2:
                lin4 = (gates2[0], proj2[0], proj2[1], gates2[1])
        if lin4:
            CALLS["lin"] += 1
        logits = lin4[0] if lin4 else b.gate(x)
        if m & {"route_const", "inds_const", "scores_const"} and self.const is None:
            g = mx.softmax(logits, axis=-1, precise=True)
            i = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
            s = mx.take_along_axis(g, i, axis=-1)
            s = s / s.sum(axis=-1, keepdims=True)
            mx.eval(i, s)
            self.const = (i, s)
        if "route_const" in m:
            inds, scores = self.const
        elif "fused" in m:
            inds, scores = fused_route(logits, k)
        elif "sm8" in m:
            inds = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
            scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1, precise=True)
        else:
            gates = mx.softmax(logits, axis=-1, precise=True)
            if "inds_const" in m:
                inds = self.const[0]
            elif "argpart_logits" in m:
                inds = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
            else:
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            if "scores_const" in m:
                scores = self.const[1]
            else:
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)
        y = b.switch_mlp(x, inds)
        if lin4:
            shared = b.shared_expert.down_proj(lang.swiglu(lin4[1], lin4[2]))
            sgate = lin4[3]
        else:
            shared = b.shared_expert(x, False)
            sgate = b.shared_expert_gate(x)
        if "comb" in m:
            return fused_combine(y, scores, shared, sgate)
        y = (y * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(sgate) * shared


wrapped = []
for layer in eng.layers:
    layer.mlp = MoeVariant(layer.mlp)
    wrapped.append(layer.mlp)
    if getattr(layer, "is_linear", False):
        layer.linear_attn = DeltaVariant(layer.linear_attn)

for depth in depths:
    ids = c.prompt_ids(d, depth)
    toks = {}
    for arm in order:
        MODE.clear()
        for w in wrapped:
            w.const = None
        eng.prefill(ids, 2048)
        y = mx.array([ids[-1]], dtype=mx.int32)
        seq = []
        for _ in range(2):
            y = eng.step_full(y); mx.eval(y); seq.append(int(y.item()))
        MODE.update(x for x in arm.split("+") if x != "base")
        CALLS["delta"] = 0
        CALLS["lin"] = 0
        for _ in range(64):
            y = eng.step_full(y); mx.eval(y); seq.append(int(y.item()))
        y, prims = c.graph_census(eng, y)
        segs = []
        for _ in range(3):
            y, t = c.loop_pipelined(eng.step_full, y, 96); segs.append(statistics.median(t))
        y, t, b = c.loop_sync(eng.step_full, y, 96)
        base = toks.get("base", seq)
        row = {"arm": arm, "depth": depth, "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")},
               "pipelined_ms": round(statistics.median(segs), 3),
               "pipelined_segs": [round(s, 3) for s in segs], "sync_ms": round(statistics.median(t), 3),
               "build_ms": round(statistics.median(b), 3), "primitives": sum(prims.values()),
               "arange": prims.get("Arange", 0), "delta_variant_calls": CALLS["delta"], "fused_linear_calls": CALLS["lin"],
               "ids_equal_base": (seq == base) if "base" in toks else None,
               "first_diff": next((i for i, (p, q) in enumerate(zip(seq, base)) if p != q), None)}
        toks.setdefault(arm, seq)
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
