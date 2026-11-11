#!/usr/bin/env python3
"""Raw-MLX A/B inside one process for the Flint decode step (mlx-lm ``flint``:
RecursiveSI/flint-int4, mlx-community/flint-oQ*e): what the small launches of
a layer cost next to its weight reads. Every arm re-prefills, runs 64
teacher-forced steps on the base arm's tokens (KL against the base arm's
logprobs, argmax agreement), then three 96-step pipelined segments and one
96-step sync segment.

Arms (combine with "+"):
  base         stock mlx-lm forward
  qkv          the three attention projections as one quantized matmul over
               concatenated weights
  mlp2         gate_proj and up_proj of the shared expert and of the dense MLP
               layers as one quantized matmul
  front        qkv, then one Metal launch for the q and k head norms with
               their scales and the rotary embedding of the sliding layers
               (stock: two rms_norm, two rope)
  comb         one launch for the MoE block's tail: weighted sum of the routed
               experts plus the shared expert (stock: multiply, sum, astype,
               add)
  mid          one launch for the residual add after attention and the
               post-attention norm (stock: add, rms_norm)
  tail         comb, the block's residual add and the next layer's input norm
               in one launch (stock: multiply, sum, astype, add, add, rms_norm)
  gate32       the router matmul reads a float32 copy of its bf16 weight made
               once (stock casts the weight on every step; 2 MB of extra
               resident memory per MoE layer)
  gatek        the router logits in one launch that reads the bf16 weight and
               the hidden state directly (stock: two casts and a matmul)
  sglu         the tree's fused routed-expert kernel (moe/fused_swiglu.py):
               gate projection, SiLU and the product with the up projection
               in one launch
  route_const  expert indices and weights frozen per layer (a ceiling, not a
               candidate: removes the casts, the router matmul and the routing
               kernel while the same experts are read)
  route_replay like route_const, but the routing of the 64 fidelity steps is
               recorded and played back in a cycle, so the experts change
               from step to step as they do in a real decode
  all          front+mlp2+mid+tail+gatek
  all32        front+mlp2+mid+tail+gate32

Every arm except sglu, route_const and route_replay is bit-identical to stock by
construction: a folded quantized matmul computes each output column from its
own weight row, and the kernels sum in the order of the stock kernels they
replace (rms_norm: four consecutive values per thread, SIMD sum, then across
SIMD groups; column reduce: row after row; the matrix-vector kernel: 4x4
tiles per thread, shuffle-down reduction) and round to the model dtype
wherever the stock chain does. ``--selftest`` checks each kernel against the
stock chain in float32, where any other order shows, and in bf16. The router
kernel is used only where it reproduces the stock matmul bit for bit on the
block's own weight (the stock tile layout depends on the matrix shape).

usage: mlx_flint_decode_probe.py <depths csv> <out jsonl> [arms csv]
       mlx_flint_decode_probe.py --selftest [trials]
env:   MODEL (default RecursiveSI/flint-int4), CHUNK (prefill chunk, 1024),
       CACHE_GB (MLX buffer cache limit, 2)
"""
import importlib.util, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.kernels import expert_routing
from mlx_lm.kernels._common import scalar
from mlx_lm.models import flint as fl
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask

MODE = set()
CALLS = {"fold": 0, "front": 0, "comb": 0, "mid": 0, "tail": 0, "gate": 0, "sglu": 0}
ALIASES = {"all": "front+mlp2+mid+tail+gatek", "all32": "front+mlp2+mid+tail+gate32"}
DEFAULT_ARMS = ("base,qkv,mlp2,front,comb,mid,tail,gate32,gatek,sglu,all,all32,all+sglu,"
                "route_const,all+route_const")
_FOLD, _K, _W32, _GATE_OK, _SGLU, _CONST = {}, {}, {}, {}, {}, {}
_REPLAY, _REPLAY_AT, REPLAY_STEPS = {}, {}, 64

# The sum over a threadgroup of TG threads in the order of mx.fast.rms_norm.
_BLOCK_SUM_HEADER = """
template <int TG>
inline float block_sum_once(float v, threadgroup float* buf, uint lane, uint sg) {
    v = simd_sum(v);
    if (TG > 32) {
        if (lane == 0) {
            buf[sg] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = simd_sum(lane < TG / 32 ? buf[lane] : 0.0f);
    }
    return v;
}
"""

# One SIMD group per head. The rows are the HQ query heads, then the HK key
# heads: the first (HQ + HK) * D values of the folded q|k|v output. The sum of
# squares runs in the stock order (lane l adds the values 4 l .. 4 l + 3); for
# the outputs a lane holds the values d = lane + 32 i of its head, so both
# halves of a rotary pair (d, d + D / 2) sit in one lane.
_FRONT_SRC = """
    constexpr int R = D / 32;
    constexpr int HALF = R / 2;
    uint h = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    size_t base = size_t(h) * D;

    float acc = 0.0f;
    if (lane * 4 < D) {
        for (int i = 0; i < 4; i++) {
            float xi = static_cast<float>(qkv[base + lane * 4 + i]);
            acc += xi * xi;
        }
    }
    float rstd = metal::precise::rsqrt(simd_sum(acc) / D + eps);
    T y[R];
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        T n = static_cast<T>(static_cast<float>(qkv[base + d]) * rstd);
        float w = h < HQ ? static_cast<float>(qw[d]) : static_cast<float>(kw[d]);
        y[i] = static_cast<T>(w * static_cast<float>(n));
    }
    if (ROPE) {
        float pos = static_cast<float>(offset);
        for (int i = 0; i < HALF; i++) {
            int d = lane + 32 * i;
            float inv_freq = metal::exp2(-(static_cast<float>(d) / static_cast<float>(D / 2)) * log2_base);
            float theta = pos * inv_freq;
            float c = metal::fast::cos(theta);
            float s = metal::fast::sin(theta);
            float x1 = static_cast<float>(y[i]);
            float x2 = static_cast<float>(y[i + HALF]);
            y[i] = static_cast<T>(x1 * c - x2 * s);
            y[i + HALF] = static_cast<T>(x1 * s + x2 * c);
        }
    }
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        if (h < HQ) {
            q[base + d] = y[i];
        } else {
            k[size_t(h - HQ) * D + d] = y[i];
        }
    }
"""

# One threadgroup per row of D values; a thread owns four consecutive values.
_ROW_HEAD = """
    threadgroup float buf[32];
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t row = threadgroup_position_in_grid.x;
    size_t base = row * D;
    T s[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
        int d = t * 4 + i;
        s[i] = static_cast<T>(0.0f);
        if (d < D) {
"""

_ROW_TAIL = """
            res[base + d] = s[i];
            float f = static_cast<float>(s[i]);
            acc += f * f;
        }
    }
    float rstd = metal::precise::rsqrt(block_sum_once<TG>(acc, buf, lane, sg) / D + eps);
    for (int i = 0; i < 4; i++) {
        int d = t * 4 + i;
        if (d < D) {
            T n = static_cast<T>(static_cast<float>(s[i]) * rstd);
            normed[base + d] = static_cast<T>(static_cast<float>(nw[d]) * static_cast<float>(n));
        }
    }
"""

_ADD_NORM_SRC = _ROW_HEAD + """
            s[i] = static_cast<T>(static_cast<float>(xa[base + d]) + static_cast<float>(xb[base + d]));
""" + _ROW_TAIL

# The weighted sum adds the experts one after another, like the stock column
# reduce; the product is rounded to float32 before it is added.
_COMB_EXPR = """
            float tot = 0.0f;
            for (int j = 0; j < K; j++) {
                float p = static_cast<float>(y[(row * K + j) * D + d]) * static_cast<float>(wts[row * K + j]);
                tot = p + tot;
            }
            T r = static_cast<T>(tot);
            r = static_cast<T>(static_cast<float>(r) + static_cast<float>(shared[base + d]));
"""

_COMB_ADD_NORM_SRC = _ROW_HEAD + _COMB_EXPR + """
            s[i] = static_cast<T>(static_cast<float>(xa[base + d]) + static_cast<float>(r));
""" + _ROW_TAIL

_COMB_SRC = """
    uint d = thread_position_in_grid.x;
    size_t row = thread_position_in_grid.y;
    size_t base = row * D;
""" + _COMB_EXPR + """
    out[base + d] = r;
"""

# The stock matrix-vector kernel for this shape: a SIMD group owns four rows,
# a lane four consecutive columns of every 128, then a shuffle-down reduction.
_GATE_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint row = (thread_position_in_grid.x / 32) * 4;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = lane * 4;
    for (int i = 0; i < D / 128; i++) {
        float v[4];
        for (int tn = 0; tn < 4; tn++) {
            v[tn] = static_cast<float>(x[bn + tn]);
        }
        for (int tm = 0; tm < 4; tm++) {
            for (int tn = 0; tn < 4; tn++) {
                result[tm] += static_cast<float>(w[size_t(row + tm) * D + bn + tn]) * v[tn];
            }
        }
        bn += 128;
    }
    for (int tm = 0; tm < 4; tm++) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    if (lane == 0) {
        for (int tm = 0; tm < 4; tm++) {
            logits[row + tm] = result[tm];
        }
    }
"""


def _kernel(name, ins, outs, src, header=""):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(
            name=name, input_names=ins, output_names=outs, source=src, header=header
        )
    return _K[name]


def _row_tg(d):
    """Threads of mx.fast.rms_norm's threadgroup for a row of d values."""
    return 32 * ((d + 127) // 128)


def front(qkv, qw, kw, eps, offset, rope, hq, hk, d, log2_base):
    """q (1, hq, 1, d) and k (1, hk, 1, d) from one token's folded q|k|v row."""
    return _kernel(
        "flint_front", ["qkv", "qw", "kw", "eps", "offset", "log2_base"], ["q", "k"], _FRONT_SRC
    )(
        inputs=[qkv, qw, kw, scalar(eps), mx.array(offset, dtype=mx.int32), scalar(log2_base)],
        template=[("T", qkv.dtype), ("D", d), ("HQ", hq), ("HK", hk), ("ROPE", int(rope))],
        grid=((hq + hk) * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, hq, 1, d), (1, hk, 1, d)],
        output_dtypes=[qkv.dtype, qkv.dtype],
    )


def add_norm(a, b, norm):
    """``a + b`` and ``norm(a + b)`` in one launch."""
    d = a.shape[-1]
    tg = _row_tg(d)
    return _kernel(
        "flint_add_norm", ["xa", "xb", "nw", "eps"], ["res", "normed"], _ADD_NORM_SRC,
        _BLOCK_SUM_HEADER,
    )(
        inputs=[a, b, norm.weight, scalar(norm.eps)],
        template=[("T", a.dtype), ("D", d), ("TG", tg)],
        grid=(a.size // d * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[a.shape, a.shape],
        output_dtypes=[a.dtype, a.dtype],
    )


def comb_add_norm(y, wts, shared, h, norm):
    """``h + (sum_k y_k w_k + shared)`` and its norm in one launch."""
    k, d = y.shape[-2:]
    tg = _row_tg(d)
    return _kernel(
        "flint_comb_add_norm", ["y", "wts", "shared", "xa", "nw", "eps"], ["res", "normed"],
        _COMB_ADD_NORM_SRC, _BLOCK_SUM_HEADER,
    )(
        inputs=[y, wts, shared, h, norm.weight, scalar(norm.eps)],
        template=[("T", y.dtype), ("D", d), ("K", k), ("TG", tg)],
        grid=(h.size // d * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[h.shape, h.shape],
        output_dtypes=[y.dtype, y.dtype],
    )


def comb(y, wts, shared):
    """``sum_k y_k w_k + shared`` in one launch."""
    k, d = y.shape[-2:]
    rows = shared.size // d
    return _kernel("flint_comb", ["y", "wts", "shared"], ["out"], _COMB_SRC)(
        inputs=[y, wts, shared],
        template=[("T", y.dtype), ("D", d), ("K", k)],
        grid=(d, rows, 1),
        threadgroup=(min(d, 256), 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[y.dtype],
    )[0]


def gate_logits(weight, x):
    """float32 router logits (1, 1, E) of one token from the (E, D) weight."""
    e, d = weight.shape
    return _kernel("flint_gate", ["w", "x"], ["logits"], _GATE_SRC)(
        inputs=[weight, x],
        template=[("D", d)],
        grid=(e // 4 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, e)],
        output_dtypes=[mx.float32],
    )[0]


def gate_kernel_ok(gate, x):
    """True where the kernel reproduces the stock router matmul bit for bit on
    this weight (checked once per block on random hidden states)."""
    ok = _GATE_OK.get(id(gate))
    if ok is None:
        w = gate.weight
        ok = ("bias" not in gate and w.ndim == 2 and w.shape[0] % 4 == 0
              and w.shape[1] % 128 == 0 and x.size == w.shape[1])
        if ok:
            key = mx.random.key(len(_GATE_OK) + 1)
            for i in range(4):
                probe = (2 * mx.random.normal(x.shape, key=mx.random.split(key, 4)[i])).astype(x.dtype)
                if not mx.array_equal(gate_logits(w, probe), gate(probe.astype(mx.float32))).item():
                    ok = False
                    break
        if not ok and not any(v is False for v in _GATE_OK.values()):
            print(f"# router kernel not bit-identical for weight {tuple(w.shape)}: stock matmul kept",
                  file=sys.stderr, flush=True)
        _GATE_OK[id(gate)] = ok
    return ok


def fold(linears, x):
    """(raw output, split points) of one quantized matmul over the linears'
    concatenated weights, or None when they do not share a quantization."""
    first = linears[0]
    ent = _FOLD.get(id(first))
    if ent is None:
        ok = all(
            isinstance(l, nn.QuantizedLinear) and "bias" not in l
            and getattr(l, "biases", None) is not None
            and (l.bits, l.group_size, l.mode) == (first.bits, first.group_size, first.mode)
            and l.scales.dtype == x.dtype and l.biases.dtype == x.dtype
            for l in linears
        )
        ent = False
        if ok:
            w = mx.concatenate([l.weight for l in linears], axis=0)
            sc = mx.concatenate([l.scales for l in linears], axis=0)
            bi = mx.concatenate([l.biases for l in linears], axis=0)
            splits, off = [], 0
            for l in linears[:-1]:
                off += l.weight.shape[0]
                splits.append(off)
            mx.eval(w, sc, bi)
            ent = (w, sc, bi, splits)
        _FOLD[id(first)] = ent
    if not ent:
        return None
    w, sc, bi, splits = ent
    CALLS["fold"] += 1
    out = mx.quantized_matmul(x, w, scales=sc, biases=bi, transpose=True,
                              group_size=first.group_size, bits=first.bits, mode=first.mode)
    return out, splits


def _tree_module(rel):
    """A module of the SGLang MLX backend loaded by path (it imports mlx only)."""
    path = os.path.join(HERE, "..", "..", "components", "sglang", "python", "sglang", "srt",
                        "hardware_backend", "mlx", rel)
    name = "_probe_" + os.path.basename(rel)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_attn_call = fl.Attention.__call__
_mlp_call = fl.MLP.__call__
_moe_call = fl.SparseMoeBlock.__call__
_model_call = fl.FlintModel.__call__


def attn_call(self, x, mask=None, cache=None):
    if not (MODE & {"qkv", "front"}) or x.shape[0] != 1 or x.shape[1] != 1 or cache is None:
        return _attn_call(self, x, mask, cache)
    folded = fold((self.q_proj, self.k_proj, self.v_proj), x)
    if folded is None:
        return _attn_call(self, x, mask, cache)
    out, splits = folded
    hq, hk = self.n_heads, self.n_kv_heads
    d = splits[0] // hq
    if "front" in MODE and d in (64, 128):
        CALLS["front"] += 1
        rope = self.rope
        queries, keys = front(
            out, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps, cache.offset,
            rope is not None, hq, hk, d, math.log2(rope.base) if rope is not None else 0.0,
        )
        values = out[..., splits[1]:].reshape(1, 1, hk, d).transpose(0, 2, 1, 3)
    else:
        q, k, v = mx.split(out, splits, axis=-1)
        queries = self.q_norm(q.reshape(1, 1, hq, d)).transpose(0, 2, 1, 3)
        keys = self.k_norm(k.reshape(1, 1, hk, d)).transpose(0, 2, 1, 3)
        values = v.reshape(1, 1, hk, d).transpose(0, 2, 1, 3)
        if self.rope is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
    keys, values = cache.update_and_fetch(keys, values)
    output = fl.scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask, sinks=self.softmax_offset
    )
    return self.o_proj(output.transpose(0, 2, 1, 3).reshape(1, 1, -1))


def mlp_call(self, x):
    if "mlp2" in MODE and x.shape[1] == 1:
        folded = fold((self.gate_proj, self.up_proj), x)
        if folded is not None:
            g, u = mx.split(folded[0], folded[1], axis=-1)
            return self.down_proj(swiglu(g, u))
    return _mlp_call(self, x)


def route(block, x):
    if "route_const" in MODE and id(block) in _CONST:
        return _CONST[id(block)]
    seq = _REPLAY.setdefault(id(block), []) if "route_replay" in MODE else None
    if seq is not None and len(seq) == REPLAY_STEPS:
        at = _REPLAY_AT[id(block)] = (_REPLAY_AT.get(id(block), -1) + 1) % REPLAY_STEPS
        return seq[at]
    gate = block.gate
    if "gatek" in MODE and gate_kernel_ok(gate, x):
        CALLS["gate"] += 1
        logits = gate_logits(gate.weight, x)
    elif "gate32" in MODE and "bias" not in gate:
        CALLS["gate"] += 1
        w32 = _W32.get(id(gate))
        if w32 is None:
            w32 = _W32[id(gate)] = gate.weight.astype(mx.float32)
            mx.eval(w32)
        logits = x.astype(mx.float32) @ w32.T
    else:
        logits = gate(x.astype(mx.float32))
    inds, wts = expert_routing(
        logits, block._zero_bias, top_k=block.top_k, multiplier=1.0, score_eps=0.0, route_scale=1.0
    )
    if "route_const" in MODE:
        mx.eval(inds, wts)
        _CONST[id(block)] = (inds, wts)
    if seq is not None:
        mx.eval(inds, wts)
        seq.append((inds, wts))
    return inds, wts


def experts(block, x, inds):
    if "sglu" in MODE:
        fused = _SGLU.get(id(block))
        if fused is None:
            fs = _SGLU.get("module") or _SGLU.setdefault("module", _tree_module("moe/fused_swiglu.py"))
            fused = fs.FusedSwitchSwiGLU(block.switch_mlp) if fs.can_fuse(block.switch_mlp) else False
            _SGLU[id(block)] = fused
        if fused:
            CALLS["sglu"] += 1
            return fused.fused_forward(x, inds)
    return block.switch_mlp(x, inds)


def moe_call(self, x):
    if (not (MODE & {"comb", "route_const", "route_replay", "gate32", "gatek", "sglu"}) or x.shape[1] != 1
            or self.sharding_group is not None):
        return _moe_call(self, x)
    inds, wts = route(self, x)
    y = experts(self, x, inds)
    if "comb" in MODE:
        CALLS["comb"] += 1
        return comb(y, wts, self.shared_expert(x))
    y = (y * wts[..., None]).sum(axis=-2).astype(y.dtype)
    return y + self.shared_expert(x)


def model_call(self, inputs, cache=None, input_embeddings=None):
    if not (MODE & {"mid", "tail"}) or cache is None or inputs.shape != (1, 1):
        return _model_call(self, inputs, cache, input_embeddings)
    h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
    fa_mask = swa_mask = None
    if self.fa_idx is not None:
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
    if self.swa_idx is not None:
        swa_mask = create_attention_mask(
            h, cache[self.swa_idx], window_size=self.args.sliding_window
        )
    n = len(self.layers)
    normed = self.layers[0].input_layernorm(h)
    for i, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
        nxt = self.layers[i + 1].input_layernorm if i + 1 < n else self.norm
        a = layer.self_attn(normed, swa_mask if layer.use_sliding else fa_mask, c)
        if "mid" in MODE:
            CALLS["mid"] += 1
            h, n2 = add_norm(h, a, layer.post_attention_layernorm)
        else:
            h = h + a
            n2 = layer.post_attention_layernorm(h)
        mlp = layer.mlp
        if "tail" not in MODE:
            h = h + mlp(n2)
            normed = nxt(h)
        elif isinstance(mlp, fl.SparseMoeBlock) and mlp.sharding_group is None:
            CALLS["tail"] += 1
            inds, wts = route(mlp, n2)
            y = experts(mlp, n2, inds)
            h, normed = comb_add_norm(y, wts, mlp.shared_expert(n2), h, nxt)
        else:
            CALLS["tail"] += 1
            h, normed = add_norm(h, mlp(n2), nxt)
    return normed


def set_mode(arm):
    """Switch to an arm ("+"-joined names and aliases) and zero the counters."""
    MODE.clear()
    for state in (_CONST, _REPLAY, _REPLAY_AT):
        state.clear()
    for name in arm.split("+"):
        MODE.update(ALIASES.get(name, name).split("+"))
    MODE.discard("base")
    for key in CALLS:
        CALLS[key] = 0


def reset():
    """Forget every per-module cache (they are keyed by module identity, so a
    second model in the same process needs this)."""
    module = _SGLU.get("module")
    for cache in (_FOLD, _W32, _GATE_OK, _SGLU, _CONST):
        cache.clear()
    if module is not None:
        _SGLU["module"] = module


def install():
    fl.Attention.__call__ = attn_call
    fl.MLP.__call__ = mlp_call
    fl.SparseMoeBlock.__call__ = moe_call
    fl.FlintModel.__call__ = model_call


class _Tally:
    def __init__(self):
        self.rows = {}

    def add(self, name, got, want):
        bad, n = self.rows.get(name, (0, 0))
        self.rows[name] = (bad + int((got != want).sum().item()), n + got.size)

    def show(self, expect_equal=True):
        ok = True
        for name, (bad, n) in self.rows.items():
            good = (bad == 0) if expect_equal else (bad > 0)
            ok &= good
            print(f"  {name:<34} {bad:>8} of {n} values differ{'' if good else '   <-- UNEXPECTED'}")
        return ok


def selftest(trials=40):
    """Each kernel against the stock chain on random tensors: float32, where a
    different summation order shows, then bf16 with the stock rounding points.
    Ends with controls that must differ."""
    mx.random.seed(0)
    D, HQ, HK, H, K, E = 64, 32, 4, 2048, 6, 256
    ok = True
    for T in (mx.float32, mx.bfloat16):
        tally = _Tally()
        qn, kn, norm = nn.RMSNorm(D, eps=1e-6), nn.RMSNorm(D, eps=1e-6), nn.RMSNorm(H, eps=1e-6)
        rope = nn.RoPE(D, traditional=False, base=10000.0)
        lin = nn.Linear(H, E, bias=False)
        for trial in range(trials):
            qn.weight = (1 + 0.3 * mx.random.normal((D,))).astype(T)
            kn.weight = (1 + 0.3 * mx.random.normal((D,))).astype(T)
            norm.weight = (1 + 0.3 * mx.random.normal((H,))).astype(T)
            off = (0, 1, 2, 77, 512, 513, 4097, 30000, 131071, 262143)[trial % 10] + trial // 10
            qkv = (3 * mx.random.normal((1, 1, (HQ + 2 * HK) * D))).astype(T)
            q = qn(qkv[..., : HQ * D].reshape(1, 1, HQ, D)).transpose(0, 2, 1, 3)
            k = kn(qkv[..., HQ * D : (HQ + HK) * D].reshape(1, 1, HK, D)).transpose(0, 2, 1, 3)
            for rp in (False, True):
                qs, ks = (rope(q, offset=off), rope(k, offset=off)) if rp else (q, k)
                qf, kf = front(qkv, qn.weight, kn.weight, 1e-6, off, rp, HQ, HK, D, math.log2(10000.0))
                tally.add(f"front q rope={int(rp)}", qf, qs)
                tally.add(f"front k rope={int(rp)}", kf, ks)
            a = (2 * mx.random.normal((1, 1, H))).astype(T)
            b = (0.5 * mx.random.normal((1, 1, H))).astype(T)
            y = mx.random.normal((1, 1, K, H)).astype(T)
            shared = mx.random.normal((1, 1, H)).astype(T)
            wts = mx.softmax(mx.random.normal((1, 1, K)), axis=-1)
            res, normed = add_norm(a, b, norm)
            tally.add("add_norm sum", res, a + b)
            tally.add("add_norm normed", normed, norm(a + b))
            stock = (y * wts[..., None]).sum(axis=-2).astype(T) + shared
            tally.add("comb", comb(y, wts, shared), stock)
            res, normed = comb_add_norm(y, wts, shared, a, norm)
            tally.add("comb_add_norm sum", res, a + stock)
            tally.add("comb_add_norm normed", normed, norm(a + stock))
            lin.weight = (0.05 * mx.random.normal((E, H))).astype(T)
            x = (2 * mx.random.normal((1, 1, H))).astype(T)
            tally.add("router logits (256 x 2048)", gate_logits(lin.weight, x), lin(x.astype(mx.float32)))
            tally.add("router, float32 weight copy", x.astype(mx.float32) @ lin.weight.astype(mx.float32).T,
                      lin(x.astype(mx.float32)))
        print(f"{T}, {trials} trials: kernel against the stock chain")
        ok &= tally.show()
    T = mx.bfloat16
    ctl = _Tally()
    qf, _ = front(qkv, qn.weight, kn.weight, 1e-6, 700, True, HQ, HK, D, math.log2(10000.0))
    ctl.add("rope offset 700 against 701", qf, rope(q, offset=701))
    qf, _ = front(qkv, kn.weight, qn.weight, 1e-6, 700, True, HQ, HK, D, math.log2(10000.0))
    ctl.add("q head norm with the k scale", qf, rope(q, offset=700))
    ctl.add("comb with reversed weights", comb(y, wts[..., ::-1], shared), stock)
    one = ((y.astype(mx.float32) * wts[..., None]).sum(axis=-2) + shared.astype(mx.float32)).astype(T)
    ctl.add("comb rounded once, not twice", one, stock)
    ctl.add("normed against the plain sum", normed, a + stock)
    print("controls (each must differ)")
    ok &= ctl.show(expect_equal=False)
    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return ok


def logprobs(logits):
    l = logits.astype(mx.float32)
    return l - mx.logsumexp(l, axis=-1, keepdims=True)


def teacher(eng, first, forced, steps=64):
    """Logprobs and argmax of ``steps`` decode steps fed ``first`` and then the
    forced tokens (the arm's own argmax when ``forced`` is None)."""
    lps, arg, tok = [], [], first
    for i in range(steps):
        lp = logprobs(eng.logits(mx.array([[tok]], dtype=mx.int32))[:, -1, :])
        a = mx.argmax(lp, axis=-1)
        mx.eval(lp, a)
        lps.append(lp)
        arg.append(int(a.item()))
        tok = arg[-1] if forced is None else forced[i]
    return lps, arg


def main():
    import mlx_decode_census as c

    depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
    arms = (sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ARMS).split(",")
    order = arms + arms[::-1]
    model = os.environ.get("MODEL", "RecursiveSI/flint-int4")
    chunk = int(os.environ.get("CHUNK", "1024"))
    mx.set_cache_limit(int(float(os.environ.get("CACHE_GB", "2")) * 2**30))
    d = c.resolve_model_dir(model)
    eng = c.Engine(d)
    install()
    for depth in depths:
        ids = c.prompt_ids(d, depth)
        ref = None
        for arm in order:
            set_mode("base")
            eng.prefill(ids, chunk)
            set_mode(arm)
            lps, arg = teacher(eng, ids[-1], ref[1] if ref else None)
            calls = dict(CALLS)
            if ref is None:
                ref = (lps, arg)
            kls = [float((mx.exp(b) * (b - a)).sum().item()) for a, b in zip(lps, ref[0])]
            same = all(mx.array_equal(a, b).item() for a, b in zip(lps, ref[0]))
            agree = sum(int(p == q) for p, q in zip(arg, ref[1]))
            y = mx.array([arg[-1]], dtype=mx.int32)
            y, prims = c.graph_census(eng, y)
            segs = []
            for _ in range(3):
                y, t = c.loop_pipelined(eng.step_full, y, 96)
                segs.append(statistics.median(t))
            y, t, b = c.loop_sync(eng.step_full, y, 96)
            row = {"arm": arm, "depth": depth, "model": os.path.basename(model.rstrip("/")),
                   "chunk": chunk,
                   "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")},
                   "pipelined_ms": round(statistics.median(segs), 3),
                   "pipelined_segs": [round(s, 3) for s in segs],
                   "tok_s": round(1000.0 / statistics.median(segs), 2),
                   "sync_ms": round(statistics.median(t), 3),
                   "build_ms": round(statistics.median(b), 3),
                   "primitives": sum(prims.values()), "primitive_counts": prims,
                   "calls_64_steps": calls, "logprobs_bit_identical": same,
                   "kl_mean": round(statistics.mean(kls), 8), "kl_max": round(max(kls), 8),
                   "argmax_agree": f"{agree}/{len(arg)}"}
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest(int(sys.argv[2]) if len(sys.argv) > 2 else 40) else 1)
    main()
