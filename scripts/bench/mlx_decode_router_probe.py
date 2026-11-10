#!/usr/bin/env python3
"""Raw-MLX A/B inside one process: where the MoE block's time goes beyond the
routed-expert weights (Qwen3.5/3.6 MoE hybrids on mlx-vlm). Every arm
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

The frozen arms change the routing, so their text differs by construction;
they are ceilings, not candidates.

usage: mlx_decode_router_probe.py <depths csv> <out jsonl> [arms csv]
"""
import json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx_decode_census as c
from mlx_vlm.models import switch_layers as sl

depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
order = (sys.argv[3] if len(sys.argv) > 3 else
         "base,lhs,route_const,inds_const,scores_const,argpart_logits,sm8,sm8,argpart_logits,"
         "scores_const,inds_const,route_const,lhs,base").split(",")
model = os.environ.get("MODEL", "mlx-community/Qwen3.6-35B-A3B-4bit")
mx.set_cache_limit(4 * 2**30)
d = c.resolve_model_dir(model)
eng = c.Engine(d)
MODE = set()
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


class MoeVariant:
    def __init__(self, inner):
        self.inner = inner
        self.const = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, x, target_verify=False):
        m = MODE - {"lhs"}
        if not m or target_verify or x.shape[1] != 1:
            return self.inner(x, target_verify)
        b, k = self.inner, self.inner.top_k
        logits = b.gate(x)
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
        shared = b.shared_expert(x, False)
        if "comb" in m:
            return fused_combine(y, scores, shared, b.shared_expert_gate(x))
        y = (y * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(b.shared_expert_gate(x)) * shared


wrapped = []
for layer in eng.layers:
    layer.mlp = MoeVariant(layer.mlp)
    wrapped.append(layer.mlp)

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
               "arange": prims.get("Arange", 0),
               "ids_equal_base": (seq == base) if "base" in toks else None,
               "first_diff": next((i for i, (p, q) in enumerate(zip(seq, base)) if p != q), None)}
        toks.setdefault(arm, seq)
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
