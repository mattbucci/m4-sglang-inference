#!/usr/bin/env python3
"""Raw-MLX A/B inside one process: the stock decode step against a step whose
MoE blocks and GatedDeltaNet mixers run through mx.compile at the decode
shape (Qwen3.5/3.6 hybrids on mlx-vlm). Each arm re-prefills, takes 64 sync
steps that double as the greedy identity sample against the base arm, then
three 96-step pipelined segments and one 96-step sync segment.

usage: mlx_decode_compile_probe.py <depths csv> <out jsonl> [arms csv]
       arms default: base,moe,dn,moe+dn,moe+dn,dn,moe,base
"""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx_decode_census as c

depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
order = (sys.argv[3] if len(sys.argv) > 3 else "base,moe,dn,moe+dn,moe+dn,dn,moe,base").split(",")
model = "mlx-community/Qwen3.6-35B-A3B-4bit"
mx.set_cache_limit(4 * 2**30)
d = c.resolve_model_dir(model)
eng = c.Engine(d)


class CompiledBlock:
    """Decode-shape calls go through a compiled trace; anything else falls through."""
    on = False

    def __init__(self, inner):
        self.inner = inner
        self.fn = mx.compile(lambda x: inner(x))

    def __call__(self, x, *a, **k):
        if CompiledBlock.on and not a and not k and x.shape[0] == 1 and x.shape[1] == 1:
            return self.fn(x)
        return self.inner(x, *a, **k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Positional:
    """mlx_vlm passes target_verify positionally; treat a False flag as absent."""
    def __init__(self, blk):
        self.blk = blk

    def __call__(self, x, target_verify=False):
        if target_verify:
            return self.blk.inner(x, target_verify)
        return self.blk(x)

    def __getattr__(self, name):
        return getattr(self.blk.inner, name)


from mlx_vlm.models.qwen3_5 import language as _lang


class CompiledDeltaNet:
    """Decode-shape DeltaNet call as a compiled function of (x, conv state, recurrent state)."""
    on = False

    def __init__(self, inner):
        self.inner = inner

        def f(x, cs, rs):
            cache = [cs, rs]
            out = inner(x, None, cache)
            return out, cache[0], cache[1]

        self.fn = mx.compile(f)

    def __call__(self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False):
        if (CompiledDeltaNet.on and mask is None and gdn_sink is None and not target_verify
                and cache is not None and cache[0] is not None and cache[1] is not None
                and getattr(cache, "lengths", None) is None
                and inputs.shape[0] == 1 and inputs.shape[1] == 1):
            out, cache[0], cache[1] = self.fn(inputs, cache[0], cache[1])
            if hasattr(cache, "advance"):
                cache.advance(1)
                _lang._qwen3_5_advance_left_padding_info(cache, 1)
                _lang._qwen3_5_advance_lengths_info(cache, 1)
            return out
        return self.inner(inputs, mask, cache, gdn_sink=gdn_sink, target_verify=target_verify)

    def __getattr__(self, name):
        return getattr(self.inner, name)


for layer in eng.layers:
    layer.mlp = Positional(CompiledBlock(layer.mlp))
    if getattr(layer, "is_linear", False):
        layer.linear_attn = CompiledDeltaNet(layer.linear_attn)

for depth in depths:
    ids = c.prompt_ids(d, depth)
    toks = {}
    for arm in order:
        CompiledBlock.on = False; CompiledDeltaNet.on = False
        pre, _ = eng.prefill(ids, 2048)
        y = mx.array([ids[-1]], dtype=mx.int32)
        seq = []
        for _ in range(2):  # stock steps: populate the decode-conv weight cache outside any trace
            y = eng.step_full(y); mx.eval(y); seq.append(int(y.item()))
        CompiledBlock.on = "moe" in arm
        CompiledDeltaNet.on = "dn" in arm
        for _ in range(64):  # sync warm steps double as the identity sample
            y = eng.step_full(y); mx.eval(y); seq.append(int(y.item()))
        segs = []
        for _ in range(3):
            y, t = c.loop_pipelined(eng.step_full, y, 96); segs.append(statistics.median(t))
        y, t, b = c.loop_sync(eng.step_full, y, 96)
        row = {"arm": arm, "depth": depth, "pipelined_ms": round(statistics.median(segs), 3),
               "pipelined_segs": [round(s, 3) for s in segs], "sync_ms": round(statistics.median(t), 3),
               "build_ms": round(statistics.median(b), 3),
               "ids_equal_base": (seq == toks["base"]) if "base" in toks else None,
               "first_diff": next((i for i, (p, q) in enumerate(zip(seq, toks.get("base", seq))) if p != q), None)}
        toks.setdefault(arm, seq)
        print(json.dumps(row), flush=True)
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")
