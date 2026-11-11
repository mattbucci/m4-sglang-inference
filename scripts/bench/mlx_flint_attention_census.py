#!/usr/bin/env python3
"""Raw-MLX knock-out census inside Flint's decode attention.

The decode census prices attention as one family. This one splits it: each arm
takes one part of every layer's attention out of the step and reports the step
time that saves next to the bytes the part reads.

  q_proj, kv_proj           the projection returns an output captured from one
                            real call, so its launch and its reads leave
  o_proj                    returns the leading columns of its input at its
                            output width, so its launch and its reads leave
  head_norms                q_norm and k_norm pass their input through
  rope                      the rotary embedding passes its input through
  cache_write               update_and_fetch returns the captured buffers: no
                            cache write, and the key and value path is dead
  sdpa_ko                   attention returns its queries: the launch leaves,
                            and so does everything cache_write removes
  sdpa_x2, sdpa_ctl         attention runs twice per layer, and a control with
                            the same two extra launches; x2 minus ctl is one
                            more attention launch per layer on warm buffers
  attention                 the whole module passes its input through

Derived rows: sdpa is sdpa_ko minus cache_write; cache_write_alone is
cache_write minus kv_proj and half of head_norms and rope (the key side).

Arms run in both orders; every arm re-prefills the same prompt.

usage: mlx_flint_attention_census.py <depths csv> <out jsonl>
env:   MODEL (default RecursiveSI/flint-int4), CHUNK (prefill chunk, 1024),
       STEPS (per segment, 96), GBPS (read rate for the floor column, 200),
       CACHE_GB (MLX buffer cache limit, 2)
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_flatten


class Captured:
    """The first call is real; later calls return that call's output."""

    def __init__(self, inner):
        self._inner, self._out = inner, None

    def __call__(self, *a, **k):
        if self._out is None:
            self._out = self._inner(*a, **k)
        return self._out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Narrowed:
    """Returns the leading columns of its input at the module's output width."""

    def __init__(self, inner):
        self._inner, self._width = inner, inner.weight.shape[0]

    def __call__(self, x, *a, **k):
        return x[..., : self._width]

    def __getattr__(self, name):
        return getattr(self._inner, name)


def shadow(targets):
    """Shadow (owner, attr, replacement) triples; returns the undo call."""
    for owner, attr, fn in targets:
        object.__setattr__(owner, attr, fn)

    def undo():
        for owner, attr, _ in targets:
            object.__delattr__(owner, attr)

    return undo


def module_arms(c, attns):
    def over(attrs, kind):
        def install(eng):
            return shadow([(a, n, kind(getattr(a, n))) for a in attns for n in attrs
                           if getattr(a, n) is not None])
        return install

    return {
        "q_proj": over(["q_proj"], Captured),
        "kv_proj": over(["k_proj", "v_proj"], Captured),
        "o_proj": over(["o_proj"], Narrowed),
        "head_norms": over(["q_norm", "k_norm"], c.PassThrough),
        "rope": over(["rope"], c.PassThrough),
    }


def cache_write_arm(eng):
    return shadow([(k, "update_and_fetch", Captured(k.update_and_fetch)) for k in eng.cache])


def sdpa_arm(mode):
    import mlx_lm.models.flint as fm

    def install(eng):
        real, first = fm.scaled_dot_product_attention, {}

        def call(queries, keys, values, **kw):
            if mode == "ko":
                return queries
            out = real(queries, keys, values, **kw)
            if mode == "x2":
                return out + 0.0 * real(queries, keys, values, **kw)
            return out + 0.0 * first.setdefault(id(kw.get("cache")), out)

        fm.scaled_dot_product_attention = call
        return lambda: setattr(fm, "scaled_dot_product_attention", real)

    return install


def measure(c, eng, ids, chunk, steps, install):
    eng.prefill(ids, chunk)
    undo = install(eng) if install else (lambda: None)
    try:
        y = mx.array([ids[-1]], dtype=mx.int32)
        y, _ = c.loop_pipelined(eng.step_full, y, 32)
        segs = []
        for _ in range(3):
            y, total = c.loop_pipelined(eng.step_full, y, steps)
            segs.append(statistics.median(total))
    finally:
        undo()
    return segs


def nbytes(module):
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


def main():
    import mlx_decode_census as c

    depths, out = [int(x) for x in sys.argv[1].split(",")], sys.argv[2]
    model = os.environ.get("MODEL", "RecursiveSI/flint-int4")
    chunk = int(os.environ.get("CHUNK", "1024"))
    steps = int(os.environ.get("STEPS", "96"))
    gbps = float(os.environ.get("GBPS", "200"))
    mx.set_cache_limit(int(float(os.environ.get("CACHE_GB", "2")) * 2**30))
    d = c.resolve_model_dir(model)
    eng = c.Engine(d)
    attns = [layer.self_attn for layer in eng.layers]
    arms = {"base": None, **module_arms(c, attns), "cache_write": cache_write_arm,
            "sdpa_ko": sdpa_arm("ko"), "sdpa_x2": sdpa_arm("x2"), "sdpa_ctl": sdpa_arm("ctl"),
            "attention": lambda eng: shadow([(layer, "self_attn", c.PassThrough(layer.self_attn))
                                             for layer in eng.layers])}
    weights = {
        "q_proj": sum(nbytes(a.q_proj) for a in attns),
        "kv_proj": sum(nbytes(a.k_proj) + nbytes(a.v_proj) for a in attns),
        "o_proj": sum(nbytes(a.o_proj) for a in attns),
    }
    for depth in depths:
        ids = c.prompt_ids(d, depth)
        eng.prefill(ids, chunk)
        kv = sum(a.nbytes for k in eng.cache for a in k.state if isinstance(a, mx.array))
        reads = {**weights, "sdpa": kv, "sdpa_again": kv, "attention": sum(weights.values()) + kv}
        times = {}
        for name in list(arms) + list(arms)[::-1]:
            segs = measure(c, eng, ids, chunk, steps, arms[name])
            times.setdefault(name, []).append(statistics.median(segs))
            row = {"model": os.path.basename(model.rstrip("/")), "depth": depth, "arm": name,
                   "step_ms": round(statistics.median(segs), 3), "step_segs": [round(x, 3) for x in segs]}
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        step = {k: statistics.mean(v) for k, v in times.items()}
        saved = {k: step["base"] - v for k, v in step.items() if k != "base"}
        saved["sdpa"] = saved["sdpa_ko"] - saved["cache_write"]
        saved["sdpa_again"] = step["sdpa_x2"] - step["sdpa_ctl"]
        saved["cache_write_alone"] = (saved["cache_write"] - saved["kv_proj"]
                                      - saved["head_norms"] / 2 - saved["rope"] / 2)
        parts = ["q_proj", "kv_proj", "o_proj", "head_norms", "rope", "cache_write_alone", "sdpa"]
        saved["rest"] = saved["attention"] - sum(saved[p] for p in parts)
        summary = {"model": os.path.basename(model.rstrip("/")), "depth": depth, "summary": True,
                   "base_ms": round(step["base"], 3), "base_pair": [round(x, 3) for x in times["base"]],
                   "layers": len(attns), "gbps": gbps, "parts": {}}
        print(f"\n== depth {depth}: base {step['base']:.3f} ms ({times['base'][0]:.3f} / {times['base'][1]:.3f})")
        print("| part | saved ms | per layer µs | MB read | read ms | above reads ms |")
        print("|---|--:|--:|--:|--:|--:|")
        for p in parts + ["rest", "attention", "sdpa_again"]:
            mb = reads.get(p, 0) / 1e6
            read_ms = mb / gbps
            summary["parts"][p] = {"saved_ms": round(saved[p], 3), "mb_read": round(mb, 1),
                                   "read_ms": round(read_ms, 3), "above_read_ms": round(saved[p] - read_ms, 3)}
            print(f"| {p} | {saved[p]:.3f} | {saved[p] * 1e3 / len(attns):.1f} | {mb:.1f} | {read_ms:.3f} "
                  f"| {saved[p] - read_ms:.3f} |")
        with open(out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
