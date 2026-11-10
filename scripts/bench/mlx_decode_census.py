#!/usr/bin/env python3
"""Decode-step census for an MLX checkpoint, no server.

Loads the checkpoint the way the SGLang MLX runner does (mlx_vlm model-only
load for VLM configs, mlx_lm for text), prefills to each requested depth and
attributes one decode step:

  loop modes   sync       build the lazy graph, then block on the token (the
                          shape of the scheduler's normal event loop)
               pipelined  build step n+1 and hand it to mx.async_eval before
                          blocking on step n (mlx_lm.generate_step; the shape
                          of the SGLang MLX overlap loop)
               trunk      pipelined, no logit head (head cost by difference)
               fused      pipelined, greedy head in mlx-vlm's quantized argmax
                          kernel (what mlx_vlm's own generator runs at T=0)
  build/eval   Python graph-build time against device time in the sync loop
  knock-outs   step time with one module family replaced by a pass-through;
               the drop is that family's marginal cost in the pipelined regime
  graph        primitives per decode step, counted from the exported graph

The raw loops are the comparator engine for the server: server ms/step minus
the pipelined row is the host gap (scheduler, IPC, detokenizer, streaming).

  python scripts/bench/mlx_decode_census.py --model mlx-community/Qwen3.6-35B-A3B-4bit \
      --depths 24,8192,32768 --out benchmarks/perf/<date>/census-qwen36.json

Run the OOM guard for depths of 32K and beyond; it does not know this
process, so keep --depths inside what the server recipe has already served.
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import mlx.core as mx

WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his "
    "from at which but have an had they you were their one all we can her has there "
    "been if more when will would who so no out up into than them only some could "
    "time these two may then do first any my now such like our over man me even most "
    "made after also did many before must through years where much your way well down "
    "should because each just those people how too little state good very make world "
    "still own see men work long get here between both life being under never day same "
    "another know while last might us great old year off come since against go came "
    "right used take three function return value index buffer cache token model layer "
    "decode kernel memory server request context length array state update result error"
).split()


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    # The hub cache directly: snapshot_download(local_files_only=True) rejects
    # snapshots that lack README/.gitattributes, which the server never needs.
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snaps = sorted(
        glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots", "*")),
        key=os.path.getmtime,
    )
    snaps = [d for d in snaps if os.path.exists(os.path.join(d, "config.json"))]
    if not snaps:
        raise SystemExit(f"no cached snapshot for {model} under {hub}")
    return snaps[-1]


def is_vlm_config(model_dir: str) -> bool:
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return any(k in cfg for k in ("vision_config", "image_token_id", "image_token_index"))


def prompt_ids(model_dir: str, n: int, seed: int = 7) -> list[int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    rng = random.Random(seed)
    ids: list[int] = []
    while len(ids) < n:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18)))
        ids.extend(tok.encode(sent.capitalize() + ". ", add_special_tokens=False).ids)
    return ids[:n]


class Engine:
    """One loaded checkpoint plus the calls the census needs."""

    def __init__(self, model_dir: str):
        self.vlm = is_vlm_config(model_dir)
        if self.vlm:
            from mlx_vlm.utils import load_model

            self.model = load_model(pathlib.Path(model_dir))
            self.lm = self.model.language_model
            self.trunk_module = self.lm.model
        else:
            from mlx_lm.utils import load_model

            self.model, _ = load_model(pathlib.Path(model_dir))
            self.lm = self.model
            self.trunk_module = getattr(self.model, "model", None)
        mx.eval(self.model.parameters())
        self.layers = list(self.lm.layers)
        self.cache = None

    def new_cache(self):
        if hasattr(self.lm, "make_cache"):
            self.cache = self.lm.make_cache()
        else:
            from mlx_lm.models.cache import make_prompt_cache

            self.cache = make_prompt_cache(self.lm)
        if self.vlm:
            # mlx_vlm keeps mRoPE position state on the language model.
            self.lm._position_ids = None
            self.lm._rope_deltas = None

    def cache_arrays(self):
        out = []

        def collect(v):
            if isinstance(v, mx.array):
                out.append(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    collect(x)

        for c in self.cache:
            if c is None:
                continue
            try:
                collect(c.state)
            except Exception:
                pass
        return out

    def logits(self, ids2d):
        out = self.lm(ids2d, cache=self.cache)
        return getattr(out, "logits", out)

    def step_full(self, y):
        return mx.argmax(self.logits(y[:, None])[:, -1, :], axis=-1).astype(mx.int32)

    def step_trunk(self, y):
        if self.vlm:
            out = self.lm(y[:, None], cache=self.cache, skip_logits=True, return_hidden=True)
            hidden = out.hidden_states[-1]
        else:
            hidden = self.trunk_module(y[:, None], cache=self.cache)
        # A token that depends on the trunk output, so the step is evaluated.
        return (hidden[:, -1, 0] > 0).astype(mx.int32) + 1000

    def step_fused(self, y):
        tok = self.lm.fused_greedy_decode(y[:, None], cache=self.cache)
        if tok is None:
            return None
        if tok.ndim == 2:
            tok = tok[:, 0]
        return tok.astype(mx.int32)

    def prefill(self, ids: list[int], chunk: int):
        self.new_cache()
        t0 = time.perf_counter()
        rates = []
        for i in range(0, len(ids), chunk):
            part = mx.array(ids[i : i + chunk], dtype=mx.int32)[None]
            tc = time.perf_counter()
            if self.vlm:
                self.lm(part, cache=self.cache, skip_logits=True)
            elif self.trunk_module is not None:
                self.trunk_module(part, cache=self.cache)
            else:
                self.lm(part, cache=self.cache)
            mx.eval(self.cache_arrays())
            rates.append(part.shape[1] / (time.perf_counter() - tc))
            mx.clear_cache()
        return time.perf_counter() - t0, rates


def loop_sync(step, y, n):
    total, build = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        nxt = step(y)
        t1 = time.perf_counter()
        mx.eval(nxt)
        t2 = time.perf_counter()
        total.append((t2 - t0) * 1e3)
        build.append((t1 - t0) * 1e3)
        y = nxt
    return y, total, build


def loop_pipelined(step, y, n):
    nxt = step(y)
    mx.async_eval(nxt)
    total = []
    for _ in range(n):
        t0 = time.perf_counter()
        ahead = step(nxt)
        mx.async_eval(ahead)
        nxt.item()
        total.append((time.perf_counter() - t0) * 1e3)
        nxt = ahead
    mx.eval(nxt)
    return nxt, total


class GpuUtil:
    """Samples the GPU busy percentage (IOAccelerator PerformanceStatistics)
    while a loop runs. A busy share well under 100 means the step is bound by
    command encoding on the CPU side rather than by the kernels themselves."""

    def __init__(self, period=0.2):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                m = re.search(r'"Device Utilization %"=(\d+)', out)
                if m:
                    self.samples.append(int(m.group(1)))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=6)

    def median(self):
        return statistics.median(self.samples) if self.samples else None


def med(xs):
    return round(statistics.median(xs), 3)


def summarize(xs):
    xs = sorted(xs)
    return {
        "median_ms": med(xs),
        "p10_ms": round(xs[len(xs) // 10], 3),
        "p90_ms": round(xs[(len(xs) * 9) // 10], 3),
        "tok_s": round(1000.0 / statistics.median(xs), 2),
    }


class KnockOut:
    """Shadow a module attribute with a pass-through for the duration."""

    def __init__(self, targets):
        self.targets = targets  # [(owner, attr, replacement)]

    def __enter__(self):
        for owner, attr, fn in self.targets:
            object.__setattr__(owner, attr, fn)
        return self

    def __exit__(self, *exc):
        for owner, attr, _ in self.targets:
            object.__delattr__(owner, attr)


class PassThrough:
    """Returns its first argument; attribute reads reach the real module
    (model forwards inspect attention attributes such as ``rotary_emb``)."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, x, *a, **k):
        return x

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SwitchPassThrough(PassThrough):
    """Stands in for a routed-expert block: [..., k, hidden] from [..., hidden]."""

    def __call__(self, x, inds, *a, **k):
        shape = (*x.shape[:-1], inds.shape[-1], x.shape[-1])
        return mx.broadcast_to(x[..., None, :], shape)


def knockout_families(eng: Engine):
    fams = collections.OrderedDict()

    def add(name, owner, attr, kind=PassThrough):
        inner = getattr(owner, attr, None)
        if inner is not None:
            fams.setdefault(name, []).append((owner, attr, kind(inner)))

    for layer in eng.layers:
        add("linear_attn (DeltaNet)", layer, "linear_attn")
        add("self_attn (attention)", layer, "self_attn")
        add("mlp (whole block)", layer, "mlp")
        add("norms (layer RMSNorm x2)", layer, "input_layernorm")
        add("norms (layer RMSNorm x2)", layer, "post_attention_layernorm")
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            add("mlp.switch_mlp (routed experts)", mlp, "switch_mlp", SwitchPassThrough)
            add("mlp.shared_expert", mlp, "shared_expert")
    return fams


def graph_census(eng: Engine, y):
    """Primitive counts for one lazy decode step (the step is then evaluated)."""
    nxt = eng.step_full(y)
    counts = collections.Counter()
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "step.dot")
            mx.export_to_dot(path, nxt)
            for line in open(path):
                m = re.search(r'label\s*=\s*"?([A-Za-z_][A-Za-z0-9_]*)', line)
                if m and "->" not in line:
                    counts[m.group(1)] += 1
    except Exception as e:  # export is a diagnostic; never fail the run on it
        counts = collections.Counter({"export_failed: " + str(e)[:80]: 1})
    mx.eval(nxt)
    return nxt, dict(counts.most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="24,8192,32768")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk tokens")
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--no-knockouts", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    mx.set_cache_limit(int(a.cache_limit_gb * 2**30))
    model_dir = resolve_model_dir(a.model)
    t0 = time.perf_counter()
    eng = Engine(model_dir)
    load_s = time.perf_counter() - t0
    print(f"loaded {a.model} in {load_s:.1f}s vlm={eng.vlm} layers={len(eng.layers)} "
          f"active={mx.get_active_memory() / 2**30:.2f} GB", flush=True)

    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    all_ids = prompt_ids(model_dir, max(depths))
    fams = knockout_families(eng)
    result = {
        "model": a.model, "vlm": eng.vlm, "layers": len(eng.layers),
        "mlx": mx.__version__, "steps": a.steps, "warmup": a.warmup, "chunk": a.chunk,
        "cache_limit_gb": a.cache_limit_gb, "load_s": round(load_s, 1), "depths": [],
    }

    for depth in depths:
        row = {"depth": depth}
        pre_s, rates = eng.prefill(all_ids[:depth], a.chunk)
        row["prefill_s"] = round(pre_s, 2)
        row["prefill_tok_s"] = round(depth / pre_s, 1)
        row["prefill_chunk_tok_s_first_last"] = [round(rates[0], 1), round(rates[-1], 1)]
        row["active_gb_after_prefill"] = round(mx.get_active_memory() / 2**30, 2)
        y = mx.array([all_ids[depth - 1]], dtype=mx.int32)
        print(f"\n== depth {depth}: prefill {pre_s:.1f}s ({depth / pre_s:.0f} tok/s) "
              f"active {row['active_gb_after_prefill']} GB", flush=True)

        y, _ = loop_pipelined(eng.step_full, y, a.warmup)
        y, counts = graph_census(eng, y)
        row["graph_primitives"] = counts
        row["graph_primitive_total"] = sum(counts.values())

        y, total, build = loop_sync(eng.step_full, y, a.steps)
        row["sync"] = summarize(total)
        row["sync"]["build_median_ms"] = med(build)
        row["sync"]["eval_median_ms"] = round(row["sync"]["median_ms"] - med(build), 3)

        y, total = loop_pipelined(eng.step_full, y, a.steps)
        row["pipelined"] = summarize(total)

        with GpuUtil() as gu:
            y, total = loop_pipelined(eng.step_full, y, max(a.steps * 3, 200))
        row["gpu_busy_pct_pipelined"] = gu.median()
        row["pipelined_while_sampled_ms"] = med(total)

        y, total = loop_pipelined(eng.step_trunk, y, a.steps)
        row["trunk_only"] = summarize(total)
        row["head_cost_ms"] = round(row["pipelined"]["median_ms"] - row["trunk_only"]["median_ms"], 3)

        if eng.vlm and callable(getattr(eng.lm, "fused_greedy_decode", None)):
            probe = eng.step_fused(y)
            if probe is not None:
                mx.eval(probe)
                y, total = loop_pipelined(eng.step_fused, probe, a.steps)
                row["fused_greedy"] = summarize(total)

        print(f"  sync       {row['sync']['median_ms']:.2f} ms  (build {row['sync']['build_median_ms']:.2f} "
              f"+ eval {row['sync']['eval_median_ms']:.2f})  {row['sync']['tok_s']} tok/s")
        print(f"  pipelined  {row['pipelined']['median_ms']:.2f} ms  {row['pipelined']['tok_s']} tok/s")
        print(f"  trunk only {row['trunk_only']['median_ms']:.2f} ms  (head {row['head_cost_ms']:.2f} ms)")
        if "fused_greedy" in row:
            print(f"  fused head {row['fused_greedy']['median_ms']:.2f} ms  {row['fused_greedy']['tok_s']} tok/s")
        print(f"  graph      {row['graph_primitive_total']} primitives/step   GPU busy "
              f"{row['gpu_busy_pct_pipelined']}% (step {row['pipelined_while_sampled_ms']:.2f} ms while sampled)",
              flush=True)

        if not a.no_knockouts:
            base = row["pipelined"]["median_ms"]
            base_build = row["sync"]["build_median_ms"]
            kos = {}
            for name, targets in fams.items():
                with KnockOut(targets):
                    y, _ = loop_pipelined(eng.step_full, y, 8)
                    y, total = loop_pipelined(eng.step_full, y, a.steps)
                    y, _, build = loop_sync(eng.step_full, y, a.steps // 2)
                kos[name] = {
                    "modules": len(targets),
                    "step_ms_without": med(total),
                    "cost_ms": round(base - med(total), 3),
                    "cost_pct": round(100 * (base - med(total)) / base, 1),
                    "build_cost_ms": round(base_build - med(build), 3),
                }
                print(f"  -{name:34s} {kos[name]['cost_ms']:6.2f} ms ({kos[name]['cost_pct']:5.1f}%)  "
                      f"build {kos[name]['build_cost_ms']:.2f} ms  n={len(targets)}", flush=True)
            everything = [t for name, ts in fams.items()
                          if not name.startswith("mlp.") for t in ts]
            with KnockOut(everything):
                y, _ = loop_pipelined(eng.step_full, y, 8)
                y, total = loop_pipelined(eng.step_full, y, a.steps)
                y, _, build = loop_sync(eng.step_full, y, a.steps // 2)
            row["floor_embed_head_loop"] = {"step_ms": med(total), "build_ms": med(build)}
            print(f"  floor (embed + final norm + head + loop) {med(total):.2f} ms  build {med(build):.2f} ms")
            # Drift check: the untouched step again after the knock-outs.
            y, _ = loop_pipelined(eng.step_full, y, 8)
            y, total = loop_pipelined(eng.step_full, y, a.steps)
            row["pipelined_recheck"] = summarize(total)
            print(f"  pipelined recheck {row['pipelined_recheck']['median_ms']:.2f} ms", flush=True)
            row["knockouts"] = kos

        row["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
        result["depths"].append(row)
        eng.cache = None
        mx.clear_cache()

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=2)
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
