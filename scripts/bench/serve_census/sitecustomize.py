"""Serving-step census for the MLX overlap decode loop.

Put this directory on PYTHONPATH when launching the server; the hook is active
only when SERVE_CENSUS_OUT is set, and edits no component file:

    PYTHONPATH=scripts/bench/serve_census SERVE_CENSUS_OUT=/tmp/serve.jsonl \
        scripts/launch.sh qwen36

It wraps the phases of one scheduler iteration with perf_counter_ns
accumulators and writes one JSON line per SERVE_CENSUS_WINDOW (default 64)
chained decode iterations: sums in ns, inclusive of nested phases. Only pure
chained iterations count (one chained launch, one finalize, no fresh launch).

Nesting: finalize_job > finalize_result > token_wait, and finalize_job >
process_result > result_decode > stream_output; launch_chained > graph_build
> (fwd, select) and launch_chained > async_eval. `async_eval` is the encode of
the next step and blocks while more than ten command buffers are in flight,
so it carries the GPU time of the step; the rest of the iteration is host
time during which the GPU runs only what was queued when it returned.

SERVE_CENSUS_PROFILE=<path> adds a cProfile of the graph build over
iterations 200-455 (pstats dump); profiled iterations run slower, so read
phase times from a run without it. Summarise with serve_census_report.py.

SERVE_CENSUS_GRAPH=<path> writes the primitive histogram of one launch (the
400th `mx.async_eval`, a chained decode step in any run past a few hundred
tokens) as JSON, for comparison with the raw-MLX step graph that
mlx_decode_census.py counts.

SERVE_CENSUS_PREFILL=<jsonl> (alone or with SERVE_CENSUS_OUT) writes one row
per prefill chunk launch: `gap_ms` from the previous chunk's token to this
launch (scheduler time, GPU idle), `launch_ms` for the graph build and its
submission (`async_eval_ms` of it inside `mx.async_eval`, `trunk_ms`,
`pool_sync_ms` and `track_ms` building the forward, the pool scatter and the
tracked-state snapshot), and `wait_ms` blocking on the chunk's token. The
snapshot evaluates synchronously, so with the radix cache on the chunk's
forward runs inside `track_ms`; a raw chunked prefill of the same depth spends
everything in the equivalent of `track_ms + async_eval_ms + wait_ms`. `evals`
counts the synchronous `mx.eval` calls of the launch, each one a submission
the launch waits for. Summarise with prefill_census_report.py.
"""
import os
import sys

_OUT = os.environ.get("SERVE_CENSUS_OUT")
_PRE = os.environ.get("SERVE_CENSUS_PREFILL")
if _OUT or _PRE:
    import importlib.abc
    import importlib.util
    import json
    import time

    _ns = time.perf_counter_ns
    _WINDOW = int(os.environ.get("SERVE_CENSUS_WINDOW", "64"))
    _PROF = os.environ.get("SERVE_CENSUS_PROFILE")
    _GRAPH = os.environ.get("SERVE_CENSUS_GRAPH")
    _GRAPH_AT = int(os.environ.get("SERVE_CENSUS_GRAPH_AT", "400"))
    _cur = {}          # label -> ns in the current iteration
    _win = {}          # label -> ns in the current window
    _state = {"t0": None, "n": 0, "seq": 0}
    _pre = {"last_end": None, "open": None, "acc": {}, "evals": 0}

    def _pre_wrap(obj, name, label):
        f = getattr(obj, name)

        def w(*a, **k):
            t = _ns()
            try:
                return f(*a, **k)
            finally:
                if _pre["open"] is not None:
                    _pre["acc"][label] = _pre["acc"].get(label, 0) + _ns() - t

        w.__name__ = getattr(f, "__name__", name)
        setattr(obj, name, w)

    def _add(label, d):
        _cur[label] = _cur.get(label, 0) + d

    def _wrap(obj, name, label):
        f = getattr(obj, name)

        def w(*a, **k):
            t = _ns()
            try:
                return f(*a, **k)
            finally:
                _add(label, _ns() - t)

        w.__name__ = getattr(f, "__name__", name)
        setattr(obj, name, w)

    def _dump_graph(mx, args):
        import collections
        import re
        import tempfile

        counts = collections.Counter()
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "step.dot")
                mx.export_to_dot(path, *args)
                for line in open(path):
                    m = re.search(r'label\s*=\s*"?([A-Za-z_][A-Za-z0-9_]*)', line)
                    if m and "->" not in line:
                        counts[m.group(1)] += 1
            row = {"total": sum(counts.values()), "primitives": dict(counts.most_common())}
        except Exception as e:  # diagnostic only
            row = {"error": repr(e)[:200]}
        with open(_GRAPH, "w") as fh:
            json.dump(row, fh)

    def _close_iteration(now):
        t0 = _state["t0"]
        _state["t0"] = now
        if t0 is None:
            _cur.clear()
            return
        # A pure chained decode iteration: one chained launch, one finalize,
        # no fresh launch.
        if _cur.get("#chained") == 1 and "#fresh" not in _cur and _cur.get("#finalize") == 1:
            _win["wall"] = _win.get("wall", 0) + (now - t0)
            for k, v in _cur.items():
                if not k.startswith("#"):
                    _win[k] = _win.get(k, 0) + v
            _state["n"] += 1
            if _state["n"] >= _WINDOW:
                _state["seq"] += 1
                row = {"seq": _state["seq"], "t": time.time(), "n": _state["n"], "pid": os.getpid()}
                row.update({k: v for k, v in _win.items()})
                if _OUT:
                    with open(_OUT, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
                _win.clear()
                _state["n"] = 0
        _cur.clear()

    def _patch_prefill_worker(cls):
        launch = cls._async_extend_batch
        finalize = cls.finalize_mlx_result

        def extend(self, batch):
            t = _ns()
            _pre["open"], _pre["acc"], _pre["evals"] = t, {}, 0
            try:
                return launch(self, batch)
            finally:
                end = _ns()
                acc, last = _pre["acc"], _pre["last_end"]
                _pre["open"] = None
                _pre["row"] = {
                    "t": time.time(), "pid": os.getpid(),
                    "tokens": int(sum(batch.extend_lens)), "reqs": len(batch.reqs),
                    "gap_ms": None if last is None else round((t - last) / 1e6, 3),
                    "launch_ms": round((end - t) / 1e6, 3), "evals": _pre["evals"],
                    **{k + "_ms": round(v / 1e6, 3) for k, v in acc.items()},
                }

        def fin(self, launch_obj, reqs):
            row = _pre.pop("row", None) if launch_obj.mode == "extend" else None
            t = _ns()
            try:
                return finalize(self, launch_obj, reqs)
            finally:
                end = _ns()
                if row is not None:
                    _pre["last_end"] = end
                    row["wait_ms"] = round((end - t) / 1e6, 3)
                    with open(_PRE, "a") as fh:
                        fh.write(json.dumps(row) + "\n")

        cls._async_extend_batch = extend
        cls.finalize_mlx_result = fin

    def _patch(name, module):
        if name == "sglang.srt.managers.scheduler":
            cls = module.Scheduler
            f = cls.ingest_requests

            def ingest(self, *a, **k):
                t = _ns()
                _close_iteration(t)
                try:
                    return f(self, *a, **k)
                finally:
                    _add("ingest", _ns() - t)

            cls.ingest_requests = ingest
            _wrap(cls, "process_batch_result", "process_result")
        elif name == "sglang.srt.hardware_backend.mlx.scheduler_mixin":
            cls = module.SchedulerMlxOverlapMixin
            f2 = cls._finalize_mlx_pending_job

            def fin(self, pending):
                t = _ns()
                try:
                    return f2(self, pending)
                finally:
                    _add("finalize_job", _ns() - t)
                    _cur["#finalize"] = _cur.get("#finalize", 0) + 1

            cls._finalize_mlx_pending_job = fin
        elif name == "sglang.srt.hardware_backend.mlx.tp_worker":
            cls = module.MlxTpModelWorker
            f3 = cls.async_chained_decode_mlx

            def chained(self, prev):
                t = _ns()
                try:
                    return f3(self, prev)
                finally:
                    _add("launch_chained", _ns() - t)
                    _cur["#chained"] = _cur.get("#chained", 0) + 1

            cls.async_chained_decode_mlx = chained
            f4 = cls.async_forward_batch_generation_mlx

            def fresh(self, batch):
                _cur["#fresh"] = 1
                return f4(self, batch)

            cls.async_forward_batch_generation_mlx = fresh
            _wrap(cls, "finalize_mlx_result", "finalize_result")
            if _PRE:
                _patch_prefill_worker(cls)
            mx = module.mx
            g = mx.async_eval

            calls = {"n": 0}

            def async_eval(*a, **k):
                if _GRAPH:
                    calls["n"] += 1
                    if calls["n"] == _GRAPH_AT:
                        _dump_graph(mx, a)
                t = _ns()
                try:
                    return g(*a, **k)
                finally:
                    d = _ns() - t
                    _add("async_eval", d)
                    if _pre["open"] is not None:
                        _pre["acc"]["async_eval"] = _pre["acc"].get("async_eval", 0) + d

            mx.async_eval = async_eval
            if _PRE:
                sync = mx.eval

                def sync_eval(*a, **k):
                    if _pre["open"] is not None:
                        _pre["evals"] += 1
                    return sync(*a, **k)

                mx.eval = sync_eval
        elif name == "sglang.srt.hardware_backend.mlx.model_runner":
            cls = module.MlxModelRunner
            for attr, label in (
                ("_decode_with_native_cache", "fwd"),
                ("_decode_with_hybrid_batching", "fwd"),
                ("_decode_with_batched_attention", "fwd"),
                ("_select_tokens_with_logprobs", "select"),
            ):
                if hasattr(cls, attr):
                    _wrap(cls, attr, label)
            _wrap(cls, "decode_batch_start_chained", "graph_build")
            _wrap(cls, "decode_batch_finalize", "token_wait")
            if _PRE:
                _pre_wrap(cls, "_trunk_forward", "trunk")
                _pre_wrap(cls, "_sync_new_kv_to_pool", "pool_sync")
                _pre_wrap(cls, "_store_tracked_auxiliary_state", "track")
            if _PROF:
                import cProfile

                prof = cProfile.Profile()
                st = {"n": 0}
                fb = cls.decode_batch_start_chained

                def build(self, prev):
                    st["n"] += 1
                    if 200 <= st["n"] < 456:
                        prof.enable()
                        try:
                            return fb(self, prev)
                        finally:
                            prof.disable()
                            if st["n"] == 455:
                                prof.dump_stats(_PROF)
                    return fb(self, prev)

                cls.decode_batch_start_chained = build
        elif name == "sglang.srt.managers.scheduler_components.output_streamer":
            _wrap(module.SchedulerOutputStreamer, "stream_output", "stream_output")
        elif name == "sglang.srt.managers.scheduler_components.batch_result_processor":
            _wrap(module.SchedulerBatchResultProcessor, "process_batch_result_decode", "result_decode")

    _TARGETS = {
        "sglang.srt.managers.scheduler",
        "sglang.srt.hardware_backend.mlx.scheduler_mixin",
        "sglang.srt.hardware_backend.mlx.tp_worker",
        "sglang.srt.hardware_backend.mlx.model_runner",
        "sglang.srt.managers.scheduler_components.output_streamer",
        "sglang.srt.managers.scheduler_components.batch_result_processor",
    }

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig, _name=name):
                _orig(module)
                try:
                    _patch(_name, module)
                except Exception as e:  # never break the server
                    sys.stderr.write(f"[serve-census] patch {_name} failed: {e!r}\n")

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
