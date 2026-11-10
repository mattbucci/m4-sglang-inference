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
"""
import os
import sys

_OUT = os.environ.get("SERVE_CENSUS_OUT")
if _OUT:
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
                with open(_OUT, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                _win.clear()
                _state["n"] = 0
        _cur.clear()

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
                    _add("async_eval", _ns() - t)

            mx.async_eval = async_eval
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
