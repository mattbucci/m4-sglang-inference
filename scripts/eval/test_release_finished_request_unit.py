#!/usr/bin/env python3
"""CPU unit test for patch 023: the MLX worker's KV-release hook drops the
runner's per-request state for every finished request, and runs the decode
sync (leaf re-snapshot, tracked-slot clear, row release) only for a request
that decoded.

    python scripts/eval/test_release_finished_request_unit.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_MLX", "1")

from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


class Runner:
    """Records the runner calls the hook makes; ``known`` are live requests."""

    def __init__(self, known):
        self.known = set(known)
        self.calls = []

    def has_request(self, rid):
        return rid in self.known

    def store_auxiliary_state_for_request(self, rid):
        self.calls.append(("store", rid))

    def release_request_row(self, rid, owned_len):
        self.calls.append(("release", rid, owned_len))

    def remove_request(self, rid):
        self.calls.append(("remove", rid))
        self.known.discard(rid)


def worker(runner, rids):
    return SimpleNamespace(
        _mlx_runner=runner,
        _mlx_active_rids=set(rids),
        _req_retraction_count={rid: 0 for rid in rids},
    )


def req(rid, n_out):
    return SimpleNamespace(
        rid=rid,
        output_ids=list(range(n_out)),
        kv=SimpleNamespace(mamba_last_track_seqlen=4096),
        owned_kv_len=lambda: 1234,
    )


hook = MlxTpModelWorker.prepare_for_kv_cache_release


def pruned(w, rid):
    return rid not in w._mlx_active_rids and rid not in w._req_retraction_count


# A request its prefill finished (one output token): drop the state, nothing to sync.
r = Runner(["a"]); w = worker(r, ["a"]); q = req("a", 1); hook(w, q)
check(r.calls == [("remove", "a")], f"prefill-finished request: remove only ({r.calls})")
check(not r.has_request("a") and pruned(w, "a"), "prefill-finished request: runner and worker bookkeeping dropped")
check(q.kv.mamba_last_track_seqlen == 4096, "prefill-finished request: tracked-slot marker untouched")

# A request that decoded: snapshot the leaf, clear the tracked marker, sync the row, then drop.
r = Runner(["b"]); w = worker(r, ["b"]); q = req("b", 3); hook(w, q)
check(r.calls == [("store", "b"), ("release", "b", 1234), ("remove", "b")],
      f"decode-finished request: store, release, remove in order ({r.calls})")
check(q.kv.mamba_last_track_seqlen is None, "decode-finished request: tracked-slot marker cleared")
check(not r.has_request("b") and pruned(w, "b"), "decode-finished request: bookkeeping dropped")

# A request the runner never saw (retracted, or swept already): prune the worker sets only.
r = Runner([]); w = worker(r, ["c"]); q = req("c", 2); hook(w, q)
check(r.calls == [] and pruned(w, "c"), f"unknown request: no runner calls, worker sets pruned ({r.calls})")

# The hook is idempotent: a second call finds nothing and changes nothing.
r = Runner(["d"]); w = worker(r, ["d"]); q = req("d", 2); hook(w, q); n = len(r.calls); hook(w, q)
check(len(r.calls) == n and pruned(w, "d"), "second call: no further runner calls")

# Only the finishing request is touched: other live requests keep their state.
r = Runner(["e", "f"]); w = worker(r, ["e", "f"]); hook(w, req("e", 1))
check(r.has_request("f") and "f" in w._mlx_active_rids and "f" in w._req_retraction_count,
      "other live request untouched")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
