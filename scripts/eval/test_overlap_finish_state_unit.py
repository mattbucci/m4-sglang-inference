#!/usr/bin/env python3
"""CPU unit test for patch 028: a hybrid request that finishes under the
pipelined (overlap) decode loop donates the auxiliary state that belongs to
its radix key.

Covers the three pieces: the frozen view a chained step takes of the state it
was built on, the runner's refusal to store a state whose length is not the
key's, and the release hook's commit of the decode tokens chained steps never
passed through the scheduler.

    python scripts/eval/test_overlap_finish_state_unit.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx
import torch

import sglang.srt.hardware_backend.mlx.tp_worker as tp_worker
import sglang.srt.mem_cache.common as mem_common
from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import FrozenCacheView
from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# --- the frozen view keeps the state a chained step was built on -------------
class LayerCache:
    """Shape of an mlx-vlm ArraysCache: a list the model rebinds each step."""

    def __init__(self):
        self.cache = [mx.zeros((2,)), mx.ones((2,))]
        self.lengths = None
        self.left_padding = None

    @property
    def state(self):
        return self.cache

    @property
    def meta_state(self):
        return ("meta",)


live = LayerCache()
view = FrozenCacheView(live)
live.cache[0] = mx.full((2,), 7.0)  # the next step rebinds an entry
live.cache[1][0] = 9.0  # ... or writes one in place
check(view.state[0].tolist() == [0.0, 0.0], "frozen view: a rebound entry keeps its old value")
check(view.state[1].tolist() == [1.0, 1.0], "frozen view: an in-place write does not reach it")
check(view.meta_state == ("meta",) and view.lengths is None, "frozen view: meta state and attributes carried")
lazy = mx.zeros((2,)) + 3.0
live.cache[0] = lazy
view = FrozenCacheView(live)
live.cache[0] = lazy * 2.0
check(view.state[0].tolist() == [3.0, 3.0], "frozen view: a lazy (unevaluated) entry is pinned too")


# --- the runner stores only a state whose length is the key's ----------------
def runner(n_tokens, lookahead=None):
    r = SimpleNamespace(
        _has_snapshot_state=True,
        _req_pool_idx={"a": 5},
        _req_caches={"a": "LIVE"},
        _req_token_ids={"a": list(range(n_tokens))},
        _lookahead_aux_base={} if lookahead is None else {"a": (1, lookahead)},
        _lookup_rows=0,
        stored=[],
    )
    r._store_auxiliary_state = lambda idx, cache: r.stored.append((idx, cache))
    return r


store = MlxModelRunner.store_auxiliary_state_for_request

# 100 prompt tokens + 4 sampled: the stored state has consumed 103 tokens.
r = runner(104)
check(store(r, "a", 103) is True and r.stored == [(5, "LIVE")], "no lookahead: matching key stores the live state")
r = runner(104, lookahead="BASE")
check(store(r, "a", 103) is True and r.stored == [(5, "BASE")],
      "lookahead in flight: matching key stores the pre-lookahead state")
r = runner(104, lookahead="BASE")
check(store(r, "a", 101) is False and r.stored == [], "short key (chained tokens not committed): nothing stored")
r = runner(104)
check(store(r, "a", 100) is False and r.stored == [], "prompt-only key: nothing stored")
r = runner(104)
r._has_snapshot_state = False
check(store(r, "a", 0) is True and r.stored == [], "model without auxiliary state: nothing to match")
r = runner(104)
r._req_caches = {}
check(store(r, "a", 103) is False, "request the runner dropped: reports no state")


# --- the release hook commits chained tokens, or skips the insert ------------
class Pool:
    def __init__(self):
        self.writes = []

    def write(self, index, values):
        self.writes.append((index, values.tolist()))


def request(prompt, n_out, fresh, skip=False):
    q = SimpleNamespace(
        rid="a",
        origin_input_ids=list(range(prompt)),
        output_ids=list(range(n_out)),
        skip_radix_cache_insert=skip,
        kv=SimpleNamespace(
            req_pool_idx=3,
            kv_allocated_len=prompt + fresh,
            kv_committed_len=prompt + fresh,
            mamba_last_track_seqlen=64,
        ),
    )
    q.owned_kv_len = lambda: q.kv.kv_committed_len
    return q


class HookRunner:
    disable_radix_cache = False

    def __init__(self, state_len):
        self.state_len = state_len
        self.calls = []

    def has_request(self, rid):
        return True

    def store_auxiliary_state_for_request(self, rid, key_len=None):
        self.calls.append(("store", key_len))
        return key_len == self.state_len

    def release_request_row(self, rid, owned_len):
        self.calls.append(("release", owned_len))

    def remove_request(self, rid):
        self.calls.append(("remove",))


def worker(r):
    return SimpleNamespace(
        _mlx_runner=r,
        _mlx_active_rids={"a"},
        _req_retraction_count={},
        _inflight_launches={},
        _remove_after_inflight=set(),
        _commit_chained_decode_tokens=lambda req, tree: MlxTpModelWorker._commit_chained_decode_tokens(
            None, req, tree
        ),
    )


tp_worker.get_serving = lambda: SimpleNamespace(strip_thinking_cache=False)
allocs, evictions = [], []
mem_common.evict_from_tree_cache = lambda tree_cache, n: evictions.append(n)
hook = MlxTpModelWorker.prepare_for_kv_cache_release


class Allocator:
    def __init__(self, full, page_size):
        self.full, self.page_size = full, page_size

    def alloc(self, n):
        allocs.append(n)
        return None if self.full else torch.arange(900, 900 + n)


def tree(full=False, page_size=1):
    return SimpleNamespace(req_to_token_pool=Pool(), token_to_kv_pool_allocator=Allocator(full, page_size))


# 100 prompt tokens, 6 outputs, one fresh decode launch: 105 tokens are
# committed in the runner, the scheduler owns 101.
os.environ["SGLANG_MLX_COMMIT_CHAINED"] = "1"
allocs.clear(); r = HookRunner(105); t = tree(); q = request(100, 6, fresh=1); hook(worker(r), q, t)
check(allocs == [4] and evictions[-1:] == [4]
      and t.req_to_token_pool.writes == [((3, slice(101, 105)), [900, 901, 902, 903])],
      f"commit: four chained tokens get slots at rows 101..104 ({allocs}, {t.req_to_token_pool.writes})")
check(q.kv.kv_allocated_len == 105 and q.kv.kv_committed_len == 105, "commit: allocated and committed advance together")
check(r.calls[:2] == [("store", 105), ("release", 105)] and not q.skip_radix_cache_insert,
      f"commit: state stored and row synced at the full key ({r.calls})")

allocs.clear(); r = HookRunner(105); t = tree(); q = request(100, 6, fresh=5); hook(worker(r), q, t)
check(allocs == [] and not q.skip_radix_cache_insert, "every step fresh (normal loop): nothing to commit, insert kept")

allocs.clear(); r = HookRunner(105); t = tree(full=True); q = request(100, 6, fresh=1); hook(worker(r), q, t)
check(allocs == [4] and q.kv.kv_committed_len == 101 and q.skip_radix_cache_insert,
      "pool exhausted: lengths untouched, insert skipped instead of keying a mismatched state")

allocs.clear(); r = HookRunner(105); t = tree(page_size=16); q = request(100, 6, fresh=1); hook(worker(r), q, t)
check(allocs == [] and q.skip_radix_cache_insert, "paged allocator: no single-slot commit, insert skipped")

allocs.clear(); r = HookRunner(105); t = tree(); q = request(100, 6, fresh=1, skip=True); hook(worker(r), q, t)
check(allocs == [] and q.kv.kv_committed_len == 101, "insert already skipped: no slots taken")

tp_worker.get_serving = lambda: SimpleNamespace(strip_thinking_cache=True)
allocs.clear(); r = HookRunner(105); t = tree(); q = request(100, 6, fresh=1); hook(worker(r), q, t)
check(allocs == [], "thinking-strip caching (prompt-only key): no slots taken")
tp_worker.get_serving = lambda: SimpleNamespace(strip_thinking_cache=False)

os.environ["SGLANG_MLX_COMMIT_CHAINED"] = "0"
allocs.clear(); r = HookRunner(105); t = tree(); q = request(100, 6, fresh=1); hook(worker(r), q, t)
check(allocs == [] and q.skip_radix_cache_insert and r.calls[:2] == [("store", 101), ("release", 101)],
      f"commit off: short key, insert skipped ({r.calls})")

allocs.clear(); r = HookRunner(105); q = request(100, 6, fresh=5); hook(worker(r), q)
check(not q.skip_radix_cache_insert, "hook without a tree cache argument still runs")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
