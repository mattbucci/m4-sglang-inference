#!/usr/bin/env python3
"""CPU unit test for the windowed lookahead view: a chained decode step keeps,
for a sliding-window layer, the cache and its offset instead of array handles,
and a request that finishes with a lookahead step in flight still donates the
window that belongs to its radix key.

Covers the cache method that reads the window one token back (plain writes,
compactions, a shrink after an oversized prefill chunk, a window that is not
full yet), the view against a frozen view taken at the same moment, a read at
any other distance, the dispatch per cache type, and a store and restore
through the auxiliary-state pool followed by further decode steps.

    python scripts/eval/test_swa_lookahead_view_unit.py
"""
import os
import random

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    WindowedAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
    FrozenCacheView,
    MlxAuxiliaryStatePool,
    WindowedLookaheadView,
    lookahead_view,
)

fails = 0
HEADS, DIM = 2, 4


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


def chunk(n, key):
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(key))
    return k, k + 100.0


def frozen(state):
    """Materialised copy of a (keys, values) state, or () for an empty one."""
    out = tuple(mx.array(a) for a in state)
    mx.eval(*out)
    return out


def same(a, b):
    if len(a) != len(b):
        return False
    return all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a, b))


def filled(window, prefill, key=0):
    cache = WindowedAttentionKVCache(window)
    for i, n in enumerate(prefill):
        cache.update_and_fetch(*chunk(n, 1000 * key + i))
    return cache


# --- the window one token back, across every kind of write -------------------
rng = random.Random(7)
cases = [
    (4, [3]),            # window not full
    (4, [4]),            # window exactly full
    (4, [9]),            # oversized prefill chunk: the first decode write shrinks
    (5, [2, 2, 2]),      # several small chunks
    (7, [20, 3]),        # oversized chunk, then a short one
    (16, [16, 16, 5]),
    (513, [600]),        # Flint's window
]
cases += [(rng.randint(2, 12), [rng.randint(1, 30) for _ in range(rng.randint(1, 3))]) for _ in range(12)]
bad = []
compactions = 0
for n, (window, prefill) in enumerate(cases):
    cache = filled(window, prefill, key=n)
    steps = 3 * window + 5 if window < 100 else 600
    for step in range(steps):
        want = frozen(cache.snapshot_state())
        want_offset = cache.offset
        pinned = FrozenCacheView(cache)
        view = lookahead_view(cache)
        local_before = cache._local
        cache.write_token(*chunk(1, 10_000 * (n + 1) + step))
        compactions += cache._local != local_before + 1
        got = cache.snapshot_state_one_token_back()
        ok = (
            same(got, want)
            and same(view.state, want)
            and same(view.state, pinned.state)
            and view.offset == want_offset == pinned.offset
            and cache.offset == want_offset + 1
        )
        if not ok:
            bad.append((window, prefill, step))
            break
check(not bad, f"window one token back equals the pre-write snapshot ({len(cases)} caches, {compactions} writes that dropped history){' ' + str(bad[:3]) if bad else ''}")

cache = WindowedAttentionKVCache(4)
cache.write_token(*chunk(1, 1))
check(cache.snapshot_state_one_token_back() == (), "first token ever: the state one token back is empty")

# --- the view is only that: one token back -----------------------------------
cache = filled(4, [6])
view = lookahead_view(cache)
check(isinstance(view, WindowedLookaheadView), "dispatch: a windowed cache gets the integer view")
check(set(WindowedLookaheadView.__slots__) == {"offset", "_cache"} and not hasattr(view, "__dict__"),
      "the view holds the cache and an integer, no array handles")
try:
    view.state
    check(False, "read before the next token: refused")
except RuntimeError:
    check(True, "read before the next token: refused")
cache.write_token(*chunk(1, 2))
check(len(view.state) == 2, "read one token later: served")
cache.write_token(*chunk(1, 3))
try:
    view.state
    check(False, "read two tokens later: refused")
except RuntimeError:
    check(True, "read two tokens later: refused")


class ArraysLike:
    def __init__(self):
        self.cache = [mx.zeros((2,)), mx.ones((2,))]
        self.lengths = None

    @property
    def state(self):
        return self.cache


check(isinstance(lookahead_view(ArraysLike()), FrozenCacheView), "dispatch: any other cache keeps the frozen view")

# --- store through the pool, restore, keep decoding --------------------------
for window, prefill in [(4, [9]), (6, [3]), (5, [5, 5])]:
    live = filled(window, prefill, key=50)
    for step in range(window + 2):
        live.write_token(*chunk(1, 500 + step))
    twin = filled(window, prefill, key=50)
    for step in range(window + 2):
        twin.write_token(*chunk(1, 500 + step))

    views = {3: lookahead_view(live)}
    live.write_token(*chunk(1, 999))  # the lookahead step, in flight at the finish
    pool = MlxAuxiliaryStatePool(size=4, device="cpu")
    pool.store_cache(2, views, (3,))
    live.write_token(*chunk(1, 1000))  # the live cache moves on; the snapshot must not

    restored = [None, None, None, WindowedAttentionKVCache(window)]
    hit = pool.restore_cache(2, restored, (3,))
    got = restored[3]
    ok = hit and got.offset == twin.offset and same(frozen(got.snapshot_state()), frozen(twin.snapshot_state()))
    for step in range(2 * window + 3):
        k, v = chunk(1, 2000 + step)
        got.write_token(k, v)
        twin.write_token(k, v)
        a, b = got.get_kv(window=window), twin.get_kv(window=window)
        ok = ok and got.offset == twin.offset and same(frozen(a), frozen(b))
    check(ok, f"window {window}, prefill {prefill}: pool snapshot of the view restores the pre-lookahead cache")

print("\nALL PASS" if not fails else f"\n{fails} FAILED")
raise SystemExit(1 if fails else 0)
