#!/usr/bin/env python3
"""CPU unit test for patch 039: an auxiliary-state snapshot schedules the
evaluation of each layer's state and waits for none of it, holds the state the
caches had when it was taken, and restores it.

A prefill chunk hands the snapshot lazy cache state. A synchronous evaluation
per layer runs the chunk's forward as that many submit-and-wait cycles; the
scheduled form submits the same work and leaves the wait to the chunk's token.

    python scripts/eval/test_snapshot_async_eval_unit.py
"""
import sys

import mlx.core as mx

import sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state as aux
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    WindowedAttentionKVCache,
)

mx.set_default_device(mx.cpu)
W, H, D = 513, 4, 8
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


class Evals:
    """Records the evaluations the snapshot module makes: ``waited`` holds the
    array count of every synchronous ``mx.eval``, ``scheduled`` of every
    ``mx.async_eval``."""

    def __init__(self):
        self.waited = []
        self.scheduled = []
        self._eval = mx.eval
        self._async_eval = mx.async_eval

    def __enter__(self):
        def waited(*arrays):
            self.waited.append(len(arrays))
            return self._eval(*arrays)

        def scheduled(*arrays):
            self.scheduled.append(len(arrays))
            return self._async_eval(*arrays)

        aux.mx.eval = waited
        aux.mx.async_eval = scheduled
        return self

    def __exit__(self, *exc):
        aux.mx.eval = self._eval
        aux.mx.async_eval = self._async_eval


class LayerCache:
    """Shape of an mlx-vlm ArraysCache: a list the model rebinds each step."""

    def __init__(self, state):
        self.cache = state
        self.lengths = None
        self.left_padding = None

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value

    @property
    def meta_state(self):
        return ("meta",)

    @meta_state.setter
    def meta_state(self, value):
        pass


def same(a, b):
    return a.shape == b.shape and bool(mx.all(a == b))


def pool_and_slot():
    pool = aux.MlxAuxiliaryStatePool(size=4, device="cpu")
    return pool, pool.alloc(1)


# --- every layer scheduled, none awaited ------------------------------------
# Each layer's state is built on the layer before it, like a forward.
N = 35
caches, x = [], mx.ones((4,))
for i in range(N):
    x = x * 2.0 + float(i)
    caches.append(LayerCache([x, x + 1.0]))
want = [[a * 1.0 for a in c.cache] for c in caches]
pool, slot = pool_and_slot()
with Evals() as ev:
    pool.store_cache(slot, caches, range(N))
check(ev.waited == [] and ev.scheduled == [2] * N,
      f"35 layers of lazy state: no wait, one scheduled evaluation per layer ({len(ev.scheduled)}, {ev.waited})")

with Evals() as ev:
    pool.store_cache(slot, caches, range(0, N, 5))
check(ev.waited == [] and ev.scheduled == [2] * 7,
      f"a subset of the layers: only its layers scheduled ({ev.scheduled}, {ev.waited})")

with Evals() as ev:
    pool.store_cache(slot, caches, [])
check(ev.waited == [] and ev.scheduled == [] and pool.has_snapshot(slot),
      f"no layers: no evaluation, an empty snapshot ({ev.scheduled}, {ev.waited})")

empty = [WindowedAttentionKVCache(W) for _ in range(3)]
with Evals() as ev:
    pool.store_cache(slot, empty, range(3))
check(ev.waited == [] and ev.scheduled == [],
      f"caches that hold no arrays: no evaluation ({ev.scheduled}, {ev.waited})")

with Evals() as ev:
    snap = aux._snapshot_cache(caches[0])
check(ev.waited == [] and ev.scheduled == [2], "a single layer snapshot schedules its two arrays and returns")

# --- the snapshot holds the state of the moment it was taken ----------------
pool, slot = pool_and_slot()
pool.store_cache(slot, caches, range(N))
for c in caches:
    c.cache[0] = c.cache[0] * 0.0 - 1.0  # the next step rebinds an entry
    c.cache[1][0] = 9.0  # ... or writes one in place
fresh = [LayerCache([mx.zeros((4,)), mx.zeros((4,))]) for _ in range(N)]
check(pool.restore_cache(slot, fresh), "restore finds the snapshot")
ok = all(same(f.cache[0], w[0]) and same(f.cache[1], w[1]) for f, w in zip(fresh, want))
check(ok, "restored state equals the state at store time, after rebinds and in-place writes")

lazy = mx.zeros((2,)) + 3.0
pinned = LayerCache([lazy])
snap = aux._snapshot_cache(pinned)
pinned.cache[0] = lazy * 2.0
check(snap.state[0].tolist() == [3.0, 3.0], "a snapshot read before its evaluation ran is pinned too")

other = pool.alloc(1)
pool.copy_from(slot, other)
copied = [LayerCache([mx.zeros((4,)), mx.zeros((4,))]) for _ in range(N)]
pool.restore_cache(other, copied)
ok = all(same(f.cache[0], w[0]) and same(f.cache[1], w[1]) for f, w in zip(copied, want))
check(ok, "a copied snapshot restores the same state")

part = [LayerCache([mx.zeros((4,)), mx.zeros((4,))]) for _ in range(N)]
pool.restore_cache(slot, part, layer_indices=[0, 3])
check(same(part[3].cache[0], want[3][0]) and part[1].cache[0].tolist() == [0.0] * 4,
      "restore of selected layers leaves the others alone")


# --- sliding-window caches written lazily, as a prefill chunk leaves them ---
def chunk(n, seed):
    k = mx.random.normal((1, H, n, D), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, H, n, D), key=mx.random.key(seed + 1000)).astype(mx.bfloat16)
    return k, v


L = 5
live = [WindowedAttentionKVCache(W) for _ in range(L)]
stream = [[] for _ in range(L)]
for layer in range(L):
    for i, n in enumerate((1024, 1024)):
        k, v = chunk(n, 10 * layer + i)
        live[layer].update_and_fetch(k, v)  # left lazy: nothing evaluated yet
        stream[layer].append((k, v))
pool, slot = pool_and_slot()
with Evals() as ev:
    pool.store_cache(slot, live, range(L))
check(ev.waited == [] and len(ev.scheduled) == L,
      f"{L} lazily written sliding-window caches: scheduled per layer, none awaited ({ev.scheduled}, {ev.waited})")

for layer in range(L):
    k, v = chunk(300, 500 + layer)
    live[layer].update_and_fetch(k, v)
    for t in range(3):
        live[layer].write_token(*chunk(1, 700 + 10 * layer + t))

restored = [WindowedAttentionKVCache(W) for _ in range(L)]
pool.restore_cache(slot, restored)
ok = True
for layer in range(L):
    ref_k = mx.concatenate([k for k, _ in stream[layer]], axis=2)[:, :, 2048 - W :, :]
    ref_v = mx.concatenate([v for _, v in stream[layer]], axis=2)[:, :, 2048 - W :, :]
    rk, rv = restored[layer].get_kv(W)
    ok = ok and restored[layer].offset == 2048 and same(rk, ref_k) and same(rv, ref_v)
check(ok, "restored windows equal the stream tail at the boundary, after later live writes")

for layer in range(L):
    k, v = chunk(300, 500 + layer)
    restored[layer].update_and_fetch(k, v)
    for t in range(3):
        restored[layer].write_token(*chunk(1, 700 + 10 * layer + t))
ok = True
for layer in range(L):
    lk, lv = live[layer].get_kv(W)
    rk, rv = restored[layer].get_kv(W)
    ok = ok and restored[layer].offset == live[layer].offset and same(rk, lk) and same(rv, lv)
check(ok, "restored caches replay the continuation to the live state")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
