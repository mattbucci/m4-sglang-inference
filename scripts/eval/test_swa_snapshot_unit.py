#!/usr/bin/env python3
"""CPU unit test for patch 021: a WindowedAttentionKVCache snapshot taken at a
prefix boundary restores to the exact state a fresh cache reaches by replaying
the same tokens, and later writes to the live cache leave the snapshot alone.

    python scripts/eval/test_swa_snapshot_unit.py
"""
import sys

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    WindowedAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
    _restore_cache,
    _snapshot_cache,
)

mx.set_default_device(mx.cpu)
W, H, D = 513, 4, 8


def chunk(n, seed):
    k = mx.random.normal((1, H, n, D), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, H, n, D), key=mx.random.key(seed + 1000)).astype(mx.bfloat16)
    return k, v


def same(a, b):
    return a.shape == b.shape and bool(mx.all(a == b))


def kv(cache):
    # get_kv() returns the whole working buffer (kept window + in-flight
    # chunk); get_kv(W) is the trailing window the decode path reads.
    k, v = cache.get_kv(W)
    mx.eval(k, v)
    return k, v


fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# 1. empty cache round-trip
empty = WindowedAttentionKVCache(W)
snap0 = _snapshot_cache(empty)
tgt0 = WindowedAttentionKVCache(W)
_restore_cache(tgt0, snap0)
check(tgt0.keys is None and tgt0.offset == 0 and tgt0.get_kv is not None, "empty snapshot restores to an empty cache")

# 2. prefill in chunks, snapshot at a boundary, keep writing, restore, replay
live = WindowedAttentionKVCache(W)
stream_k, stream_v = [], []
for i, n in enumerate((1024, 1024)):
    k, v = chunk(n, i)
    live.update_and_fetch(k, v)
    stream_k.append(k)
    stream_v.append(v)
ref_k = mx.concatenate(stream_k, axis=2)
ref_v = mx.concatenate(stream_v, axis=2)
lk, lv = kv(live)
check(live.offset == 2048 and lk.shape[2] == W and live.get_kv()[0].shape[2] == 1537, f"live offset=2048, window={lk.shape[2]}, working buffer={live.get_kv()[0].shape[2]} (window + last chunk)")
check(same(lk, ref_k[:, :, 2048 - W :, :]) and same(lv, ref_v[:, :, 2048 - W :, :]), "live window matches the stream tail")

snap = _snapshot_cache(live)
sk, sv = snap.state
check(sk.shape[2] == W and snap.attrs.get("offset") == 2048, f"snapshot is compact (kept={sk.shape[2]}, offset={snap.attrs.get('offset')})")

# keep writing on the live cache: one more chunk, then single tokens
k3, v3 = chunk(300, 7)
live.update_and_fetch(k3, v3)
for t in range(5):
    tk, tv = chunk(1, 100 + t)
    live.write_token(tk, tv)
    stream_k.append(tk)
    stream_v.append(tv)
sk2, sv2 = snap.state
check(same(sk, sk2) and bool(mx.all(sk2 == ref_k[:, :, 2048 - W : 2048, :])), "later live writes do not touch the snapshot")

# restore into a fresh cache and replay the same continuation
restored = WindowedAttentionKVCache(W)
_restore_cache(restored, snap)
rk, rv = kv(restored)
check(restored.offset == 2048 and same(rk, ref_k[:, :, 2048 - W : 2048, :]) and same(rv, ref_v[:, :, 2048 - W : 2048, :]), "restored cache holds the boundary window at offset 2048")
mask_r = restored.make_mask(300, window_size=W)
mask_l = WindowedAttentionKVCache(W)
mask_l.update_and_fetch(ref_k[:, :, :2048, :], ref_v[:, :, :2048, :])
check(str(restored.make_mask(300, window_size=W)) == str(mask_l.make_mask(300, window_size=W)), "mask for the next chunk matches a replayed cache")
restored.update_and_fetch(k3, v3)
for t in range(5):
    restored.write_token(stream_k[-5 + t], stream_v[-5 + t])
rk, rv = kv(restored)
lk, lv = kv(live)
check(restored.offset == live.offset and same(rk, lk) and same(rv, lv), f"restored + replayed == live (offset {live.offset}, kept {lk.shape[2]})")

# 3. a snapshot taken mid-stream from a cache that was itself restored
snap2 = _snapshot_cache(restored)
again = WindowedAttentionKVCache(W)
_restore_cache(again, snap2)
ak, av = kv(again)
check(again.offset == live.offset and same(ak, lk) and same(av, lv), "snapshot of a restored cache round-trips too")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
