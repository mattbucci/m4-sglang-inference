#!/usr/bin/env python3
"""CPU unit test for patch 022: a PoolBackedAttentionKVCache fed two
consecutive chunks (the shape the last-position-logits split produces on a
prefix hit) returns the pool prefix followed by both chunks, exactly as one
call with the concatenated chunk and exactly as a contiguous cache replaying
the same tokens.

    python scripts/eval/test_pool_backed_cache_unit.py
"""
import sys

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    ContiguousAttentionKVCache,
    PoolBackedAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
    MlxAttentionKVPool,
)

mx.set_default_device(mx.cpu)
H, D = 4, 8
P, N1, N2 = 10, 63, 1  # prefix tokens, trunk chunk, last token

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


def same(a, b):
    return a.shape == b.shape and bool(mx.all(a == b))


def chunk(n, seed):
    k = mx.random.normal((1, H, n, D), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, H, n, D), key=mx.random.key(seed + 1000)).astype(mx.bfloat16)
    return k, v


pool = MlxAttentionKVPool(pool_size=64, num_layers=1, n_kv_heads=H, head_dim=D, dtype=mx.bfloat16)
slots = mx.arange(1, P + 1, dtype=mx.int32)
pk, pv = chunk(P, 1)
pool.set_kv(0, slots, pk[0].transpose(1, 0, 2), pv[0].transpose(1, 0, 2))
k1, v1 = chunk(N1, 2)
k2, v2 = chunk(N2, 3)
mx.eval(pk, pv, k1, v1, k2, v2, pool.k_buffer[0], pool.v_buffer[0])

# Two consecutive calls on one pool-backed cache.
split = PoolBackedAttentionKVCache(pool, 0, slots, P)
ka, va = split.update_and_fetch(k1, v1)
kb, vb = split.update_and_fetch(k2, v2)
mx.eval(ka, va, kb, vb)
check(ka.shape[2] == P + N1 and split.offset == P + N1 + N2,
      f"first call returns prefix + chunk ({ka.shape[2]} keys), offset {split.offset}")
check(kb.shape[2] == P + N1 + N2,
      f"second call returns prefix + both chunks ({kb.shape[2]} keys), not prefix + last token")

# One call with the concatenated chunk.
single = PoolBackedAttentionKVCache(pool, 0, slots, P)
ks, vs = single.update_and_fetch(mx.concatenate([k1, k2], axis=2), mx.concatenate([v1, v2], axis=2))
mx.eval(ks, vs)
check(same(kb, ks) and same(vb, vs), "split calls equal the single call")

# A contiguous cache replaying prefix, chunk, token.
ref = ContiguousAttentionKVCache(max_seq_len=128)
ref.update_and_fetch(pk, pv)
ref.update_and_fetch(k1, v1)
kr, vr = ref.update_and_fetch(k2, v2)
mx.eval(kr, vr)
check(same(kb, kr) and same(vb, vr), "split calls equal a contiguous cache replay")

# Materialisation carries both chunks.
cont = split.to_contiguous(max_seq_len=128)
kc, vc = cont.get_kv()
mx.eval(kc, vc)
check(cont.offset == P + N1 + N2 and same(kc, kr) and same(vc, vr),
      f"to_contiguous holds all {cont.offset} tokens")

# Mask bookkeeping follows the appended length.
m0 = PoolBackedAttentionKVCache(pool, 0, slots, P)
check(m0.make_mask(N1) == "causal" and m0.make_mask(1) is None,
      "causal mask before the first chunk, none for a single query")
m0.update_and_fetch(k1, v1)
check(m0.make_mask(1) is None and m0.offset == P + N1,
      "single-query mask after the trunk chunk is None at offset prefix + chunk")

# No prefix: appends still accumulate.
empty = PoolBackedAttentionKVCache(pool, 0, mx.array([], dtype=mx.int32), 0)
empty.update_and_fetch(k1, v1)
ke, ve = empty.update_and_fetch(k2, v2)
mx.eval(ke, ve)
check(ke.shape[2] == N1 + N2 and same(ke, mx.concatenate([k1, k2], axis=2)),
      "no-prefix cache accumulates chunks")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
