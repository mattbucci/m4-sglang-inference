#!/usr/bin/env python3
"""CPU unit test for patch 026: a chunked-prefill continuation chunk on the MLX
runner tracks the auxiliary (Mamba/DeltaNet) state at the last grid boundary it
crosses, like the first chunk does, so the scheduler's end-of-prefill checkpoint
publishes a state within one grid of the prompt end instead of the chunk end.

    python scripts/eval/test_extend_chunk_track_unit.py
"""
import os
import sys
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx

import sglang.srt.hardware_backend.mlx.model_runner as mr

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


class Cache:
    def __init__(self, offset):
        self.offset = offset


def make_runner(offset, snapshot=True):
    r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    r._has_snapshot_state = snapshot
    r._cache_layout = SimpleNamespace(first_attention_layer_index=0)
    r._req_caches = {"r": [Cache(offset)]}
    r._req_synced_offset = {"r": offset}
    r.disable_radix_cache = True
    r.model = mock.Mock(name="model")
    r._trunk_forward = mock.Mock(return_value=None)
    r._forward_lazy_token = mock.Mock(return_value=("tok", None))
    r._store_tracked_auxiliary_state = mock.Mock()
    return r


def run(offset, n_new, req, snapshot=True):
    r = make_runner(offset, snapshot)
    with mock.patch.object(mr, "mamba_cache_chunk_size", return_value=64):
        pending = r.extend_start(
            req_id="r", new_token_ids=list(range(1000, 1000 + n_new)),
            new_slot_ids=[], req=req,
        )
    forwarded = [c.args[0].shape[1] for c in r._trunk_forward.call_args_list]
    run_len = r._forward_lazy_token.call_args.args[0].shape[1]
    stores = [c.args[2] for c in r._store_tracked_auxiliary_state.call_args_list]
    return pending, forwarded, run_len, stores


text_req = SimpleNamespace(mamba_branching_seqlen=None, multimodal_inputs=None)

# Last chunk of a 3-chunk prompt: 8192 computed, 465 new -> grid state at 8640.
pending, fwd, run_len, stores = run(8192, 465, text_req)
check(fwd == [448], f"head forward covers the tracked span: {fwd}")
check(run_len == 17, f"the rest of the chunk runs after the store: {run_len}")
check(stores == [8640], f"state tracked at prefix + floor(new/grid)*grid: {stores}")
check(pending.new_token_ids == list(range(1000, 1465)), "pending keeps the whole chunk")

# Grid-aligned chunk end: single forward, store after it.
_, fwd, run_len, stores = run(4096, 4096, text_req)
check(fwd == [] and run_len == 4096, f"aligned chunk is not split: {fwd} {run_len}")
check(stores == [8192], f"aligned chunk tracks at its end: {stores}")

# Fewer new tokens than a grid: nothing to track.
_, fwd, run_len, stores = run(8192, 40, text_req)
check(fwd == [] and run_len == 40 and stores == [], "sub-grid chunk tracks nothing")

# Branching point inside the chunk wins when grid-aligned.
br_req = SimpleNamespace(mamba_branching_seqlen=8320, multimodal_inputs=None)
_, fwd, run_len, stores = run(8192, 465, br_req)
check(stores == [8320] and fwd == [128], f"grid-aligned branching point tracked: {stores} {fwd}")

# Image-bearing request and absent request: no split, no store.
mm_req = SimpleNamespace(mamba_branching_seqlen=None, multimodal_inputs=object())
_, fwd, run_len, stores = run(8192, 465, mm_req)
check(fwd == [] and run_len == 465 and stores == [], "multimodal request is not split")
_, fwd, run_len, stores = run(8192, 465, None)
check(fwd == [] and run_len == 465 and stores == [], "no request: no tracking")

# No snapshot state (attention-only model): untouched.
_, fwd, run_len, stores = run(8192, 465, text_req, snapshot=False)
check(fwd == [] and stores == [], "attention-only model tracks nothing")

print(f"--- {'ALL PASS' if not fails else f'{fails} FAIL'} ---")
sys.exit(1 if fails else 0)
