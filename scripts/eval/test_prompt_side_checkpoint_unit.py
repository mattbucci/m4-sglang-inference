#!/usr/bin/env python3
"""CPU unit test for patch 026: before a finished request's final insert, the
scheduler publishes the MLX runner's prompt-side tracked Mamba checkpoint as a
running-style insert, so a follow-up turn that diverges inside the chat
template's generation tail still finds a state within one tracking grid of
the prompt end.

    python scripts/eval/test_prompt_side_checkpoint_unit.py
"""
import os
import sys
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SGLANG_USE_MLX", "1")

import sglang.srt.managers.scheduler_components.batch_result_processor as brp

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


class Tree:
    def __init__(self, mamba=True):
        self.mamba = mamba
        self.calls = []

    def supports_mamba(self):
        return self.mamba

    def checkpoint_into_tree(self, req, *, up_to, as_running=False):
        self.calls.append((req.rid, up_to, as_running))


def req(rid, *, n_out=3, track=8704, slot=object(), protected=8192, owned=8781, skip=False):
    r = SimpleNamespace(
        rid=rid,
        output_ids=list(range(n_out)),
        skip_radix_cache_insert=skip,
        kv=SimpleNamespace(
            mamba_last_track_seqlen=track,
            mamba_ping_pong_track_buffer=slot,
            cache_protected_len=protected,
        ),
        owned_kv_len=lambda: owned,
        refreshed=0,
    )
    r.refresh_fill_ids = lambda: setattr(r, "refreshed", r.refreshed + 1)
    return r


publish = brp.SchedulerBatchResultProcessor._publish_prompt_side_mamba_checkpoint


def run(q, mlx=True, mamba=True):
    t = Tree(mamba)
    with mock.patch.object(brp, "use_mlx", lambda: mlx):
        publish(SimpleNamespace(tree_cache=t), q)
    return t.calls


# The agentic case: decoded, tracked checkpoint inside the last prefill chunk.
q = req("a")
check(run(q) == [("a", 8704, True)], "decoded request publishes the tracked checkpoint as a running insert")
check(q.refreshed == 1, "fill ids refreshed before the insert")

# Nothing to publish: no decode, no tracked slot, no marker, barred from the tree.
check(run(req("b", n_out=1)) == [], "prefill-only request: final insert already carries the tracked slot")
check(run(req("c", slot=None)) == [], "no tracked slot: nothing to publish")
check(run(req("d", track=None)) == [], "no tracked marker: nothing to publish")
check(run(req("e", skip=True)) == [], "request barred from the radix cache: nothing to publish")

# The checkpoint must lie strictly inside (protected, owned): already-inserted
# chunk boundaries and the end position are covered by the existing inserts.
check(run(req("f", track=8192)) == [], "checkpoint at the protected prefix: already in the tree")
check(run(req("g", track=8781)) == [], "checkpoint at the end: the final insert covers it")
check(run(req("h", track=8192 + 64)) == [("h", 8256, True)], "first grid point past the protected prefix publishes")

# Other backends and non-Mamba trees are untouched.
check(run(req("i"), mlx=False) == [], "CUDA/ROCm: no MLX tracked slot semantics")
check(run(req("j"), mamba=False) == [], "tree without Mamba component: nothing to publish")

print(f"\n{'OK' if not fails else 'FAILED'}: {fails} failure(s)")
sys.exit(1 if fails else 0)
