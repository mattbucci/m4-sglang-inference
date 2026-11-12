#!/usr/bin/env python3
"""Unit test for prompt-lookup drafts on a lone request's decode (patch 040).

Two small random models on the Metal device, no server and no download: an
mlx-vlm Qwen3.5 language model (recurrent layers plus attention, the hybrid
rollback) and an mlx-lm Qwen3 model (attention only, the trim rollback).

With ``SGLANG_MLX_LOOKUP_DECODE`` a lone request guesses its next tokens from
an earlier occurrence of its context's last n-gram and checks the guess in
one forward. A random model never repeats its context on cue, so the
runner mechanics are driven by an oracle that drafts the reference decode's
tokens with a chosen error; the index and the length policy are tested apart.

  1. Index and policy: the n-gram index against the offline replay's
     reference, the row ladder, the stop-token cut and the length cap.
  2. Tokens: every draft pattern returns the ordinary loop's tokens, on both
     models, one token per scheduler step.
  3. Caches: leaving lookup decode ahead of, behind and level with the handed
     out tokens puts the caches where the ordinary loop keeps them.
  4. Release: the auxiliary-state store and the row release settle first; a
     hybrid that cannot be settled reports it.
  5. Routing: image, multi-request, logit-edit and logprob steps keep the
     ordinary loop; lookup steps are never chained.
  6. Sampling: a seeded request draws the ordinary loop's tokens under every
     draft pattern; an unseeded one draws each token from the distribution
     of its own position (with a control that reads the row before).

  python scripts/eval/test_lookup_decode_unit.py
"""
import inspect
import sys
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

import tiny_mrope_model as tiny
from tiny_mrope_model import check, mr, mx

from mlx.utils import tree_map_with_path
from mlx_lm.models import qwen3

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx import lookup_decode as ld
from sglang.srt.hardware_backend.mlx import scheduler_mixin
from sglang.srt.hardware_backend.mlx.aot import MlxAOTKernelSet
from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.sampling import MlxSamplingParams
from sglang.srt.hardware_backend.mlx.sdpa_blocks import SdpaBlockLadder

STEPS = 48


def arm(r, hybrid):
    """The per-request records __init__ would have made."""
    r._req_caches, r._req_token_ids, r._req_sampling = {}, {}, {}
    r._req_synced_offset, r._lookahead_aux_base = {}, {}
    r._lookup_limits, r._lookup_state, r._lookup_off = {}, {}, set()
    r._lookup_rows = 0
    r._decode_step_ct, r._clear_steps = 0, 0
    r.disable_radix_cache = False
    r._cache_pool = []
    r._snapshot_layer_indices = (
        tuple(r._cache_layout.auxiliary_layer_indices) if hybrid else ()
    )
    r._has_snapshot_state = bool(r._snapshot_layer_indices)
    r.hybrid = hybrid
    return r


def make_text_runner(seed=5):
    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=96,
        num_hidden_layers=4,
        intermediate_size=192,
        num_attention_heads=3,
        rms_norm_eps=1e-6,
        vocab_size=211,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=32,
        tie_word_embeddings=False,
        rope_theta=100.0,
    )
    mx.random.seed(seed)
    model = qwen3.Model(args)
    model.update(
        tree_map_with_path(
            lambda _, v: v * 4.0 if v.ndim >= 2 else v, model.parameters()
        )
    )
    model.set_dtype(mx.float32)
    model.eval()
    mx.eval(model.parameters())
    patch_model_attention(model)
    layers, attrs = find_attention_layers(model)
    r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    r.model = model
    r.is_vlm = False
    r._cache_layout = MlxModelCacheLayout.from_attention_discovery(layers, attrs)
    r._max_seq_len = 256
    r._sdpa_block_ladder = SdpaBlockLadder(False)
    r._aot_kernels = MlxAOTKernelSet()
    r._attention_kv_pool = None
    r._req_pool_idx = {}
    r._req_to_token_pool = None
    r._req_rope_delta = {}
    r._req_position_delta = {}
    r._attention_wrapper_batchable = True
    return arm(r, hybrid=False)


def prefill(r, rid, ids, cache, grid=None):
    if r.is_vlm:
        return tiny.forward_prompt(r, tiny.Prompt(rid, ids, grid), cache)
    logits = r.model(mx.array([ids], dtype=mx.int32), cache=cache)
    mx.eval(logits, [c.state for c in cache])
    return int(np.array(logits[0, -1]).argmax())


def one_row(r, rid, cache, tok):
    """The ordinary loop's forward for one request: logits after ``tok``."""
    ids = mx.array([[tok]], dtype=mx.int32)
    if r._batches_decode(1):
        logits = r._decode_with_batched_attention([cache], ids, [rid])
    else:
        logits = r._decode_with_native_cache([cache], [ids], [rid])
    mx.eval(logits, [c.state for c in cache])
    return np.array(logits[0].astype(mx.float32))


def reference(r, rid, ids, steps=STEPS, grid=None):
    """Ordinary decode: generated tokens, and the logits behind each step."""
    cache = r._new_native_cache()
    toks, rows = [prefill(r, rid, ids, cache, grid)], []
    for _ in range(steps):
        rows.append(one_row(r, rid, cache, toks[-1]))
        toks.append(int(rows[-1].argmax()))
    return toks, rows


def admit(r, rid, ids, stop=(), limit=1 << 30, grid=None, sampling=None):
    """What prefill_start and prefill_finalize leave behind for a request."""
    cache = r._new_native_cache()
    tok = prefill(r, rid, ids, cache, grid)
    r._req_caches[rid] = cache
    r._req_token_ids[rid] = list(ids) + [tok]
    if sampling is not None:
        r._req_sampling[rid] = sampling
    if r._lookup_rows:
        r._lookup_forget(rid)
        r._lookup_limits[rid] = (frozenset(stop), limit)
    return cache


def step(r, rids, **kw):
    """One scheduler step: launch, evaluate, finalize."""
    pending = r.decode_batch_start(list(rids), **kw)
    mx.async_eval(pending.lazy_tokens)
    return pending, r.decode_batch_finalize(pending)


def oracle(ref, n_prompt, pattern):
    """A ``propose`` that drafts the reference tokens with a planned error.

    ``pattern`` is cycled once per call: None drafts the right tokens, an int
    makes the draft wrong from that index on (right if the draft is shorter),
    "skip" drafts nothing.
    """
    calls = [0]

    def propose(self, max_rows):
        plan = pattern[calls[0] % len(pattern)]
        calls[0] += 1
        if plan == "skip":
            return []
        k = len(self.stream) - n_prompt
        n = ld.draft_rows(back=8, level=self.level, avail=99, max_rows=max_rows) - 1
        draft = list(ref[k : k + n])
        if isinstance(plan, int) and plan < len(draft):
            draft[plan:] = [(t + 1) % 200 for t in draft[plan:]]
        return self.clip(draft)

    return propose


def run_lookup(r, rid, ids, ref, pattern, steps=STEPS, rows=4, **admit_kw):
    """Decode with lookup drafts; returns tokens, pendings and the stream."""
    r._lookup_rows = rows
    admit(r, rid, ids, **admit_kw)
    out, pendings = [r._req_token_ids[rid][-1]], []
    with mock.patch.object(ld.LookupStream, "propose", oracle(ref, len(ids), pattern)):
        for _ in range(steps):
            pending, toks = step(r, [rid])
            pendings.append(pending)
            out.append(toks[0])
    return out, pendings, r._lookup_state.get(rid)


def close(a, b, scale):
    return float(np.abs(a - b).max()) <= 1e-4 * scale


def drop(r, rid):
    r.remove_request(rid)
    r._req_pool_idx.pop(rid, None)


def test_index_and_policy():
    class Lookup:  # scripts/bench/prompt_lookup_acceptance.py
        def __init__(self, seq, ns):
            self.seq, self.ns = seq, sorted(ns, reverse=True)
            self.idx = {n: {} for n in ns}
            self.reg = 0

        def find(self, e):
            seq = self.seq
            while self.reg < e - 1:
                p = self.reg
                for n in self.ns:
                    if p + 1 >= n:
                        self.idx[n][tuple(seq[p + 1 - n : p + 1])] = p + 1
                self.reg += 1
            for n in self.ns:
                if e >= n:
                    p = self.idx[n].get(tuple(seq[e - n : e]))
                    if p is not None:
                        back = n
                        while back < 64 and p - back - 1 >= 0 and seq[p - back - 1] == seq[e - back - 1]:
                            back += 1
                        return p, back
            return None, 0

    rng = np.random.default_rng(3)
    agree = total = 0
    for _ in range(30):
        vocab = int(rng.choice([3, 5, 12, 50]))
        seq = [int(t) for t in rng.integers(0, vocab, int(rng.integers(1, 40)))]
        twin = list(seq)
        a, b = ld.NgramIndex(seq), Lookup(twin, ld.NGRAMS)
        for _ in range(200):
            agree += a.find() == b.find(len(twin))
            total += 1
            for t in rng.integers(0, vocab, int(rng.integers(1, 4))):
                seq.append(int(t))
                twin.append(int(t))
    check("index: agrees with the offline replay's lookup", agree == total, f"{agree}/{total}")

    seq = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
    check("index: the last n-gram finds its earlier occurrence", ld.NgramIndex(seq).find() == (3, 3))
    check("index: no earlier occurrence, no match", ld.NgramIndex([1, 2, 3, 4]).find() == (None, 0))
    check("index: the last n-gram does not match itself", ld.NgramIndex([7, 7]).find() == (None, 0))

    rows = [ld.draft_rows(back=8, level=level, avail=99, max_rows=4) for level in range(4)]
    check("policy: two rows after a miss, doubling to the cap", rows == [2, 4, 4, 4], str(rows))
    check("policy: a cap of two stays at two", [ld.draft_rows(back=8, level=level, avail=99, max_rows=2) for level in range(3)] == [2, 2, 2])
    check("policy: rows never pass what follows the match", ld.draft_rows(back=8, level=3, avail=2, max_rows=8) == 3)
    check("policy: a backward match under two tokens drafts nothing", ld.draft_rows(back=1, level=3, avail=99, max_rows=4) == 1)

    s = ld.LookupStream(tokens=[1, 2, 3, 4, 5, 6, 7, 1, 2, 3], stop_ids=frozenset(), limit=1 << 30)
    s.level = 3
    check("propose: copies what followed the match", s.propose(4) == [4, 5, 6], str(s.propose(4)))
    s = ld.LookupStream(tokens=[1, 2, 3, 4, 5, 6, 7, 1, 2, 3], stop_ids=frozenset({5}), limit=1 << 30)
    s.level = 3
    check("propose: the draft stops before a stop token", s.propose(4) == [4])
    s = ld.LookupStream(tokens=[1, 2, 3, 4, 5, 6, 7, 1, 2, 3], stop_ids=frozenset(), limit=13)
    s.level = 3
    check("propose: the draft leaves room for the forward's own token", s.propose(4) == [4, 5])
    s = ld.LookupStream(tokens=[1, 2, 3, 4, 5, 6, 7, 1, 2, 3], stop_ids=frozenset(), limit=11)
    check("propose: one token left, no draft", s.propose(4) == [])
    s = ld.LookupStream(tokens=[5, 6], stop_ids=frozenset(), limit=99)
    kept = s.accept(drafts=[7, 8, 9], picks=[7, 8, 1, 2])
    check("accept: keeps the matching drafts and the model's next token", kept == 2 and s.stream == [5, 6, 7, 8, 1] and s.level == 0)
    kept = s.accept(drafts=[4], picks=[4, 3])
    check("accept: a fully kept draft raises the ladder", kept == 1 and s.stream[-2:] == [4, 3] and s.level == 1)

    req = SimpleNamespace(
        sampling_params=SimpleNamespace(ignore_eos=False, stop_token_ids={9}, max_new_tokens=40),
        eos_token_ids={7},
        tokenizer=SimpleNamespace(eos_token_id=5, additional_stop_token_ids=[11]),
        origin_input_ids=list(range(100)),
    )
    check("limits: every stop id the scheduler checks, and prompt plus budget", ld.request_limits(req) == (frozenset({5, 7, 9, 11}), 140))
    req.sampling_params.ignore_eos = True
    req.sampling_params.max_new_tokens = None
    stop, limit = ld.request_limits(req)
    check("limits: ignore_eos has no stop ids, no budget is unbounded", stop == frozenset() and limit > 1 << 29)


PATTERNS = {
    "every draft right": [None],
    "every draft wrong at once": [0],
    "right, then wrong at the second token": [None, 1],
    "right twice, then wrong at the third token": [None, None, 2],
    "mixed, with undrafted steps": [None, 1, "skip", None, None, 0, "skip", 2, None],
}


def test_tokens(r, name, ids):
    ref, rows = reference(r, "ref", ids)
    scale = float(max(np.abs(row).max() for row in rows))
    for label, pattern in PATTERNS.items():
        out, pendings, state = run_lookup(r, "a", ids, ref, pattern)
        forwards = sum(p.lookup.lazy_rows is not None for p in pendings)
        queued = sum(p.lookup.queued for p in pendings)
        check(
            f"{name}, {label}: {STEPS} tokens equal the ordinary loop's",
            out == ref,
            f"{state.draft_forward_ct} drafting forwards, {state.replay_forward_ct} replay-only, {state.plain_forward_ct} ordinary, "
            f"{queued} queued, {state.correct_draft_ct}/{state.proposed_draft_ct} kept, {state.replay_row_ct} replayed",
        )
        check(
            f"{name}, {label}: every step is accounted for",
            forwards == state.draft_forward_ct + state.replay_forward_ct and forwards + state.plain_forward_ct + queued == STEPS,
        )
        check(
            f"{name}, {label}: every step is a lookup step and none is chainable",
            all(p.no_chain and p.lookup is not None and p.lookup.state is state for p in pendings),
        )
        if label == "every draft right":
            check(f"{name}: right drafts take fewer forwards than tokens", forwards + state.plain_forward_ct <= STEPS // 3, f"{forwards + state.plain_forward_ct} forwards")
            check(f"{name}: the request summary reads back", state.summary().startswith(f"{len(state.stream) - len(ids) - 1} tokens from {state.draft_forward_ct} drafting, 0 replay-only"), state.summary())
            check(f"{name}: right drafts replay nothing", state.replay_row_ct == 0 and state.correct_draft_ct == state.proposed_draft_ct)
        if label == "right, then wrong at the second token":
            if r.hybrid:
                check(f"{name}: a hybrid replays the tokens a missed forward kept", state.replay_row_ct > 0, f"{state.replay_row_ct} rows")
            else:
                check(f"{name}: attention-only caches are trimmed, nothing replays", state.replay_row_ct == 0)
        # The caches after the run, through one ordinary forward.
        cache, toks = r._req_caches["a"], r._req_token_ids["a"]
        settled = r._lookup_leave("a")
        got = one_row(r, "a", cache, toks[-1])
        want = one_row(r, "ref2", _advance(r, ids, ref, len(toks) - len(ids) - 1), toks[-1])
        check(
            f"{name}, {label}: caches after leaving match the ordinary loop's",
            settled and close(got, want, scale),
            f"max abs diff {float(np.abs(got - want).max()):.2e} of {scale:.2f}",
        )
        drop(r, "a")
    return ref, rows, scale


def _advance(r, ids, ref, n):
    """An ordinary-loop cache that has consumed the prompt and ref[:n]."""
    cache = r._new_native_cache()
    prefill(r, "ref2", ids, cache)
    for tok in ref[:n]:
        one_row(r, "ref2", cache, tok)
    return cache


def decode_until(r, rid, ids, ref, pattern, want, rows=4):
    """Lookup decode until ``want(state, tokens, cache offset)`` holds."""
    r._lookup_rows = rows
    cache = admit(r, rid, ids)
    with mock.patch.object(ld.LookupStream, "propose", oracle(ref, len(ids), pattern)):
        for _ in range(STEPS):
            step(r, [rid])
            state, toks = r._lookup_state[rid], r._req_token_ids[rid]
            if want(state, toks, r._first_attention_cache(cache).offset):
                return cache, state, toks
    raise AssertionError("the scenario never reached its stopping point")


def test_leave(r, name, ids, ref, scale):
    ahead = lambda state, toks, off: off > len(toks) - 1 and len(state.stream) - len(toks) >= 2
    behind = lambda state, toks, off: off < len(toks) - 1
    level = lambda state, toks, off: off == len(toks) - 1 and len(toks) - len(ids) > 6

    def settle_and_compare(label, cache, toks, how):
        ok = how()
        n = len(toks) - len(ids) - 1
        offset = r._first_attention_cache(cache).offset
        got = one_row(r, "a", cache, toks[-1])
        want = one_row(r, "ref2", _advance(r, ids, ref, n), toks[-1])
        check(
            f"{name}, {label}: caches hold all but the newest token and match",
            ok is not False and offset == len(toks) - 1 and close(got, want, scale),
            f"offset {offset} for {len(toks)} tokens, max abs diff {float(np.abs(got - want).max()):.2e}",
        )
        check(f"{name}, {label}: the request stays on the ordinary loop", "a" in r._lookup_off and "a" not in r._lookup_state)

    cache, state, toks = decode_until(r, "a", ids, ref, [None], ahead)
    check(f"{name}: a kept draft leaves the caches ahead of the handed out tokens", r._first_attention_cache(cache).offset - (len(toks) - 1) >= 2)
    if r.hybrid:
        check(f"{name}: the rollback point is held while tokens are queued", state.mark is not None and state.mark[0] <= len(toks) - 1)
    settle_and_compare("left ahead", cache, toks, lambda: r._lookup_leave("a"))
    # ...and the ordinary loop continues the same text from there.
    r._req_token_ids["a"].append(int(one_row(r, "x", _advance(r, ids, ref, len(toks) - len(ids) - 1), toks[-1]).argmax()))
    n = len(toks) - len(ids)
    more = [step(r, ["a"]) for _ in range(4)]
    check(
        f"{name}, left ahead: the ordinary loop carries on with the same tokens",
        r._req_token_ids["a"][len(ids):] == ref[: n + 4] and not any(p.no_chain or p.lookup for p, _ in more),
        f"{n} tokens at the switch",
    )
    drop(r, "a")

    if r.hybrid:
        cache, state, toks = decode_until(r, "a", ids, ref, [None, 1], behind)
        settle_and_compare("left behind", cache, toks, lambda: r._lookup_leave("a"))
        drop(r, "a")
    else:
        try:
            decode_until(r, "a", ids, ref, [None, 1, 0, 2], behind)
            check(f"{name}: attention-only caches never fall behind", False)
        except AssertionError:
            check(f"{name}: attention-only caches never fall behind", True)
        drop(r, "a")

    cache, state, toks = decode_until(r, "a", ids, ref, ["skip", 0], level)
    settle_and_compare("left level", cache, toks, lambda: r._lookup_leave("a"))
    drop(r, "a")

    # Release-time hooks.
    stored = []
    r._store_auxiliary_state = lambda idx, c: stored.append((idx, r._first_attention_cache(r._req_caches["a"]).offset))
    cache, state, toks = decode_until(r, "a", ids, ref, [None], ahead)
    r._req_pool_idx["a"] = 3
    key_len = len(toks) - 1
    settle_and_compare("store for the radix insert", cache, list(toks), lambda: r.store_auxiliary_state_for_request("a", key_len))
    if r.hybrid:
        check(f"{name}: the store sees the settled caches", stored == [(3, key_len)], str(stored))
    else:
        check(f"{name}: nothing to store without recurrent state", stored == [])
    drop(r, "a")

    cache, state, toks = decode_until(r, "a", ids, ref, [None], ahead)
    r._req_pool_idx["a"] = 3
    r.release_request_row("a", len(toks) - 1)
    check(
        f"{name}: the row release settles before it syncs",
        r._first_attention_cache(cache).offset == len(toks) - 1 and "a" not in r._lookup_state and "a" not in r._req_pool_idx,
    )
    drop(r, "a")

    cache, state, toks = decode_until(r, "a", ids, ref, [None], ahead)
    before = r._first_attention_cache(cache).offset
    r.disable_radix_cache = True
    ok = r._lookup_leave("a", finishing=True)
    r.disable_radix_cache = False
    check(f"{name}: with the radix cache off a finished request is left as is", ok and r._first_attention_cache(cache).offset == before)
    drop(r, "a")

    if r.hybrid:
        cache, state, toks = decode_until(r, "a", ids, ref, [None], ahead)
        state.mark = None
        r._req_pool_idx["a"] = 3
        stored.clear()
        check(
            f"{name}: recurrent state that cannot be set back skips the radix insert",
            r.store_auxiliary_state_for_request("a", len(toks) - 1) is False and stored == [],
        )
        drop(r, "a")
    del r._store_auxiliary_state

    # A budget of 12 tokens: the scheduler stops the request after 11 steps.
    out, pendings, state = run_lookup(r, "a", ids, ref, [None], steps=11, limit=len(ids) + 12)
    drafts = [len(p.lookup.drafts) for p in pendings if p.lookup.lazy_rows is not None]
    check(
        f"{name}: no forward verifies a token past prompt plus max_new_tokens",
        len(state.stream) == len(ids) + 12 and out == ref[:12] and state.correct_draft_ct == state.proposed_draft_ct,
        f"drafts of {drafts}, stream {len(state.stream) - len(ids)} outputs for a budget of 12",
    )
    drop(r, "a")
    stop_at = ref[9]
    first = ref.index(stop_at)
    out, pendings, state = run_lookup(r, "a", ids, ref, [None], steps=first, stop=[stop_at])
    check(
        f"{name}: no forward runs past a drafted stop token",
        out == ref[: first + 1] and len(state.stream) <= len(ids) + first + 1,
        f"stop token at output {first}, stream holds {len(state.stream) - len(ids)} outputs",
    )
    drop(r, "a")


def test_routing(r, name, ids, ref, other_ids):
    r._lookup_rows = 4
    other_ref, _ = reference(r, "ref", other_ids, steps=8)
    cache_a = admit(r, "a", ids)
    with mock.patch.object(ld.LookupStream, "propose", oracle(ref, len(ids), [None])):
        for _ in range(3):
            step(r, ["a"])
        n = len(r._req_token_ids["a"]) - len(ids)
        check(f"{name}: verified tokens are queued when a second request joins", len(r._lookup_state["a"].stream) > len(r._req_token_ids["a"]))
        admit(r, "b", other_ids)
        joined = [step(r, ["a", "b"]) for _ in range(4)]
        alone = [step(r, ["a"]) for _ in range(3)]
    check(
        f"{name}: a step of two requests ends lookup decode for both",
        r._lookup_off >= {"a", "b"} and not r._lookup_state and not any(p.no_chain or p.lookup for p, _ in joined + alone),
    )
    check(f"{name}: the first request keeps its tokens through the switch", r._req_token_ids["a"][len(ids):] == ref[: n + 7], f"{n} tokens before the switch")
    check(f"{name}: the joining request decodes its own tokens", r._req_token_ids["b"][len(other_ids):] == other_ref[:5])
    check(f"{name}: caches are level with the tokens after the switch", cache_a[r._cache_layout.first_attention_layer_index].offset == len(r._req_token_ids["a"]) - 1)
    drop(r, "a")
    drop(r, "b")

    for label, kw in (
        ("logit edits", {"edit_rows": mx.zeros((1, 211))}),
        ("a logprob request", {"logprob_spec": object()}),
    ):
        admit(r, "a", ids)
        with mock.patch.object(ld.LookupStream, "propose", oracle(ref, len(ids), [None])):
            step(r, ["a"])
            with mock.patch.object(r, "_select_tokens_with_logprobs", lambda logits, *a, **k: (mx.argmax(logits, axis=-1), None)):
                pending, _ = step(r, ["a"], **kw)
            after, _ = step(r, ["a"])
        check(
            f"{name}: a step with {label} leaves lookup decode for good",
            pending.lookup is None and not pending.no_chain and after.lookup is None and "a" in r._lookup_off
            and r._req_token_ids["a"][len(ids):] == ref[:4],
        )
        drop(r, "a")

    admit(r, "a", ids)
    r._enable_sampling = True
    r._req_sampling["a"] = MlxSamplingParams(temperature=0.7, top_k=40, top_p=0.9, min_p=0.0, seed=None)
    check(f"{name}: a sampled request opens a stream", r._lookup_open("a") is not None)
    r._req_sampling["a"] = MlxSamplingParams(temperature=0.0, top_k=1, top_p=1.0, min_p=0.0, seed=None)
    check(f"{name}: a greedy request does", r._lookup_open("a") is not None)
    r._enable_sampling = False
    del r._lookup_limits["a"]
    check(f"{name}: a request whose limits are unknown does not", r._lookup_open("a") is None)
    drop(r, "a")

    admit(r, "a", ids)
    r._req_caches["a"][r._cache_layout.first_attention_layer_index].offset -= 1
    check(f"{name}: caches out of step with the tokens do not", r._lookup_open("a") is None)
    drop(r, "a")

    r._lookup_rows = 0
    admit(r, "a", ids)
    pending, _ = step(r, ["a"])
    check(f"{name}: flag off, the ordinary step is untouched", pending.lookup is None and not pending.no_chain and not r._lookup_state)
    drop(r, "a")
    r._lookup_rows = 4
    admit(r, "a", ids)
    with mock.patch.object(ld.LookupStream, "propose", oracle(ref, len(ids), [None])):
        step(r, ["a"])
    r.remove_request("a")
    check(f"{name}: removing a request drops every lookup record", not r._lookup_state and not r._lookup_limits and not r._lookup_off)
    admit(r, "a", ids)
    r.clear()
    check(f"{name}: clear drops every lookup record", not r._lookup_state and not r._lookup_limits and not r._req_caches)


def would_draft(ids, ref):
    """Steps of the reference decode at which the index has a match."""
    seq = list(ids) + [ref[0]]
    index = ld.NgramIndex(seq)
    hits = 0
    for tok in ref[1:]:
        p, back = index.find()
        hits += p is not None and back >= ld.MIN_BACK
        seq.append(tok)
    return hits


SAMPLED = dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)


def sampled_reference(r, rid, ids, params, steps=STEPS):
    """The ordinary loop under the request's sampler."""
    r._lookup_rows = 0
    admit(r, rid, ids, sampling=params)
    out = [r._req_token_ids[rid][-1]]
    for _ in range(steps):
        out.append(step(r, [rid])[1][0])
    drop(r, rid)
    return out


def test_sampled(r, name, ids, greedy):
    r._enable_sampling = True
    r._rng_key = mx.random.key(11)
    seeded = MlxSamplingParams(seed=7, **SAMPLED)
    ref = sampled_reference(r, "ref", ids, seeded)
    check(f"{name}, seeded: the ordinary loop repeats its own draw", sampled_reference(r, "ref", ids, seeded) == ref)
    parts = next((i for i, (a, b) in enumerate(zip(ref, greedy)) if a != b), None)
    check(f"{name}, seeded: the draw is not the greedy text", parts is not None, f"parts at token {parts}")
    for label, pattern in PATTERNS.items():
        out, pendings, state = run_lookup(r, "a", ids, ref, pattern, sampling=seeded)
        check(
            f"{name}, seeded, {label}: {STEPS} sampled tokens equal the ordinary loop's",
            out == ref and all(p.no_chain and p.lookup is not None for p in pendings),
            f"{state.draft_forward_ct} drafting forwards, {state.correct_draft_ct}/{state.proposed_draft_ct} kept, {state.replay_row_ct} replayed",
        )
        drop(r, "a")

    # No seed: tokens are free to differ, but each one must be a draw from
    # the distribution the ordinary loop has at that token's own position.
    # At a low temperature the draw stays near the greedy text, so drafts of
    # the greedy reference are kept; at temperature 1 most are rejected.
    free = MlxSamplingParams(seed=None, **SAMPLED)
    cold = MlxSamplingParams(seed=None, **{**SAMPLED, "temperature": 0.05})
    mixed = PATTERNS["mixed, with undrafted steps"]
    for label, pattern, params, drafts in (
        ("every draft right", [None], free, ref),
        ("mixed", mixed, free, ref),
        ("low temperature, mixed", mixed, cold, greedy),
    ):
        out, pendings, state = run_lookup(r, "a", ids, drafts, pattern, sampling=params)
        cache = r._new_native_cache()
        prefill(r, "tf", ids, cache)
        rows = [one_row(r, "tf", cache, prev) for prev in out[:-1]]
        own = [int((row > row[tok]).sum()) for row, tok in zip(rows, out[1:])]
        before = [int((row > row[tok]).sum()) for row, tok in zip(rows, out[2:])]
        check(
            f"{name}, unseeded, {label}: every token is inside the top-k of its own position",
            max(own) < SAMPLED["top_k"] and state.proposed_draft_ct > 0 and (params is free or state.correct_draft_ct > 0),
            f"worst rank {max(own)}, {state.correct_draft_ct}/{state.proposed_draft_ct} drafted tokens kept",
        )
        check(
            f"{name}, unseeded, {label}: control, read against the row before, tokens fall outside it",
            max(before) >= SAMPLED["top_k"],
            f"worst rank {max(before)}",
        )
        drop(r, "a")
        r._req_pool_idx.pop("tf", None)
    r._enable_sampling = False


def test_natural(r, name, rng):
    """No oracle: the real index drafts on a prompt it can match."""
    # A random model repeats nothing on cue. Pair the prompt's last token
    # with many others, so that whatever comes first has been seen after it.
    for attempt in range(1, 31):
        last = int(rng.integers(0, 200))
        others = [int(t) for t in rng.permutation(200) if t != last][:70]
        ids = [tok for t in others for tok in (last, t)] + [last]
        ref, _ = reference(r, "ref", ids, steps=64)
        hits = would_draft(ids, ref)
        if hits:
            break
    check(f"{name}, real index: a prompt whose decode the index can match", hits > 0, f"attempt {attempt}, {hits} steps with a match")
    r._lookup_rows = 4
    admit(r, "a", ids)
    out = [r._req_token_ids["a"][-1]]
    for _ in range(64):
        out.append(step(r, ["a"])[1][0])
    state = r._lookup_state["a"]
    check(f"{name}, real index: tokens equal the ordinary loop's", out == ref, state.summary())
    check(f"{name}, real index: drafts were made and checked", state.proposed_draft_ct > 0 and state.draft_forward_ct > 0)
    drop(r, "a")


def main():
    rng = np.random.default_rng(23)
    print("module:", mr.__file__)
    test_index_and_policy()

    hybrid = arm(tiny.make_runner(tiny.make_lm()), hybrid=True)
    text = make_text_runner()
    check("models: one with recurrent layers, one attention only", hybrid._has_snapshot_state and not text._has_snapshot_state and text._batches_decode(1) and not hybrid._batches_decode(1))
    for r, name in ((hybrid, "hybrid"), (text, "dense")):
        ids = [int(t) for t in rng.integers(0, 200, 37)]
        other = [int(t) for t in rng.integers(0, 200, 21)]
        ref, rows, scale = test_tokens(r, name, ids)
        test_leave(r, name, ids, ref, scale)
        test_routing(r, name, ids, ref, other)
        test_sampled(r, name, ids, ref)
        test_natural(r, name, rng)

    image = tiny.image_prompt("img", rng, head=5, grid=(1, 8, 12), tail=9)
    hybrid._lookup_rows = 4
    admit(hybrid, "img", image.ids, grid=image.grid)
    check("hybrid: an image request does not open a stream", hybrid._position_delta("img") != 0 and hybrid._lookup_open("img") is None)
    pending, _ = step(hybrid, ["img"])
    check("hybrid: an image request decodes on the ordinary loop", pending.lookup is None and "img" in hybrid._lookup_off)
    drop(hybrid, "img")

    # The worker evaluates a step's first token; finalize reads every row.
    x, y = mx.random.normal((1500, 1500)), mx.random.normal((1500, 1500))
    mx.eval(x, y)
    lazy_rows = mx.argmax((x @ y) @ (y @ x) @ (x + y), axis=-1)
    head = lazy_rows[:1]
    t0 = time.perf_counter()
    mx.async_eval(head)
    head.tolist()
    t1 = time.perf_counter()
    lazy_rows.tolist()
    t2 = time.perf_counter()
    check(
        "mlx: evaluating the first row's token evaluates the forward's other rows",
        (t2 - t1) < (t1 - t0) / 5,
        f"{1e3 * (t1 - t0):.1f} ms for the slice, {1e3 * (t2 - t1):.3f} ms for the rest",
    )

    bare = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    check("flag: a bare runner has lookup decode off", bare._lookup_rows == 0)
    check("flag: off unless set, four rows by default", envs.SGLANG_MLX_LOOKUP_DECODE.get() is False and envs.SGLANG_MLX_LOOKUP_DECODE_ROWS.get() == 4)
    src = inspect.getsource(scheduler_mixin)
    fresh = src[src.index("def _launch_fresh") : src.index("def _launch_chained")]
    check("scheduler: a fresh launch reads no_chain into chain_safe", "launch.decode.no_chain" in fresh and "chain_safe=self._mlx_batch_chain_safe(batch)\n" in fresh)
    check("pending: no_chain and lookup default off", mr.MlxPendingDecode(lazy_tokens=None, req_ids=[], caches=[]).no_chain is False)

    print(f"\n{len(tiny.FAILS)} failed" if tiny.FAILS else "\nall passed")
    sys.exit(1 if tiny.FAILS else 0)


if __name__ == "__main__":
    main()
