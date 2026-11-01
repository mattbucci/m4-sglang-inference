#!/usr/bin/env python3
"""CPU unit test for patch 025: structured reads on the Gemma4 renoise loop.

Drives Gemma4Renoise.step with synthetic logits (no model, no server): a
seeded canvas replaces the random one, pinned positions hold their seed on
every step and in the emitted canvas, the step cap finishes the read, a
constrained request only emits tokens from the requested set, and a
read-only request reports per-position logprobs for those ids.

    python scripts/eval/test_dllm_structured_reads_unit.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_MLX", "1")
import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.gemma4_renoise import (
    Gemma4Renoise,
    _validate_diffusion_options,
)
from sglang.srt.dllm.config import DllmConfig

B, V, H = 8, 32, 4  # canvas, vocabulary, embedding width
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


def algo(max_steps=6):
    cfg = DllmConfig("Gemma4Renoise", {"max_denoising_steps": max_steps, "stopping_config": {"confidence_threshold": 0.0}},
                     block_size=B, mask_id=-1, max_running_requests=1, requires_separate_context_encoding=True)
    a = Gemma4Renoise(cfg)
    a.vocab_size = V
    a.embed_tokens = SimpleNamespace(weight=torch.randn(V, H), embed_scale=1.0)
    return a


def batch(options):
    return SimpleNamespace(batch_size=len(options), input_ids=torch.zeros(len(options) * B, dtype=torch.int64),
                           rids=[f"r{i}" for i in range(len(options))], dllm_options=options)


def run(a, fb, logits_fn, steps=10):
    states = a.init_step_state(fb)
    for _ in range(steps):
        done = a.step(fb, logits_fn(states), states)
        if all(done):
            break
    return states


torch.manual_seed(0)
seed = [(20 + i) % V for i in range(B)]  # valid vocabulary ids
peaked = torch.full((1, B, V), -5.0)
for p in range(B):
    peaked[0, p, (p * 3) % V] = 8.0  # a confident, non-seed answer at every position
peaked_logits = lambda states: peaked.clone()

# Seed canvas and pins.
a = algo()
fb = batch([{"seed_canvas": seed, "pinned": [0, 1, 2, 5]}])
states = a.init_step_state(fb)
check(states[0]["current"].tolist() == seed, "seed canvas replaces the random canvas")
states = run(a, fb, peaked_logits)
out = fb.input_ids.view(1, B)[0].tolist()
check(all(out[p] == seed[p] for p in (0, 1, 2, 5)), f"pinned positions keep the seed in the emitted canvas ({out})")
check(all(out[p] == (p * 3) % V for p in (3, 4, 6, 7)), "unpinned positions take the model's argmax")

# Step cap.
a = algo(max_steps=6)
fb = batch([{"seed_canvas": seed, "max_steps": 2}])
states = a.init_step_state(fb)
check(states[0]["step"] == 2, "max_steps caps the denoising steps")
noisy = lambda states: torch.randn(1, B, V)
calls = 0
def counting(states):
    global calls; calls += 1; return torch.randn(1, B, V)
states = run(a, fb, counting)
check(states[0]["finished"] and calls == 2, f"finishes after the capped steps ({calls} steps)")

# Constrained + read-only: emitted tokens in the set, logprobs renormalized.
allowed = [3, 7, 11]
a = algo(max_steps=4)
fb = batch([{"seed_canvas": seed, "pinned": [0], "read_only": True, "constrained": True, "max_steps": 3, "_token_ids": allowed}])
states = run(a, fb, noisy)
out = fb.input_ids.view(1, B)[0].tolist()
check(out[0] == seed[0] and all(t in allowed for t in out[1:]), f"constrained read emits only allowed tokens off the pin ({out})")
read = states[0]["read_logprobs"]
check(read is not None and len(read[0]) == B and len(read[0][0]) == len(allowed), "read-only state carries per-position logprobs for the requested ids")
sums = [sum(torch.tensor(row).exp().tolist()) for row in read[0][1:]]
check(all(abs(s - 1.0) < 1e-4 for s in sums), f"constrained logprobs renormalize over the set (sums {['%.3f' % s for s in sums[:3]]})")
check(all(abs(read[1][p] - max(read[0][p])) < 1e-5 for p in range(1, B)), "emitted token's logprob is the max over the set")

# Read-only without constraint: logprobs are the full-vocabulary values.
a = algo(max_steps=4)
fb = batch([{"seed_canvas": seed, "read_only": True, "max_steps": 1, "_token_ids": allowed}])
states = run(a, fb, peaked_logits)
read = states[0]["read_logprobs"]
check(read is not None and all(sum(torch.tensor(r).exp().tolist()) < 1.0 for r in read[0]), "unconstrained read keeps full-vocabulary logprobs")

# A request without options behaves as before.
a = algo()
fb = batch([None])
states = run(a, fb, peaked_logits)
check(states[0]["pinned"] is None and not states[0]["read_only"] and states[0]["read_logprobs"] is None, "no options: plain renoise state")

# The base loop hands read states back only when a read is present.
check(DllmAlgorithm._run_sync.__code__.co_varnames.__contains__("read_states"), "base loop returns read states")

# Option validation.
req = SimpleNamespace(dllm_config=SimpleNamespace(block_size=B, first_done_first_out_mode=False), return_logprob=True,
                      logprob=SimpleNamespace(token_ids_logprob=allowed))
check(_validate_diffusion_options({"seed_canvas": seed, "pinned": [0], "max_steps": 2, "read_only": True, "constrained": True}, req) is None, "valid options accepted")
check(_validate_diffusion_options({"seed_canvas": seed[:-1]}, req) is not None, "wrong canvas length rejected")
check(_validate_diffusion_options({"seed_canvas": seed, "pinned": [B]}, req) is not None, "pin out of range rejected")
check(_validate_diffusion_options({"pinned": [0]}, req) is not None, "pins without a seed canvas rejected")
check(_validate_diffusion_options({"bogus": 1}, req) is not None, "unknown option rejected")
req2 = SimpleNamespace(dllm_config=SimpleNamespace(block_size=B, first_done_first_out_mode=False), return_logprob=False, logprob=SimpleNamespace(token_ids_logprob=None))
check(_validate_diffusion_options({"constrained": True}, req2) is not None, "constrained without token_ids_logprob rejected")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
