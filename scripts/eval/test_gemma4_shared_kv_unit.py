#!/usr/bin/env python3
"""Unit test for patch 042: mlx_vlm's Gemma 4 on the MLX attention contract.

Gemma 4 (26B/31B) mixes two kinds of attention layer. The sliding-window ones
have 16 KV heads of 256 and their own value projection; the full-attention
ones have 4 KV heads of 512 and take their values from the key projection
(``attention_k_eq_v``), so they carry no ``v_proj``. Covers the four things the
patch changes: the contract accepts a layer without ``v_proj`` only when it
declares the shared projection; the per-layer window map is read from the
container's ``config``; the uniform-KV-shape check spans the pooled
(full-attention) layers only; and the batched-decode wrapper records the
Gemma-specific value handling at patch time. CPU only, no checkpoint.

    python scripts/eval/test_gemma4_shared_kv_unit.py
"""
import os
import sys
import types

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache import attention_contract as contract
from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import MLXAttentionWrapper
from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


HIDDEN = 64


class GemmaAttention(nn.Module):
    """The attribute surface of mlx_vlm's Gemma 4 ``Attention``."""

    def __init__(self, sliding: bool):
        super().__init__()
        self.is_sliding = sliding
        self.use_k_eq_v = not sliding
        self.n_heads = 4
        self.n_kv_heads = 2 if sliding else 1
        self.head_dim = 8 if sliding else 16
        self.scale = 1.0
        self.q_proj = nn.Linear(HIDDEN, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(HIDDEN, self.n_kv_heads * self.head_dim, bias=False)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(HIDDEN, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, HIDDEN, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.v_norm = nn.RMSNorm(self.head_dim)
        self.rope = nn.RoPE(self.head_dim)
        self.is_kv_shared_layer = False


class PlainAttention(nn.Module):
    """An ordinary mlx-lm attention module."""

    def __init__(self):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim, self.scale = 4, 2, 8, 8**-0.5
        self.q_proj = nn.Linear(HIDDEN, 32, bias=False)
        self.k_proj = nn.Linear(HIDDEN, 16, bias=False)
        self.v_proj = nn.Linear(HIDDEN, 16, bias=False)
        self.o_proj = nn.Linear(32, HIDDEN, bias=False)
        self.rope = nn.RoPE(8)


sliding, full, plain = GemmaAttention(True), GemmaAttention(False), PlainAttention()

# --- the contract ---------------------------------------------------------------
check(contract.is_attention_module(sliding), "a Gemma 4 sliding layer (has v_proj) is attention")
check(contract.is_attention_module(full), "a Gemma 4 full layer (k_eq_v, no v_proj) is attention")
check(contract.shares_key_value_projection(full) and not contract.shares_key_value_projection(sliding),
      "only the layer without v_proj that declares k_eq_v shares its projection")
check(contract.is_attention_module(plain), "an ordinary attention module is still attention")

headless = PlainAttention()
del headless.v_proj
check(not contract.is_attention_module(headless),
      "a module without v_proj that does not declare k_eq_v is not attention")
declared = PlainAttention()
declared.use_k_eq_v = True
check(contract.is_attention_module(declared) and not contract.shares_key_value_projection(declared),
      "use_k_eq_v on a module that keeps v_proj changes nothing")

# --- the window map --------------------------------------------------------------
layer_types = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]
text_model = types.SimpleNamespace(
    window_size=1024, config=types.SimpleNamespace(layer_types=layer_types, sliding_window=1024)
)
gemma = types.SimpleNamespace(language_model=types.SimpleNamespace(model=text_model))
check(contract.get_layer_window_sizes(gemma) == {0: 1024, 1: 1024, 2: None, 3: 1024},
      "layer_types on the container's config gives the per-layer window map")
args_model = types.SimpleNamespace(args=types.SimpleNamespace(layer_types=layer_types, sliding_window=512))
check(contract.get_layer_window_sizes(types.SimpleNamespace(model=args_model))[0] == 512,
      "layer_types on args is still read")
check(contract.get_layer_window_sizes(types.SimpleNamespace(model=types.SimpleNamespace())) == {},
      "a model that declares nothing has no window map")

# --- the uniform-shape check spans pooled layers only ----------------------------


def runner(full_idx, attn_idx, shapes):
    fake = types.SimpleNamespace()
    fake._cache_layout = types.SimpleNamespace(
        num_attention_layers=len(attn_idx),
        full_attention_layer_indices=tuple(full_idx),
        attention_layer_indices=tuple(attn_idx),
    )
    fake._attention_kv_config_for_layer = lambda i: shapes[i]
    fake._pooled_attention_layer_indices = lambda: MlxModelRunner._pooled_attention_layer_indices(fake)
    return fake


SW, FULL = (16, 256, mx.bfloat16), (4, 512, mx.bfloat16)
mixed = runner([2, 5], [0, 1, 2, 3, 4, 5], {0: SW, 1: SW, 2: FULL, 3: SW, 4: SW, 5: FULL})
check(MlxModelRunner._pooled_attention_layer_indices(mixed) == (2, 5), "the pooled layers are the full-attention ones")
check(MlxModelRunner._get_attn_config(mixed) == FULL,
      "sliding layers of another shape do not block the pool; its shape is the full layers'")

all_sliding = runner([], [0, 1], {0: SW, 1: SW})
check(MlxModelRunner._get_attn_config(all_sliding) == SW, "an all-sliding model falls back to every attention layer")

uneven = runner([0, 1], [0, 1], {0: FULL, 1: SW})
try:
    MlxModelRunner._get_attn_config(uneven)
    check(False, "two shapes among the pooled layers are refused")
except NotImplementedError:
    check(True, "two shapes among the pooled layers are refused")

# --- what the wrapper records at patch time --------------------------------------
w_full, w_sliding, w_plain = (
    MLXAttentionWrapper(full, 2),
    MLXAttentionWrapper(sliding, 0, window_size=1024),
    MLXAttentionWrapper(plain, 0),
)
check(w_full._k_eq_v and not w_sliding._k_eq_v and not w_plain._k_eq_v,
      "only the full Gemma layer takes its values from the key projection")
check(w_full._has_v_norm and w_sliding._has_v_norm and not w_plain._has_v_norm,
      "both Gemma layers normalise their values; an ordinary module does not")
check(w_full._returns_kv_tuple and w_sliding._returns_kv_tuple and not w_plain._returns_kv_tuple,
      "Gemma layers get the (output, shared_kv, offset) tuple; an ordinary module a tensor")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
