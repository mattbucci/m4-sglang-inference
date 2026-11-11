"""Tiny random mlx-vlm Qwen3.5 language model and a bare runner around it.

Shared by the per-request position delta test and the batched mRoPE decode
test. Float32 weights on the Metal device; no server, no download.
"""
import os
import types

import numpy as np

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map_with_path  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as lang  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402

from sglang.srt.hardware_backend.mlx import model_runner as mr  # noqa: E402
from sglang.srt.hardware_backend.mlx.aot import MlxAOTKernelSet  # noqa: E402
from sglang.srt.hardware_backend.mlx.kv_cache.layout import (  # noqa: E402
    MlxModelCacheLayout,
)
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (  # noqa: E402
    find_attention_layers,
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.sdpa_blocks import SdpaBlockLadder  # noqa: E402

IMAGE, VIDEO, VISION_START = 208, 209, 210
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_lm(seed=7):
    args = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=96,
        intermediate_size=192,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=8,
        num_attention_heads=3,
        rms_norm_eps=1e-6,
        vocab_size=211,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 100.0,
            "partial_rotary_factor": 0.5,
        },
        full_attention_interval=4,
    )
    config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2),
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=VISION_START,
    )
    mx.random.seed(seed)
    lm = lang.LanguageModel(args, config)

    # Default inits leave the attention path too faint to tell positions
    # apart; widen every weight matrix so the logits depend on them.
    def widen(_, value):
        return value * 4.0 if value.ndim >= 2 else value

    lm.update(tree_map_with_path(widen, lm.parameters()))
    lm.set_dtype(mx.float32)
    lm.eval()
    mx.eval(lm.parameters())
    return lm


class TinyVLM:
    """What the runner's VLM shim looks like from the decode paths."""

    def __init__(self, lm):
        self.language_model = lm

    def __call__(self, input_ids, cache=None, **kwargs):
        return self.language_model(input_ids, cache=cache, **kwargs).logits

    def make_cache(self):
        return self.language_model.make_cache()


def make_runner(lm, max_seq_len=256):
    vlm = TinyVLM(lm)
    patch_model_attention(vlm)
    layers, attrs = find_attention_layers(vlm)
    r = mr.MlxModelRunner.__new__(mr.MlxModelRunner)
    r.model = vlm
    r.is_vlm = True
    r._cache_layout = MlxModelCacheLayout.from_attention_discovery(layers, attrs)
    r._max_seq_len = max_seq_len
    r._sdpa_block_ladder = SdpaBlockLadder(False)
    r._aot_kernels = MlxAOTKernelSet()
    r._attention_kv_pool = None
    r._req_pool_idx = {}
    r._req_to_token_pool = None
    r._req_rope_delta = {}
    r._req_position_delta = {}
    r._attention_wrapper_batchable = False
    r._mrope_batched_decode = False
    r._model_embed, r._model_norm, r._model_lm_head = r._extract_model_components()
    return r


class Prompt:
    """Token ids plus the image grid (t, h, w in patches) they carry, if any."""

    def __init__(self, rid, ids, grid=None):
        self.rid, self.ids, self.grid = rid, ids, grid


def text_prompt(rid, rng, n):
    return Prompt(rid, [int(t) for t in rng.integers(0, 200, n)])


def image_prompt(rid, rng, head, grid, tail):
    t, h, w = grid
    n = t * (h // 2) * (w // 2)
    ids = (
        [int(x) for x in rng.integers(0, 200, head)]
        + [VISION_START]
        + [IMAGE] * n
        + [int(x) for x in rng.integers(0, 200, tail)]
    )
    return Prompt(rid, ids, grid)


def forward_prompt(r, prompt, cache, ids=None, first=True):
    """One prompt forward the way prefill_start / extend_start run it."""
    if first:
        r._reset_mlx_vlm_position_cache()
        r._req_rope_delta.pop(prompt.rid, None)
        r._req_position_delta.pop(prompt.rid, None)
    else:
        r._restore_position_delta(prompt.rid)
    kwargs = {}
    if prompt.grid is not None and first:
        kwargs["image_grid_thw"] = mx.array([list(prompt.grid)])
    logits = r.model(mx.array([prompt.ids if ids is None else ids], dtype=mx.int32), cache=cache, **kwargs)
    r._keep_position_delta(prompt.rid)
    mx.eval(logits, [c.state for c in cache])
    return int(np.array(logits[0, -1].astype(mx.float32)).argmax())


def rows(logits, caches):
    mx.eval(logits, [c.state for cache in caches for c in cache])
    return np.array(logits.astype(mx.float32))
