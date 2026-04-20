#!/usr/bin/env python3
"""Post-patch source edits for SGLang MLX backend.

Replaces patches 006, 007, 008. The hand-written .patch files for those
fail `git apply --check` (corrupt diff format), so we apply the same
changes via idempotent in-place edits here.

Patches:
  006-mlx-offsetcache-subscript: add __getitem__/__setitem__/lengths/advance
                                  stubs to OffsetCache so DeltaNet hybrids
                                  (Qwen3.5/Coder-Next) don't AttributeError.
  007-mlx-vlm-fallback: route VLM model loads through mlx_vlm.load with
                        a TextOnlyVLMShim so SGLang can load (text-only
                        for now) Idefics3/SmolVLM/Qwen2-VL etc.
  008-mlx-hybrid-serial-decode: detect DeltaNet-hybrid caches in
                                decode_batch and route to per-request
                                serial decode path. Stopgap that
                                actually works (Qwen3.5 + Coder-Next
                                concurrent decode now produces correct
                                outputs); proper fix needs per-request
                                state stacked into batched cache.
  009-modality-multi-images: add MULTI_IMAGES member to Modality enum
                             that SGLang's transformers_auto multimodal
                             processor references but doesn't define.
                             Without it, ANY image-bearing request 500s
                             with AttributeError before reaching the
                             model. Upstream SGLang bug.
  010-mlx-vlm-pixel-values: thread pixel_values from
                            req.multimodal_inputs through tp_worker →
                            MlxModelRunner.prefill → TextOnlyVLMShim
                            so VLM models can actually do image
                            inference (not just text-only). Additive
                            change — text-only path unchanged, only
                            triggers when multimodal_inputs is set.

  Update (full): tp_worker now also extracts model_specific_data
  (image_grid_thw etc.) and passes via mm_extra_kwargs; prefill
  threads through; TextOnlyVLMShim forwards via **kwargs to
  vlm.__call__. Verified end-to-end with Qwen2-VL-2B: response
  "A redcircle" for synthetic red circle image.
  011-mps-stub-cuda-redirect: SGLang's _mps_stub patches torch.Tensor.to
                              for MPS but doesn't handle .to('cuda') —
                              transformers' image processor unconditionally
                              calls .to('cuda') on Apple, hitting CUDA's
                              _lazy_init which crashes with "Torch not
                              compiled with CUDA enabled." Fix: redirect
                              cuda → cpu in the _patched_to so the image
                              tensor lands somewhere torch can handle.

  012-mm-utils-shm-page-rounding: SharedMemory page-rounds size on macOS
                                  (16 KB pages on M-series), so
                                  torch.frombuffer(shm.buf, ...) produces
                                  a tensor LARGER than the logical nbytes.
                                  Fix: slice to logical size on both
                                  the write and read sides.

  Plus: VLM detection in _load_model now reads config.json via
  hf_hub_download to detect vision_config / image_token_id and force
  mlx_vlm.load even when mlx_lm.load would have succeeded (Qwen2-VL,
  Qwen3-VL). Without this, mlx_lm loads the model but its __call__ has
  no pixel_values param and just produces text-only output.

Run as:
  python patches/post_apply.py <SGLANG_DIR>
"""
from __future__ import annotations

import sys
from pathlib import Path


# -- Patch 006: OffsetCache stubs --

OFFSETCACHE_FILE = "python/sglang/srt/hardware_backend/mlx/kv_cache/contiguous_cache.py"

OFFSETCACHE_OLD = '''    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        return ()  # Empty -- safe for mx.eval unpacking

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(self, keys, values):
        raise RuntimeError("OffsetCache should not store data")'''

OFFSETCACHE_NEW = '''    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        return ()  # Empty -- safe for mx.eval unpacking

    def __getitem__(self, key):
        # mlx-lm DeltaNet/Mamba layers do `cache[0] is not None` to detect
        # first-call vs resumed state. None = "no cached state, use cold path".
        return None

    def __setitem__(self, key, value):
        # DeltaNet writes its updated state back here. We don't persist;
        # this is a stopgap (loses recurrent state across batched decode steps).
        pass

    def __len__(self):
        return 0

    # mlx-lm DeltaNet probes `cache.lengths is not None`.
    lengths = None

    def advance(self, S):
        """No-op stub — DeltaNet calls this after writing state."""
        pass

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(self, keys, values):
        raise RuntimeError("OffsetCache should not store data")


class ContiguousKVCache:'''


# Patch 006 also needs to add the same subscript stubs to ContiguousKVCache
# so DeltaNet hybrid VLMs (Qwen3.5/3.6) don't TypeError when their
# linear_attn layer probes cache[0] during single-batch decode. We extend
# the existing class definition rather than replace it (the class body is
# created by patch 001).
CONTIGUOUS_OLD = '''class ContiguousKVCache:
    """Pre-allocated KV buffer for one request x one layer.

    Shape ``(1, n_kv_heads, max_seq_len, head_dim)``.  Slice assignment
    instead of ``mx.concatenate``.  Lazy-allocated on first write.

    When kv_cache_mode is FP8 or TURBOQUANT, K/V data is stored in
    quantized form and dequantized on read.
    """

    __slots__ = ('''

CONTIGUOUS_NEW = '''class ContiguousKVCache:
    """Pre-allocated KV buffer for one request x one layer.

    Shape ``(1, n_kv_heads, max_seq_len, head_dim)``.  Slice assignment
    instead of ``mx.concatenate``.  Lazy-allocated on first write.

    When kv_cache_mode is FP8 or TURBOQUANT, K/V data is stored in
    quantized form and dequantized on read.

    DeltaNet subscript protocol stubs (see OffsetCache for rationale):
    needed so a hybrid VLM (Qwen3.5/3.6, Coder-Next) can use this cache
    for both attention layers (real KV) and DeltaNet layers (cold path
    each step).
    """

    lengths = None

    def __getitem__(self, key):
        return None

    def __setitem__(self, key, value):
        pass

    def __len__(self):
        return 0

    def advance(self, S):
        pass

    __slots__ = ('''


# -- Patch 007: mlx_vlm fallback in MlxModelRunner._load_model --

MODEL_RUNNER_FILE = "python/sglang/srt/hardware_backend/mlx/model_runner.py"

MODEL_RUNNER_OLD = '''    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )
        # Force-evaluate weights so mx.get_active_memory() reflects
        # actual usage before KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")'''

MODEL_RUNNER_NEW = '''    def _load_model(self):
        """Load model using mlx_lm; fall back to mlx_vlm for VLM architectures."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        try:
            self.model, _ = mlx_lm_load(
                self.model_path,
                tokenizer_config={"trust_remote_code": self.trust_remote_code},
            )
            self.is_vlm = False
        except ValueError as e:
            # mlx_lm raises ValueError("Model type X not supported.") for VLMs
            # like Idefics3/SmolVLM/Qwen2-VL. Try mlx_vlm as a fallback.
            if "not supported" not in str(e):
                raise
            logger.warning(
                f"mlx_lm cannot load {self.model_path} ({e}); "
                f"falling back to mlx_vlm. VLM inference path through "
                f"SGLang is text-only for now (no pixel_values plumbing)."
            )
            try:
                from mlx_vlm import load as mlx_vlm_load
            except ImportError:
                raise RuntimeError(
                    f"Model {self.model_path} requires mlx_vlm but it isn't "
                    f"installed. Run `pip install mlx-vlm`."
                ) from e
            self.model, self.processor = mlx_vlm_load(self.model_path)
            self.is_vlm = True
            # Wrap so calls without pixel_values delegate to language_model.
            # TODO(patch-010): when pixel_values is non-None, route through
            # self._vlm(input_ids, pixel_values, cache=cache) instead.
            # Requires plumbing pixel_values from
            # model_worker_batch.mm_input through tp_worker.forward_batch_generation
            # → MlxModelRunner.prefill → here. SGLang's multimodal processor
            # (after patch 009) extracts pixel_values into mm_input but our
            # prefill signature drops it.
            _vlm_model = self.model
            class _TextOnlyVLMShim:
                def __init__(self, vlm):
                    self._vlm = vlm
                def __call__(self, input_ids, cache=None, pixel_values=None, **kwargs):
                    if pixel_values is not None:
                        # Patch-010 path: real multimodal forward via vlm.__call__.
                        # Once tp_worker.forward_batch_generation passes
                        # pixel_values through MlxModelRunner.prefill, this
                        # branch will produce actual image-aware output.
                        out = self._vlm(input_ids, pixel_values, cache=cache)
                    else:
                        # Text-only path: bypass image embeddings.
                        out = self._vlm.language_model(input_ids, cache=cache)
                    return getattr(out, "logits", out)
                def __getattr__(self, name):
                    return getattr(self._vlm, name)
            self.model = _TextOnlyVLMShim(_vlm_model)

        # Force-evaluate weights so mx.get_active_memory() reflects
        # actual usage before KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(
            f"MLX model loaded in {load_time:.2f}s "
            f"(vlm={getattr(self, 'is_vlm', False)})"
        )'''


# -- Patch 008: hybrid serial decode fallback --

DECODE_BATCH_OLD = '''        seq_lens = [_get_offset(caches[i]) for i in range(batch_size)]

        if batch_size == 1:'''

DECODE_BATCH_NEW = '''        seq_lens = [_get_offset(caches[i]) for i in range(batch_size)]

        # Hybrid models (DeltaNet/Mamba) need per-request recurrent state in
        # cache[layer]. Our batched-decode shim_cache (OffsetCache) discards
        # that state, causing reshape errors and missing-attribute crashes.
        # Stopgap: detect hybrid layers and force serial decode. Real fix
        # would stack per-request DeltaNet state into batched cache.
        is_hybrid = batch_size > 1 and any(
            not isinstance(c, ContiguousKVCache) and not isinstance(c, OffsetCache)
            for c in caches[0]
        )
        if is_hybrid:
            results = []
            for rid in req_ids:
                serial = self.decode_batch([rid])
                results.append(serial[0])
            return results

        if batch_size == 1:'''


# -- Patch 009: add MULTI_IMAGES to Modality enum --

MODALITY_FILE = "python/sglang/srt/managers/schedule_batch.py"

MODALITY_OLD = '''class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()'''

MODALITY_NEW = '''class Modality(Enum):
    IMAGE = auto()
    MULTI_IMAGES = auto()  # M4 patch 009: SGLang's transformers_auto.py:133
                           # references this; without it, multimodal requests
                           # crash with AttributeError before reaching the model.
    VIDEO = auto()
    AUDIO = auto()'''

MODALITY_ALL_OLD = '''        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]'''

MODALITY_ALL_NEW = '''        return [Modality.IMAGE, Modality.MULTI_IMAGES, Modality.VIDEO, Modality.AUDIO]'''


# -- Patch 010: pixel_values plumbing in prefill() and tp_worker --

PREFILL_OLD = '''    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> int:
        """Prefill a request.  Returns next_token_id."""
        num_layers = self._num_layers
        prefix_len = len(prefix_slot_ids)

        if self.disable_radix_cache:
            cache = self._acquire_cache()
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)'''

PREFILL_NEW = '''    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        pixel_values=None,
        mm_extra_kwargs=None,
    ) -> int:
        """Prefill a request.  Returns next_token_id.

        ``pixel_values``: optional mlx array for VLM image input.
        ``mm_extra_kwargs``: optional dict of model-specific multimodal
        kwargs (e.g. image_grid_thw for Qwen2-VL).
        """
        num_layers = self._num_layers
        prefix_len = len(prefix_slot_ids)

        _model_kwargs: dict = {}
        if pixel_values is not None and getattr(self, "is_vlm", False):
            _model_kwargs["pixel_values"] = pixel_values
            if mm_extra_kwargs:
                _model_kwargs.update(mm_extra_kwargs)

        if self.disable_radix_cache:
            cache = self._acquire_cache()
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache, **_model_kwargs)'''

# The model_output= line later in prefill (with cache=cache only) needs the
# same kwargs treatment. Pattern is unique within prefill body.
PREFILL_OLD2 = '''        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)'''

PREFILL_NEW2 = '''        model_output = self.model(input_ids, cache=cache, **_model_kwargs)
        logits = self._extract_logits(model_output)'''


TP_WORKER_FILE = "python/sglang/srt/hardware_backend/mlx/tp_worker.py"

TP_WORKER_OLD = '''                else:
                    # New prefill
                    prefix_slot_ids = req.prefix_indices.tolist()
                    full_token_ids = list(req.fill_ids)
                    next_token = self._mlx_runner.prefill(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                    )
                    prefill_rids.append((req.rid, next_token))'''

MPS_STUB_OLD = '''    def _patched_to(self, *args, **kwargs):
        if kwargs.get("non_blocking"):
            # Detect target device from positional or keyword args
            device = None
            if args and isinstance(args[0], (str, torch.device)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
            elif "device" in kwargs:
                d = kwargs["device"]
                device = torch.device(d) if isinstance(d, str) else d
            if device is not None and device.type == "mps":
                kwargs = {**kwargs, "non_blocking": False}
        return _original_to(self, *args, **kwargs)'''

MM_UTILS_FILE = "python/sglang/srt/managers/mm_utils.py"

MM_UTILS_WRITE_OLD = '''        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dst.copy_(tensor.view(torch.uint8).reshape(-1))'''

MM_UTILS_WRITE_NEW = '''        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            # M4 patch 012: macOS rounds shm size up to a page (16 KB on
            # Apple Silicon), so torch.frombuffer(shm.buf, ...) yields a
            # tensor LARGER than nbytes. Slice dst to the actual logical size.
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)[:nbytes]
            dst.copy_(tensor.view(torch.uint8).reshape(-1))'''

MM_UTILS_READ_OLD = '''    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.shm = None
        self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)
        # Zero-copy view into shared memory (no clone, no unlink)
        self.tensor = torch.frombuffer(self._shm_handle.buf, dtype=self.dtype).reshape(
            self.shape
        )'''

MM_UTILS_READ_NEW = '''    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.shm = None
        self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)
        # M4 patch 012 (read side): slice to logical element count before
        # reshape — macOS page-rounds shm allocations.
        n_elements = 1
        for d in self.shape:
            n_elements *= d
        # Zero-copy view into shared memory (no clone, no unlink)
        self.tensor = torch.frombuffer(self._shm_handle.buf, dtype=self.dtype)[
            :n_elements
        ].reshape(self.shape)'''


MPS_STUB_NEW = '''    def _patched_to(self, *args, **kwargs):
        # Detect target device from positional or keyword args
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
        elif "device" in kwargs:
            d = kwargs["device"]
            device = torch.device(d) if isinstance(d, str) else d

        if device is not None:
            if device.type == "cuda":
                # M4 patch 011: We have no CUDA. Redirect cuda→cpu so SGLang's
                # multimodal image processor (and other transformers code that
                # defaults to .to('cuda')) keeps working on Apple Silicon
                # without crashing in CUDA's _lazy_init.
                if args and isinstance(args[0], (str, torch.device)):
                    args = ("cpu",) + args[1:]
                else:
                    kwargs = {**kwargs, "device": "cpu"}
            elif device.type == "mps" and kwargs.get("non_blocking"):
                kwargs = {**kwargs, "non_blocking": False}

        return _original_to(self, *args, **kwargs)'''


TP_WORKER_NEW = '''                else:
                    # New prefill
                    prefix_slot_ids = req.prefix_indices.tolist()
                    full_token_ids = list(req.fill_ids)
                    # Patch 010 + 014: stack pixel_values across all
                    # mm_items (multi-image / video frames) and concat
                    # per-architecture model_specific_data
                    # (image_grid_thw, video_grid_thw, second_per_grid_ts).
                    # Without this, Qwen3.5/3.6 raise
                    # "Image features and image tokens do not match"
                    # because the prompt has N image tokens but the model
                    # only saw 1 image's features.
                    pixel_values = None
                    mm_extra_kwargs: dict = {}
                    mm = getattr(req, "multimodal_inputs", None)
                    if mm and getattr(mm, "mm_items", None):
                        try:
                            import mlx.core as _mx
                            _to_mx = lambda t: _mx.array(t.numpy() if hasattr(t, "numpy") else t)
                            features = []
                            ms_acc: dict = {}
                            for it in mm.mm_items:
                                f = getattr(it, "feature", None)
                                if f is None:
                                    continue
                                features.append(_to_mx(f))
                                ms = getattr(it, "model_specific_data", None) or {}
                                for k, v in ms.items():
                                    if v is None:
                                        continue
                                    ms_acc.setdefault(k, []).append(v)
                            if features:
                                pixel_values = _mx.concatenate(features, axis=0) if len(features) > 1 else features[0]
                                for k, vs in ms_acc.items():
                                    try:
                                        mxs = [_to_mx(v) for v in vs]
                                        mm_extra_kwargs[k] = _mx.concatenate(mxs, axis=0) if len(mxs) > 1 else mxs[0]
                                    except Exception:
                                        mm_extra_kwargs[k] = vs[0] if len(vs) == 1 else vs
                        except Exception:
                            pixel_values = None
                            mm_extra_kwargs = {}
                    next_token = self._mlx_runner.prefill(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                        pixel_values=pixel_values,
                        mm_extra_kwargs=mm_extra_kwargs or None,
                    )
                    prefill_rids.append((req.rid, next_token))'''


# -- Patch 014: ContiguousKVCache.make_mask handles offset > 0 --
#
# `make_mask` always returned the "causal" sentinel (which mlx_vlm
# expands to a square (N,N) mx.array). Fine for full prefill, broken
# for chunked-prefill continuation: keys have accumulated to (offset+N),
# so a (N,N) mask doesn't broadcast against (1, n_heads, N, offset+N)
# scores → ValueError([broadcast_shapes]). Crashes Gemma 4 31B-it
# (and any model that uses mlx_vlm's gemma4 attention) at any context
# > chunked_prefill_size. Build the explicit non-square causal mask
# when offset > 0; keep "causal" sentinel when offset == 0 so
# mx.fast.SDPA's optimized path is preserved on the hot first chunk.

CONTIGUOUS_MAKE_MASK_FILE = "python/sglang/srt/hardware_backend/mlx/kv_cache/contiguous_cache.py"

# The OLD sequence is the make_mask immediately before _grow — that
# pinpoints the ContiguousKVCache copy (OffsetCache and PoolBackedCache
# also have make_mask but with different surrounding context).
CONTIGUOUS_MAKE_MASK_OLD = '''    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def _grow(self, required: int) -> None:'''

CONTIGUOUS_MAKE_MASK_NEW = '''    def make_mask(self, N, **kwargs):
        # Decode (N==1): no mask needed.
        if N == 1:
            return None
        # Chunked-prefill continuation (offset > 0): the keys array length
        # is offset + N (previous chunks + current chunk). The "causal"
        # sentinel only works when keys are exactly N long; once chunks
        # accumulate, mlx_vlm gemma4's mask trim sees a square (N,N) mask
        # that doesn't match the (1, n_heads, N, offset+N) scores shape
        # → broadcast_shapes ValueError. Build the explicit non-square
        # causal mask: query i (0..N-1) attends to keys 0..offset+i.
        if self.offset > 0:
            S = self.offset + N
            window = kwargs.get("window_size")
            q_idx = self.offset + mx.arange(N)
            k_idx = mx.arange(S)
            mask = q_idx[:, None] >= k_idx[None, :]
            if window is not None:
                mask = mask & (q_idx[:, None] - k_idx[None, :] < window)
            return mask
        # Full prefill from scratch (offset == 0): square mask, "causal"
        # sentinel is fine and lets mx.fast.SDPA take the optimized path.
        return "causal"

    def _grow(self, required: int) -> None:'''


# -- Patch 013: hybrid cache via VLM language_model.make_cache --

# When the model is loaded as a VLM (mlx_vlm.load), the outer wrapper has no
# `make_cache` attribute — make_cache lives on `language_model`. The original
# _acquire_cache check (`hasattr(self.model, "make_cache")`) fell through to
# the uniform-ContiguousKVCache loop, giving DeltaNet layers the wrong cache
# type and producing fluent garbage tokens (Qwen3.5/3.6).
#
# Also resets DeltaNet ArraysCache state on cache-pool reuse so request N+1
# doesn't inherit request N's recurrent state.

ACQUIRE_CACHE_OLD = '''        if self._cache_pool:
            cache = self._cache_pool.pop()
            for c in cache:
                if hasattr(c, "offset"):
                    c.offset = 0
            return cache

        # Check if model has make_cache (hybrid models like DeltaNet)
        if hasattr(self.model, "make_cache"):
            native_cache = self.model.make_cache()'''

ACQUIRE_CACHE_NEW = '''        if self._cache_pool:
            cache = self._cache_pool.pop()
            # Reset whatever the cache type allows so a returned cache from
            # a previous request doesn't leak state into a new one.
            for c in cache:
                if hasattr(c, "offset"):
                    c.offset = 0  # ContiguousKVCache / KVCache / RotatingKVCache
                    # Patch 015: RotatingKVCache has extra ring-buffer state
                    if hasattr(c, "_idx"):
                        c._idx = 0
                elif hasattr(c, "reset"):
                    c.reset()
                elif hasattr(c, "cache"):
                    # ArraysCache (DeltaNet recurrent state) — drop arrays so
                    # next prefill rebuilds from scratch. Without this the
                    # next request inherits the previous request's
                    # conv_state/ssm_state and produces garbage.
                    try:
                        c.cache = [None for _ in c.cache]
                    except Exception:
                        pass
            return cache

        # Check if model has make_cache (hybrid models like DeltaNet).
        # For VLM-wrapped hybrid models (mlx_vlm.load on Qwen3.5/3.6),
        # make_cache lives on `language_model`, not the outer wrapper.
        # Without this fallback the loop below makes uniform ContiguousKVCache
        # for every layer — and DeltaNet layers receive the wrong cache
        # type, producing fluent-looking garbage tokens.
        native_maker = None
        if hasattr(self.model, "make_cache"):
            native_maker = self.model.make_cache
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "make_cache"):
            native_maker = self.model.language_model.make_cache

        if native_maker is not None:
            native_cache = native_maker()'''

# Also need to swap the closing `if hasattr(...)` -> `if native_maker is not None`
# but the entry is in the SAME block; the OLD->NEW above replaces the start of
# the block and rewrites the conditional. The body (substitution loop) is
# identical so we don't touch it.


def apply_edit(path: Path, old: str, new: str, label: str) -> None:
    """Idempotently replace `old` with `new` in `path`."""
    if not path.exists():
        print(f"  [SKIP] {label}: {path} does not exist")
        return
    src = path.read_text()
    if new in src:
        print(f"  [OK]   {label}: already applied")
        return
    if old not in src:
        print(f"  [WARN] {label}: pre-image not found, manual fix needed in {path}")
        return
    path.write_text(src.replace(old, new))
    print(f"  [OK]   {label}: applied")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python post_apply.py <SGLANG_DIR>", file=sys.stderr)
        return 1
    sglang = Path(sys.argv[1])
    if not (sglang / "python" / "sglang").is_dir():
        print(f"ERROR: {sglang} is not an SGLang source tree", file=sys.stderr)
        return 1

    print("Applying post-patch edits (006 + 007)...")
    apply_edit(sglang / OFFSETCACHE_FILE, OFFSETCACHE_OLD, OFFSETCACHE_NEW,
               "006-mlx-offsetcache-subscript")
    apply_edit(sglang / MODEL_RUNNER_FILE, MODEL_RUNNER_OLD, MODEL_RUNNER_NEW,
               "007-mlx-vlm-fallback")
    apply_edit(sglang / MODEL_RUNNER_FILE, DECODE_BATCH_OLD, DECODE_BATCH_NEW,
               "008-mlx-hybrid-serial-decode")
    apply_edit(sglang / MODALITY_FILE, MODALITY_OLD, MODALITY_NEW,
               "009-modality-multi-images (enum)")
    apply_edit(sglang / MODALITY_FILE, MODALITY_ALL_OLD, MODALITY_ALL_NEW,
               "009-modality-multi-images (all() helper)")
    apply_edit(sglang / MODEL_RUNNER_FILE, PREFILL_OLD, PREFILL_NEW,
               "010-mlx-vlm-pixel-values (prefill signature)")
    apply_edit(sglang / MODEL_RUNNER_FILE, PREFILL_OLD2, PREFILL_NEW2,
               "010-mlx-vlm-pixel-values (radix branch model call)")
    apply_edit(sglang / TP_WORKER_FILE, TP_WORKER_OLD, TP_WORKER_NEW,
               "010-mlx-vlm-pixel-values (tp_worker plumbing)")
    apply_edit(sglang / "python/sglang/_mps_stub.py", MPS_STUB_OLD, MPS_STUB_NEW,
               "011-mps-stub-cuda-redirect")
    apply_edit(sglang / MM_UTILS_FILE, MM_UTILS_WRITE_OLD, MM_UTILS_WRITE_NEW,
               "012-mm-utils-shm-page-rounding (write)")
    apply_edit(sglang / MM_UTILS_FILE, MM_UTILS_READ_OLD, MM_UTILS_READ_NEW,
               "012-mm-utils-shm-page-rounding (read)")
    apply_edit(sglang / MODEL_RUNNER_FILE, ACQUIRE_CACHE_OLD, ACQUIRE_CACHE_NEW,
               "013-mlx-hybrid-cache-via-vlm-language-model")
    apply_edit(sglang / CONTIGUOUS_MAKE_MASK_FILE, CONTIGUOUS_MAKE_MASK_OLD, CONTIGUOUS_MAKE_MASK_NEW,
               "014-contiguous-cache-make-mask-handles-offset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
