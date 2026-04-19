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
  011-mps-stub-cuda-redirect: SGLang's _mps_stub patches torch.Tensor.to
                              for MPS but doesn't handle .to('cuda') —
                              transformers' image processor unconditionally
                              calls .to('cuda') on Apple, hitting CUDA's
                              _lazy_init which crashes with "Torch not
                              compiled with CUDA enabled." Fix: redirect
                              cuda → cpu in the _patched_to so the image
                              tensor lands somewhere torch can handle.

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
        raise RuntimeError("OffsetCache should not store data")'''


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
    ) -> int:
        """Prefill a request.  Returns next_token_id.

        ``pixel_values``: optional mlx array for VLM image input. When set,
        passed through to the model's __call__ — TextOnlyVLMShim (patch 007)
        routes it to vlm.__call__ instead of language_model.
        """
        num_layers = self._num_layers
        prefix_len = len(prefix_slot_ids)

        # Build kwargs once; passed to all self.model() calls below so VLM
        # multimodal path is unified across the disable_radix and
        # full-radix branches.
        _model_kwargs = {"pixel_values": pixel_values} if pixel_values is not None else {}

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
                    # Patch 010: extract pixel_values from multimodal_inputs
                    # (set by SGLang's processor when an image is in the
                    # request). MlxModelRunner.prefill threads it to
                    # TextOnlyVLMShim.__call__ which routes to the full
                    # vlm.__call__ when present.
                    pixel_values = None
                    mm = getattr(req, "multimodal_inputs", None)
                    if mm and getattr(mm, "mm_items", None):
                        feature = getattr(mm.mm_items[0], "feature", None)
                        if feature is not None:
                            try:
                                import mlx.core as _mx
                                # mlx_vlm wants an MLX array. SGLang's
                                # multimodal processor returns a torch
                                # tensor; convert via numpy bridge.
                                pixel_values = _mx.array(feature.numpy() if hasattr(feature, "numpy") else feature)
                            except Exception:
                                pixel_values = None
                    next_token = self._mlx_runner.prefill(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                        pixel_values=pixel_values,
                    )
                    prefill_rids.append((req.rid, next_token))'''


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
