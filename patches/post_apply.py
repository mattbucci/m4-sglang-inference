#!/usr/bin/env python3
"""Post-patch source edits for SGLang MLX backend.

Replaces patches 006 and 007. The hand-written .patch files for those
fail `git apply --check` (corrupt diff format), so we apply the same
changes via idempotent in-place edits here.

Patches:
  006-mlx-offsetcache-subscript: add __getitem__/__setitem__/lengths/advance
                                  stubs to OffsetCache so DeltaNet hybrids
                                  (Qwen3.5/Coder-Next) don't AttributeError.
  007-mlx-vlm-fallback: route VLM model loads through mlx_vlm.load with
                        a TextOnlyVLMShim so SGLang can load (text-only
                        for now) Idefics3/SmolVLM/Qwen2-VL etc.

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
            _vlm_model = self.model
            class _TextOnlyVLMShim:
                def __init__(self, vlm):
                    self._vlm = vlm
                def __call__(self, input_ids, cache=None, **kwargs):
                    out = self._vlm.language_model(input_ids, cache=cache)
                    # mlx_vlm wraps in LanguageModelOutput(logits=...).
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
