#!/usr/bin/env python3
"""Eager-import boot-chain smoke over every patch-touched module (R9700 donor).

The setup.sh patch loop can only prove a patch APPLIED; a mis-applied or
half-reverted module still imports lazily at serve time — this walks every
module the patch stack touches (inventory generated from patches/*.patch, not
hardcoded) plus the load-bearing symbols each patch introduces, and exits 1 on
any failure. CPU-only; run inside the venv with SGLANG_USE_MLX=1.

Usage: SGLANG_USE_MLX=1 python scripts/eval/import_smoke.py
"""
import glob
import importlib
import os
import re
import sys

os.environ.setdefault("SGLANG_USE_MLX", "1")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCH_DIR = os.environ.get("PATCH_DIR", os.path.join(REPO, "patches"))

# Load-bearing symbols per patched module (checked when the module appears in
# the generated inventory). Each entry maps to the patch that introduces it.
def _own(cls, name):
    """Require ``name`` defined on ``cls`` itself, not inherited — an override
    check that passes via the base class would prove nothing."""
    if name not in vars(cls):
        raise AttributeError(f"{cls.__name__}.{name} is inherited, not overridden")
    return vars(cls)[name]


def _src_has(module, needle):
    import inspect

    if needle not in inspect.getsource(module):
        raise AttributeError(f"{needle!r} not in {module.__name__} source")
    return True


ATTR_CHECKS = {
    "sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper": [
        ("MLXAttentionWrapper.__getattr__",                            # 005
         lambda m: _own(m.MLXAttentionWrapper, "__getattr__")),
    ],
    "sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache": [
        ("ContiguousAttentionKVCache.reserve",                         # 015
         lambda m: m.ContiguousAttentionKVCache.reserve),
        ("WindowedAttentionKVCache.snapshot_state",                    # 021
         lambda m: m.WindowedAttentionKVCache.snapshot_state),
    ],
    "sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state": [
        ("MlxAuxiliaryStateComponent.finalize_match_result_in_cache",  # 008
         lambda m: _own(m.MlxAuxiliaryStateComponent,
                        "finalize_match_result_in_cache")),
    ],
    "sglang.srt.hardware_backend.mlx.model_runner": [
        ("_CompactCacheModel", lambda m: m._CompactCacheModel),        # 008
        ("_NullLayerCache", lambda m: m._NullLayerCache),              # 008
        ("MlxModelRunner._load_vlm_model",                             # 008
         lambda m: m.MlxModelRunner._load_vlm_model),
        ("MlxModelRunner._reset_mlx_vlm_position_cache",               # 008
         lambda m: m.MlxModelRunner._reset_mlx_vlm_position_cache),
        ("MlxModelRunner._last_position_logits",                        # 020
         lambda m: m.MlxModelRunner._last_position_logits),
        ("MlxModelRunner.snapshots_sliding_window_prefix",             # 021
         lambda m: m.MlxModelRunner.snapshots_sliding_window_prefix),
    ],
    "sglang.srt.hardware_backend.mlx.swa_prefix": [
        ("mlx_snapshots_sliding_window_prefix",                        # 021
         lambda m: m.mlx_snapshots_sliding_window_prefix),
    ],
    "sglang.srt.mem_cache.kv_cache_builder": [
        ("_mlx_sliding_window_snapshot_state",                         # 021
         lambda m: m._mlx_sliding_window_snapshot_state),
    ],
    "sglang.srt.hardware_backend.mlx.tp_worker": [
        ("MlxTpModelWorker._extract_mm_inputs",                        # 008
         lambda m: m.MlxTpModelWorker._extract_mm_inputs),
    ],
    "sglang.srt.utils.hf_transformers.tokenizer": [
        ("_declared_tokenizer_class",                                  # 008
         lambda m: m._declared_tokenizer_class),
    ],
    "sglang.srt.utils.hf_transformers.processor": [
        ("_build_gemma4_image_only_processor",                         # 014
         lambda m: m._build_gemma4_image_only_processor),
    ],
    "sglang.srt.arg_groups.fields.model": [
        ("kv_cache_dtype MLX aliases", lambda m: _src_has(m, '"turboquant"')),  # 002
    ],
    "sglang.srt.mem_cache.allocation": [
        ("MLX aux-slot admission eviction",                            # 018
         lambda m: _src_has(m, "MLX auxiliary-state pool")),
    ],
    "sglang.srt.hardware_backend.mlx.text_checkpoint": [
        ("load_text_model", lambda m: m.load_text_model),              # 019
    ],
    "sglang.srt.hardware_backend.mlx.kv_cache.attention_contract": [
        ("_window_declaration", lambda m: m._window_declaration),      # 019
    ],
    "sglang.srt.configs.flint": [
        ("FlintConfig", lambda m: m.FlintConfig),                      # 019
    ],
    "sglang.srt.utils.hf_transformers.common": [
        ("_CONFIG_REGISTRY flint",                                     # 019
         lambda m: m._CONFIG_REGISTRY["flint"]),
    ],
    "sglang.srt.function_call.flint_detector": [
        ("FlintDetector", lambda m: m.FlintDetector),                  # 019
    ],
    "sglang.srt.function_call.function_call_parser": [
        ("ToolCallParserEnum flint",                                   # 019
         lambda m: m.FunctionCallParser.ToolCallParserEnum["flint"]),
    ],
    "sglang.srt.parser.reasoning_parser": [
        ("DetectorMap flint",                                          # 019
         lambda m: m.ReasoningParser.DetectorMap["flint"]),
    ],
}

# Modules whose import has environment side effects that cannot run here.
# Every skip carries a stated reason — never a silent drop.
SKIPS = {}


def patched_modules():
    mods = set()
    for patch in sorted(glob.glob(os.path.join(PATCH_DIR, "0[012][0-9]-*.patch"))):
        with open(patch) as f:
            for line in f:
                m = re.match(r"\+\+\+ b/python/(sglang/\S+)\.py$", line)
                if m:
                    mods.add(m.group(1).replace("/", "."))
    return sorted(mods)


def main() -> int:
    mods = patched_modules()
    if not mods:
        print(f"FATAL: no patched modules found under {PATCH_DIR}")
        return 2
    failures = 0
    print(f"== import smoke: {len(mods)} patch-touched modules ==")
    for mod in mods:
        if mod in SKIPS:
            print(f"  SKIP  {mod} — {SKIPS[mod]}")
            continue
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            print(f"  FAIL  {mod} — {type(e).__name__}: {e}")
            failures += 1
            continue
        attr_fail = []
        for name, probe in ATTR_CHECKS.get(mod, []):
            try:
                probe(m)
            except Exception as e:
                attr_fail.append(f"{name} ({type(e).__name__}: {e})")
        if attr_fail:
            print(f"  FAIL  {mod} — missing load-bearing symbols: "
                  + "; ".join(attr_fail))
            failures += 1
        else:
            n_attrs = len(ATTR_CHECKS.get(mod, []))
            extra = f" (+{n_attrs} symbol checks)" if n_attrs else ""
            print(f"  PASS  {mod}{extra}")
    print(f"== {'FAIL' if failures else 'PASS'}: "
          f"{len(mods) - failures}/{len(mods)} modules ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
