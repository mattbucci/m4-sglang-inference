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
ATTR_CHECKS = {
    "sglang.srt.managers.schedule_batch": [
        ("Modality.MULTI_IMAGES", lambda m: m.Modality.MULTI_IMAGES),  # 007
    ],
    "sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper": [
        ("MLXAttentionWrapper.__getattr__",                            # 005
         lambda m: m.MLXAttentionWrapper.__getattr__),
    ],
    "sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state": [
        ("MlxAuxiliaryStatePool.alloc_group_begin",                    # 008
         lambda m: m.MlxAuxiliaryStatePool.alloc_group_begin),
        ("MlxAuxiliaryStatePool.schedulable_available_size",
         lambda m: m.MlxAuxiliaryStatePool.schedulable_available_size),
        ("MlxAuxiliaryStateComponent.finalize_match_result",
         lambda m: m.MlxAuxiliaryStateComponent.finalize_match_result),
    ],
    "sglang.srt.hardware_backend.mlx.model_runner": [
        # (_TextOnlyVLMShim is nested inside _load_model — not module-level)
        ("_CompactCacheModel", lambda m: m._CompactCacheModel),        # 008
        ("_NullLayerCache", lambda m: m._NullLayerCache),              # 008
    ],
    "sglang.srt.utils.hf_transformers.tokenizer": [
        ("_declared_tokenizer_class",                                  # 008
         lambda m: m._declared_tokenizer_class),
    ],
    "sglang.srt.utils.hf_transformers.processor": [
        ("wrap_as_pixtral", lambda m: m.wrap_as_pixtral),              # 014
    ],
    "sglang.srt.managers.overlap_utils": [
        ("FutureMap", lambda m: m.FutureMap),                          # 008
    ],
    "sglang.srt.server_args": [
        ("ServerArgs", lambda m: m.ServerArgs),                        # 002
    ],
}

# Modules whose import has environment side effects that cannot run here.
# Every skip carries a stated reason — never a silent drop.
SKIPS = {}


def patched_modules():
    mods = set()
    for patch in sorted(glob.glob(os.path.join(PATCH_DIR, "0[01][0-9]-*.patch"))):
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
