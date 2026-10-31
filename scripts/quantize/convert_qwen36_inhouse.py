#!/usr/bin/env python3
"""In-house MLX 4-bit of Qwen3.6-35B-A3B with the hazard classes kept BF16.

Mirrors the 3090 lane's narrow-ignore recipe
(`quantize_qwen36_thinking_vision.py`, v3): the MoE routers (`mlp.gate`),
the scalar shared-expert gates (`shared_expert_gate`), the DeltaNet
alpha/beta gates (`linear_attn.in_proj_a` / `in_proj_b`) and the vision
tower stay BF16; everything else is 4-bit affine, group 64, the incumbent's
default. The incumbent `mlx-community/Qwen3.6-35B-A3B-4bit` quantizes the
routers and shared-expert gates at 8-bit and the alpha/beta gates at 4-bit;
its vision tower is already BF16.

    python scripts/quantize/convert_qwen36_inhouse.py --dry-run     # decisions only
    python scripts/quantize/convert_qwen36_inhouse.py               # convert + recipe check
"""
import argparse
import collections
import json
import os
import re
import sys
import time

KEEP_BF16 = (
    ".mlp.gate",                 # MoE router: top-k routing collapses under INT4
    ".mlp.shared_expert_gate",   # scalar shared-expert gate [1, H]
    ".linear_attn.in_proj_a",    # DeltaNet alpha gate
    ".linear_attn.in_proj_b",    # DeltaNet beta gate
)


def keep_bf16(path: str) -> bool:
    return path.endswith(KEEP_BF16)


def build_predicate():
    from mlx_vlm.utils import skip_multimodal_module

    def predicate(path, module):
        if skip_multimodal_module(path):   # vision tower + merger, mlx-vlm's default
            return False
        if keep_bf16(path):
            return False
        return True

    return predicate


def class_of(path: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", path)


def decisions(model, predicate):
    """Per module class: how many linear modules quantize and how many stay BF16."""
    table = collections.defaultdict(lambda: [0, 0])
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            continue
        q = bool(predicate(path, module)) and module.weight.shape[-1] % 64 == 0
        table[class_of(path)][0 if q else 1] += 1
    return table


def verify_output(out_dir: str) -> dict:
    """Recipe gate on the written index: no scales under the kept classes."""
    idx = json.load(open(os.path.join(out_dir, "model.safetensors.index.json")))["weight_map"]
    modules = {k.rsplit(".", 1)[0] for k in idx if k.endswith((".weight", ".scales"))}
    scaled = {k.rsplit(".", 1)[0] for k in idx if k.endswith(".scales")}
    report = collections.Counter()
    bad = []
    for m in modules:
        kept = keep_bf16(m) or "vision_tower" in m
        is_q = m in scaled
        report[(class_of(m), "quantized" if is_q else "bf16")] += 1
        if kept and is_q:
            bad.append(m)
    return {"classes": {f"{c} [{s}]": n for (c, s), n in sorted(report.items())}, "violations": bad,
            "quantized_modules": len(scaled), "modules": len(modules)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out", default=os.path.expanduser("~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse"))
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="load lazily, print the decisions, convert nothing")
    ap.add_argument("--verify-only", action="store_true", help="recipe gate on an existing --out")
    a = ap.parse_args()

    if a.verify_only:
        rep = verify_output(a.out)
        print(json.dumps(rep, indent=1))
        return 1 if rep["violations"] else 0

    import importlib
    import shutil

    from mlx_vlm.convert import convert
    from mlx_vlm.utils import fetch_from_hub, get_model_path

    def copy_into(src, dst):
        # HF cache blobs are mode 0444 and shutil.copy keeps the mode, so the
        # processor save that follows the copy fails on tokenizer.json.
        target = os.path.join(dst, os.path.basename(src)) if os.path.isdir(dst) else dst
        shutil.copyfile(src, target)
        return target

    importlib.import_module("mlx_vlm.convert").shutil.copy = copy_into

    predicate = build_predicate()
    t0 = time.time()
    model_path = get_model_path(a.hf_path)
    model, config, _ = fetch_from_hub(model_path, lazy=True, trust_remote_code=True)
    table = decisions(model, predicate)
    print(f"[recipe] {a.hf_path} -> {a.out}: {a.q_bits}-bit g{a.q_group_size} affine, kept BF16: {KEEP_BF16} + vision")
    for cls, (nq, nk) in sorted(table.items()):
        if nk or "gate" in cls or "in_proj" in cls or "vision" in cls:
            print(f"  {cls:70s} quantize={nq:3d} bf16={nk:3d}")
    print(f"  linear modules: {sum(v[0] for v in table.values())} quantize, {sum(v[1] for v in table.values())} bf16 ({time.time() - t0:.0f}s)")
    if a.dry_run:
        return 0
    del model
    convert(a.hf_path, mlx_path=a.out, quantize=True, q_group_size=a.q_group_size, q_bits=a.q_bits,
            q_mode="affine", quant_predicate=predicate)
    rep = verify_output(a.out)
    print(json.dumps(rep, indent=1))
    print(f"[done] {time.time() - t0:.0f}s, violations={len(rep['violations'])}")
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
