#!/usr/bin/env python3
"""Scan an MLX-quantized model directory for silent quantization disasters.

Sister-team finding (2026-05-06 / 3090 + R9700, see 3090 CLAUDE.md): a
calibration recipe with the wrong ``ignore`` regex left ``.scales`` /
``.biases`` tensors all-zero on the vision projector descendants of
Gemma 4 26B; ``validate_capabilities.py`` couldn't see it (the model
loaded, the server booted, generation produced empty content from
NaN logits). The 30-second forensic-safetensors diff caught what
16 hours of running could not.

MLX 4-bit/8-bit quantized weights store one ``scales`` and one
``biases`` tensor per linear layer (group_size×bits packed payload in
``weight``). If a tensor is all-zero, NaN, Inf, or has obviously
extreme magnitude (saturation / overflow during the quantize step),
dequantization in inference path produces garbage downstream.

Run this against every fresh mlx-community checkpoint before adding
it to the model gate.

Exit code is non-zero if any tensor is broken — wire into CI / smoke
test ordering before claiming ship.

Usage:
    python scripts/eval/check_mlx_quant_scales.py mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
    python scripts/eval/check_mlx_quant_scales.py /local/path/to/weights
    python scripts/eval/check_mlx_quant_scales.py mlx-community/...  --max-mag 1e6
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Iterable

import mlx.core as mx


def resolve_model_dir(name_or_path: str) -> pathlib.Path:
    """Resolve a HuggingFace repo id or local path to a directory with safetensors."""
    p = pathlib.Path(name_or_path)
    if p.is_dir():
        return p

    hub_root = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    candidate = hub_root / f"models--{name_or_path.replace('/', '--')}"
    if not candidate.exists():
        raise FileNotFoundError(
            f"{name_or_path} not on disk: looked for local path and {candidate}"
        )
    snapshots = candidate / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots/ under {candidate}")
    revs = list(snapshots.iterdir())
    if not revs:
        raise FileNotFoundError(f"No revisions under {snapshots}")
    return revs[0]


def safetensor_files(model_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(model_dir.glob("*.safetensors"))


def layer_groups(weights: dict) -> dict[str, dict[str, "mx.array"]]:
    """Group weight/scales/biases tensors by layer prefix.

    Returns ``{layer_prefix: {"weight": t, "scales": t, "biases": t}}``.
    Skips entries that aren't part of a quantized triple (e.g. norms, embeds).
    """
    groups: dict[str, dict[str, "mx.array"]] = {}
    for k, t in weights.items():
        for suffix in (".weight", ".scales", ".biases"):
            if k.endswith(suffix):
                prefix = k[: -len(suffix)]
                groups.setdefault(prefix, {})[suffix.lstrip(".")] = t
                break
    # Only keep layers that have both .scales and .weight (the quantized triple).
    return {p: g for p, g in groups.items() if "scales" in g and "weight" in g}


def check_layer(prefix: str, parts: dict[str, "mx.array"], max_mag: float) -> list[str]:
    """Return list of failure reasons for this quantized layer (empty = healthy).

    A layer is BROKEN when its dequantized output is identically zero or
    non-finite. With MLX's affine 4-bit/8-bit:
      dequant(w, s, b) = unpacked(w) * s + b
    So both branches lead to dead output:
      - weight payload all-zero AND biases all-zero → 0*s+0 = 0
      - scales all-zero → unpacked(w)*0 + b = b (constant per group)
      - any NaN/Inf in scales OR biases → cascades through forward pass
    Standalone all-zero ``.biases`` with non-zero weight payload is a legit
    symmetric-quantization outcome and is NOT flagged here.
    """
    issues: list[str] = []
    w = parts.get("weight")
    s = parts.get("scales")
    b = parts.get("biases")

    # NaN/Inf in scales is always fatal — they multiply into every dequant.
    if s is not None:
        if not bool(mx.isfinite(s).all().item()):
            n_nan = int(mx.isnan(s).sum().item())
            n_inf = int(mx.isinf(s).sum().item())
            issues.append(f"scales non-finite ({n_nan} NaN, {n_inf} Inf)")
        elif bool((s == 0).all().item()):
            issues.append("scales ALL ZERO (dequant collapses to bias-only)")
        else:
            max_s = float(mx.abs(s).max().item())
            if math.isfinite(max_s) and max_s > max_mag:
                issues.append(f"scales saturated (|max|={max_s:.3g} > {max_mag:g})")

    # NaN/Inf in biases is fatal too.
    if b is not None:
        if not bool(mx.isfinite(b).all().item()):
            n_nan = int(mx.isnan(b).sum().item())
            n_inf = int(mx.isinf(b).sum().item())
            issues.append(f"biases non-finite ({n_nan} NaN, {n_inf} Inf)")

    # Combined check: weight payload all-zero AND biases all-zero means the
    # dequantized layer outputs identically zero on every input. (Either alone
    # is a legitimate calibration outcome.)
    if w is not None:
        w_zero = bool((w == 0).all().item())
        if w_zero:
            b_zero = b is None or bool((b == 0).all().item())
            if b_zero:
                issues.append(
                    f"DEAD LAYER (weight payload all-zero AND biases all-zero) — "
                    f"this layer's dequant output is identically 0"
                )
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument(
        "--max-mag",
        type=float,
        default=1e3,
        help="Threshold above which a |max| value counts as saturation. "
        "Typical scales sit well below 10; 1000 is a generous ceiling.",
    )
    ap.add_argument(
        "--show-ok",
        action="store_true",
        help="Print healthy tensors too (default: only show issues + summary).",
    )
    args = ap.parse_args()

    try:
        model_dir = resolve_model_dir(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    files = safetensor_files(model_dir)
    if not files:
        print(f"ERROR: no *.safetensors in {model_dir}", file=sys.stderr)
        return 2

    print(f"Scanning: {model_dir}")
    print(f"  {len(files)} safetensor file(s)")
    print(f"  max-mag threshold: {args.max_mag}")
    print()

    total_layers = 0
    bad: list[tuple[str, str, list[str]]] = []  # (file, layer_prefix, issues)

    for f in files:
        weights = mx.load(str(f))
        groups = layer_groups(weights)
        total_layers += len(groups)
        for prefix, parts in groups.items():
            issues = check_layer(prefix, parts, args.max_mag)
            if issues:
                bad.append((f.name, prefix, issues))
                print(f"  [BAD] {f.name}::{prefix}")
                for issue in issues:
                    print(f"        - {issue}")
            elif args.show_ok:
                s = parts.get("scales")
                if s is not None:
                    max_s = float(mx.abs(s).max().item())
                    print(f"  [OK]  {f.name}::{prefix} (|scale max|={max_s:.4g})")

    print()
    print("=" * 70)
    if bad:
        print(f"FAILED: {len(bad)}/{total_layers} quantized layers broken")
        return 1
    print(f"OK: all {total_layers} quantized layers (weight+scales+biases) healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
