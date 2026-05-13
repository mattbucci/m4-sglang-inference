#!/usr/bin/env python3
"""Metadata-only calibration audit for mlx-community / local MLX checkpoints.

Adapted from the 3090 sister-team's ``scripts/eval/audit_calib_quality.py``
(commit 6f7f2ae). Theirs targets ``mattbucci/*-AWQ`` repos with the
``.qweight / .scales / .qzeros`` AWQ triple. Ours targets MLX-format
repos, where 4-bit/8-bit linears store ``weight / scales / biases``.

What it checks (without downloading weights or loading the model):

  1. Architecture class declared (text-only vs multimodal)
  2. Vision tower present + whether quantized
  3. Audio tower present + whether quantized
  4. MoE router (mlp.gate) quantized? must NOT be — top-k routing
     accuracy collapses under INT4
  5. DeltaNet ``linear_attn.in_proj_a / in_proj_b`` quantized?
     must NOT be — recurrent state error accumulates

For multi-shard repos we read ``model.safetensors.index.json``
(~100 KB). For single-file ones we Range-fetch the safetensors
header. Either way we never pull the weight payload, so this is
safe to run alongside an active model load.

This is the M4 analog to ``check_awq_scales.py`` (NaN/Inf/all-zero
scale scan, which we already ported as ``check_mlx_quant_scales.py``).
The two complement each other:
  - ``check_mlx_quant_scales.py`` finds dead-zero scale tensors in
    weights you've already loaded (catches per-layer corruption).
  - ``audit_mlx_quant_metadata.py`` flags quantization scheme
    mismatches from tensor names alone (catches recipe-level
    mistakes like INT4'd vision towers).

Usage:
    python scripts/eval/audit_mlx_quant_metadata.py
    python scripts/eval/audit_mlx_quant_metadata.py --repo mlx-community/Qwen3.5-27B-4bit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request

# The mlx-community checkpoints currently wired into M4 launch presets
# (scripts/launch.sh). Update when presets change.
DEFAULT_REPOS = [
    "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
    "mlx-community/Qwen3-Coder-Next-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
    "mlx-community/gemma-4-31b-it-mxfp4",
    "mlx-community/Qwen3.5-27B-4bit",
    "mlx-community/Qwen3.5-9B-MLX-8bit",
    "mlx-community/Qwen3-32B-4bit-DWQ",
    "mlx-community/Qwen3-30B-A3B-4bit-DWQ",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "mlx-community/Qwen3.6-27B-4bit",
    "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
]


def _hf_token() -> str | None:
    p = os.path.expanduser("~/.secrets/hf_token")
    if os.path.exists(p):
        return open(p).read().strip()
    return os.environ.get("HF_TOKEN")


def _get(url: str, *, range_header: str | None = None) -> bytes:
    headers = {}
    tok = _hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _tensor_keys(repo: str) -> list[str]:
    """Return the list of tensor keys in the repo without downloading weights."""
    try:
        idx = json.loads(
            _get(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json")
        )
        return list(idx.get("weight_map", {}).keys())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    # Single-file safetensors: range-fetch the 8-byte header length then the header JSON.
    n = struct.unpack(
        "<Q",
        _get(
            f"https://huggingface.co/{repo}/resolve/main/model.safetensors",
            range_header="bytes=0-7",
        ),
    )[0]
    if n > 50_000_000:
        raise RuntimeError(
            f"{repo}: safetensors header is {n / 1e6:.1f} MB — refusing to fetch (corrupt?)"
        )
    header = json.loads(
        _get(
            f"https://huggingface.co/{repo}/resolve/main/model.safetensors",
            range_header=f"bytes=8-{8 + n - 1}",
        )
    )
    header.pop("__metadata__", None)
    return list(header.keys())


# MLX-quantized linears come in two flavors:
#   - 4-bit / 8-bit (mlx_lm.utils): weight + scales + biases
#   - mxfp4 (microscaling FP4):     weight + scales  (no biases)
# Either way the presence of a sibling `.scales` is the load-bearing signal.
QUANT_SUFFIXES = (".weight", ".scales", ".biases")


def _quantized_layers(keys: list[str]) -> set[str]:
    """Return the set of layer prefixes whose linear has a sibling .scales tensor."""
    triples: dict[str, set[str]] = {}
    for k in keys:
        for sfx in QUANT_SUFFIXES:
            if k.endswith(sfx):
                prefix = k[: -len(sfx)]
                triples.setdefault(prefix, set()).add(sfx)
                break
    return {prefix for prefix, parts in triples.items() if ".scales" in parts}


def _layer_prefix(key: str) -> str | None:
    """Strip a tensor key down to its layer prefix (without trailing suffix)."""
    for sfx in QUANT_SUFFIXES:
        if key.endswith(sfx):
            return key[: -len(sfx)]
    return None


def audit(repo: str) -> dict:
    cfg = json.loads(_get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
    arch = ", ".join(cfg.get("architectures", []) or ["?"])
    qc = cfg.get("quantization", {}) or cfg.get("quantization_config", {})
    qbits = qc.get("bits")
    qgroup = qc.get("group_size")
    keys = _tensor_keys(repo)
    quantized = _quantized_layers(keys)

    def group_keys(keys, patterns):
        """Tensor keys whose name contains any of the patterns."""
        return [k for k in keys if any(t in k for t in patterns)]

    vision_keys = group_keys(
        keys,
        (
            "vision_tower",
            "visual.",
            "multi_modal_projector",
            "embed_vision",
            "vision_model",
        ),
    )
    audio_keys = group_keys(keys, ("audio_tower", "embed_audio", "audio_model"))
    router_keys = [
        k for k in keys
        if re.search(r"mlp\.gate(\.|$)", k) and "shared_expert" not in k
    ]
    deltanet_keys = [
        k for k in keys
        if re.search(r"linear_attn\.in_proj_(a|b)(\.|$)", k)
    ]
    lm_head_keys = [k for k in keys if k.startswith("lm_head.") or k.endswith("lm_head.weight")]
    embed_keys = [k for k in keys if k.startswith("model.embed_tokens.") or k.endswith("embed_tokens.weight")]

    def quantized_in(group):
        layers = {_layer_prefix(k) for k in group if _layer_prefix(k)}
        return sorted(p for p in layers if p in quantized)

    def bf16_in(group):
        layers = {_layer_prefix(k) for k in group if _layer_prefix(k)}
        return sorted(p for p in layers if p not in quantized)

    findings = []
    multimodal_arch = "ConditionalGeneration" in arch or "ForConditional" in arch

    if multimodal_arch and not vision_keys:
        findings.append("multimodal arch but NO vision_tower tensors — recipe stripped or base mismatch")
    if vision_keys:
        q = quantized_in(vision_keys)
        if q:
            findings.append(f"vision_tower has {len(q)} INT4/8 layers — silent degradation risk")
    if audio_keys:
        q = quantized_in(audio_keys)
        if q:
            findings.append(f"audio_tower has {len(q)} INT4/8 layers — silent degradation risk")
    if router_keys:
        q = quantized_in(router_keys)
        if q:
            findings.append(f"MoE router (mlp.gate) has {len(q)} quantized layers — top-k accuracy degraded")
    if deltanet_keys:
        q = quantized_in(deltanet_keys)
        if q:
            findings.append(
                f"DeltaNet in_proj_a/b has {len(q)} quantized layers — recurrent state will diverge"
            )

    return {
        "repo": repo,
        "arch": arch,
        "quant_bits": qbits,
        "quant_group_size": qgroup,
        "total_keys": len(keys),
        "total_quantized_layers": len(quantized),
        "vision": (len(vision_keys), len(bf16_in(vision_keys)), len(quantized_in(vision_keys))),
        "audio": (len(audio_keys), len(bf16_in(audio_keys)), len(quantized_in(audio_keys))),
        "router": (len(router_keys), len(bf16_in(router_keys)), len(quantized_in(router_keys))),
        "deltanet": (
            len(deltanet_keys),
            len(bf16_in(deltanet_keys)),
            len(quantized_in(deltanet_keys)),
        ),
        "lm_head_quantized": any(_layer_prefix(k) in quantized for k in lm_head_keys),
        "embed_quantized": any(_layer_prefix(k) in quantized for k in embed_keys),
        "findings": findings,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo",
        action="append",
        help="repo id (default: mlx-community checkpoints currently wired in launch.sh)",
    )
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args()
    repos = args.repo or DEFAULT_REPOS

    results = []
    issues_total: list[tuple[str, str]] = []

    if not args.json:
        # Column widths chosen for terminal display; multimodal arch class
        # strings can run long so we left-justify and let it overflow.
        print(
            f"{'repo':<60} {'bits':>4} {'vis(bf/q)':<11} {'aud(bf/q)':<11} "
            f"{'router(bf/q)':<13} {'deltanet(bf/q)'}"
        )
        print("-" * 160)

    for repo in repos:
        try:
            r = audit(repo)
        except Exception as e:
            if not args.json:
                print(f"{repo:<60} ERROR: {e}")
            results.append({"repo": repo, "error": str(e)})
            continue
        results.append(r)
        if args.json:
            continue
        v, a, ro, dn = r["vision"], r["audio"], r["router"], r["deltanet"]
        bits = str(r["quant_bits"]) if r["quant_bits"] is not None else "?"
        print(
            f"{r['repo']:<60} {bits:>4} "
            f"{v[0]:>3}({v[1]:>3}/{v[2]:<3}) {a[0]:>3}({a[1]:>3}/{a[2]:<3}) "
            f"{ro[0]:>3}({ro[1]:>3}/{ro[2]:<5}) {dn[0]:>3}({dn[1]:>3}/{dn[2]})"
        )
        for f in r["findings"]:
            print(f"  ⚠ {f}")
            issues_total.append((r["repo"], f))

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"\n{'=' * 60}\nSUMMARY: {len(issues_total)} issues across {len(repos)} repos"
        )
        for repo, f in issues_total:
            print(f"  {repo}: {f}")

    return 1 if issues_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
