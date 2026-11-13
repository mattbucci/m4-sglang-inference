#!/usr/bin/env python3
"""Where each model stands against what the hardware allows.

Decode is bound by memory traffic: one token streams every weight its matmuls
touch plus every cached key and value its attention layers read, so the floor
of a step at depth d is (weight bytes + KV bytes(d)) over the read rate.
Prefill is bound by arithmetic: a chunk reuses each weight across its rows, so
the floor per token is its FLOPs (2 per active weight, plus attention over the
prefix) over the matmul rate. Both rates come from
mlx_hardware_ceiling_probe.py; bytes and FLOPs come from the checkpoint's
safetensors headers and config (no model load, no GPU).

Measured numbers are the ones the fleet charts draw (generate_charts.MODELS:
baselines.json rungs and cold receipts). Not counted in the floor: recurrent
state of DeltaNet / Mamba layers, activations, sampling, the host. A measured
point above 100 % would mean the rate is understated.

  python scripts/bench/ceiling_report.py --hardware benchmarks/perf/<day>/ceiling/hardware-*.json \
      --out benchmarks/perf/<day>/ceiling/ceiling.json --chart benchmarks/all_models_ceiling.png
"""
import argparse
import collections
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_charts as fleet
from mlx_weight_read_census import BYTES, family

HUB = os.path.expanduser("~/.cache/huggingface/hub")
# generate_charts.MODELS key -> checkpoint (hub repo id or a local directory)
CHECKPOINTS = {
    "qwen36": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "coder-30b": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
    "qwen3-moe": "mlx-community/Qwen3-30B-A3B-4bit-DWQ",
    "nemotron-30b": "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
    "flint-oq8e": "mlx-community/flint-oQ8e",
    "flint-int4": "RecursiveSI/flint-int4",
    "qwen35-9b-8bit": "mlx-community/Qwen3.5-9B-MLX-8bit",
    "devstral": "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
    "qwen36-27b": "mlx-community/Qwen3.6-27B-4bit",
    "qwen38-27b": "mlx-community/Qwen3.8-27B-4bit",
    "qwen35": "mlx-community/Qwen3.5-27B-4bit",
    "qwen3-32b": "mlx-community/Qwen3-32B-4bit-DWQ",
}
# Rates used when no probe receipt is given: the launch microbench's independent 4-bit matmuls.
DEFAULT_READ_GB_S = 239.1
DEPTHS = (1024, 8192, 32768, 131072)
KV_BYTES = 2  # bf16 keys and values


def snapshot_dir(checkpoint):
    if os.path.isdir(os.path.expanduser(checkpoint)):
        return os.path.expanduser(checkpoint)
    hits = sorted(glob.glob(os.path.join(HUB, "models--" + checkpoint.replace("/", "--"), "snapshots", "*")))
    return hits[0] if hits else None


def _headers(root):
    for path in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with open(path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name != "__metadata__":
                yield name, meta


def _size(meta, unit=1):
    for dim in meta["shape"]:
        unit *= dim
    return unit


def attention_layers(text):
    """KV window of every attention layer: None is the whole prefix."""
    layer_types = text.get("layer_types")
    if layer_types:
        window = text.get("sliding_window")
        return [window if kind == "sliding_attention" else None
                for kind in layer_types if kind in ("full_attention", "sliding_attention")]
    pattern = text.get("hybrid_override_pattern")
    if pattern:
        return [None] * pattern.count("*")
    return [None] * int(text["num_hidden_layers"])


def checkpoint_costs(root):
    """Bytes read and weights applied per decoded token, by family, and the attention shape."""
    with open(os.path.join(root, "config.json")) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config", cfg)
    experts = int(text.get("num_experts", 0) or text.get("n_routed_experts", 0) or 0)
    top_k = int(text.get("num_experts_per_tok", 0) or 0)
    quant = cfg.get("quantization", {})
    default_group = int(quant.get("group_size", 64))

    tensors = dict(_headers(root))
    if not experts and top_k:
        # NemotronH records no expert count: read it off a routed weight's leading dimension
        experts = max((meta["shape"][0] for name, meta in tensors.items()
                       if family(name) == "routed experts" and len(meta["shape"]) == 3), default=0)
    nbytes, params = collections.Counter(), collections.Counter()
    for name, meta in tensors.items():
        fam = family(name)
        nbytes[fam] += _size(meta, BYTES[meta["dtype"]])
        if name.endswith(".scales"):
            module = name[: -len(".scales")]
            group = int(quant.get(module, {}).get("group_size", default_group)) if isinstance(
                quant.get(module), dict) else default_group
            params[fam] += _size(meta) * group
        elif name.endswith(".weight") and name[: -len("weight")] + "scales" not in tensors and len(meta["shape"]) >= 2:
            params[fam] += _size(meta)

    def active(counter, fam):
        if fam in ("vision tower", "embedding"):
            return 0
        if fam == "routed experts" and experts:
            return counter[fam] * top_k // experts
        return counter[fam]

    read = {fam: active(nbytes, fam) for fam in nbytes}
    applied = {fam: active(params, fam) for fam in params}
    heads, kv_heads = int(text["num_attention_heads"]), int(text.get("num_key_value_heads", text["num_attention_heads"]))
    head_dim = int(text.get("head_dim") or text["hidden_size"] // heads)
    return {
        "read_bytes": sum(read.values()),
        "read_bytes_by_family": {fam: size for fam, size in sorted(read.items(), key=lambda kv: -kv[1]) if size},
        "prefill_weights": sum(size for fam, size in applied.items() if fam != "lm_head"),
        "attention_windows": attention_layers(text),
        "kv_bytes_per_position": 2 * kv_heads * head_dim * KV_BYTES,
        "attention_flops_per_position": 4.0 * heads * head_dim,
    }


def kv_bytes(costs, depth):
    return sum(costs["kv_bytes_per_position"] * (min(depth, w) if w else depth) for w in costs["attention_windows"])


def decode_ceiling(costs, depth, read_gb_s):
    return read_gb_s * 1e9 / (costs["read_bytes"] + kv_bytes(costs, depth))


def prefill_ceiling(costs, depth, gflops):
    """Whole-prompt rate: weight FLOPs per token plus attention over the mean prefix (depth / 2)."""
    mean = depth / 2
    attention = sum(costs["attention_flops_per_position"] * (min(mean, w) if w else mean)
                    for w in costs["attention_windows"])
    return gflops * 1e9 / (2.0 * costs["prefill_weights"] + attention)


def hardware_rates(paths):
    """The bus (best GPU streaming read) and the best matmul rate set the ceilings.

    The isolated quantized-matmul rows are recorded beside them but are no
    limit: served dense models beat both, on reads and on prefill arithmetic.
    """
    read, compute, bus, many_row = [], [], [], []
    for path in paths:
        with open(path) as fh:
            probe = json.load(fh)
        read += [r["gb_per_s"] for r in probe["bandwidth"] if r["case"].startswith("qmm one row")]
        bus += [r["gb_per_s"] for r in probe["bandwidth"] if r["case"].startswith("gpu")]
        compute += [r["gflops"] for r in probe["compute"]]
        many_row += [r["gflops"] for r in probe["compute"] if r["case"].startswith("qmm")]
    return {"read_gb_s": max(bus) if bus else DEFAULT_READ_GB_S,
            "one_row_qmm_gb_s": max(read) if read else None,
            "prefill_gflops": max(compute) if compute else None,
            "many_row_qmm_gflops": max(many_row) if many_row else None,
            "receipts": [os.path.relpath(p, fleet.REPO) for p in paths]}


def measured_points(meta, baselines, bench_dir=fleet.BENCH_DIR):
    """{depth: {"decode": tok/s, "prefill": tok/s, "input": tokens}} from the baseline rungs and cold receipts."""
    points = {}
    for label, row in (baselines.get(meta["baseline"] or "", {}) or {}).items():
        if not label.isdigit():
            continue
        tokens = row["actual_input_tokens"]
        points[int(label)] = {"decode": row["tok_per_sec"], "input": tokens,
                              "prefill": round(tokens / (row["ttft_ms"] / 1000.0), 1)}
    cold = {}
    for rel in meta["cold"]:
        receipt = fleet._read_json(os.path.join(bench_dir, rel)) or {}
        for row in receipt.get("cold", []):
            cold.setdefault(row["target"], []).append(
                (row["cold"]["prompt_tokens"], row["cold"]["decode_tok_s"], row.get("prefill_tok_s")))
    for hits in cold.values():
        tokens = round(sum(h[0] for h in hits) / len(hits))
        depth = min(DEPTHS, key=lambda d: abs(d - tokens))
        if abs(depth - tokens) > 0.2 * depth or depth in points:
            continue
        rates = [h[2] for h in hits if h[2]]
        points[depth] = {"decode": round(sum(h[1] for h in hits) / len(hits), 1), "input": tokens,
                         "prefill": round(sum(rates) / len(rates), 1) if rates else None}
    return points


def build(hardware):
    baselines = fleet._read_json(os.path.join(fleet.BENCH_DIR, "baselines.json")) or {}
    rows = []
    for key, meta in fleet.MODELS.items():
        root = snapshot_dir(CHECKPOINTS.get(key, ""))
        if not root:
            continue
        costs = checkpoint_costs(root)
        row = {"key": key, "label": meta["label"], "cls": meta["cls"], "checkpoint": CHECKPOINTS[key],
               "read_mb_per_token": round(costs["read_bytes"] / 1e6, 1),
               "read_mb_by_family": {f: round(b / 1e6, 1) for f, b in costs["read_bytes_by_family"].items()},
               "kv_kb_per_context_token": round(kv_bytes(costs, 1) / 1e3, 2),
               "attention_layers": len(costs["attention_windows"]),
               "prefill_gflop_per_token": round(2.0 * costs["prefill_weights"] / 1e9, 2), "depths": {}}
        for depth, point in sorted(measured_points(meta, baselines).items()):
            tokens = point["input"]
            ceiling = decode_ceiling(costs, tokens, hardware["read_gb_s"])
            entry = {"input_tokens": tokens, "decode_tok_s": point["decode"], "decode_ceiling_tok_s": round(ceiling, 1),
                     "decode_pct": round(100 * point["decode"] / ceiling, 1),
                     "kv_read_mb": round(kv_bytes(costs, tokens) / 1e6, 1)}
            if hardware["prefill_gflops"] and point["prefill"]:
                top = prefill_ceiling(costs, tokens, hardware["prefill_gflops"])
                entry.update(prefill_tok_s=point["prefill"], prefill_ceiling_tok_s=round(top, 1),
                             prefill_pct=round(100 * point["prefill"] / top, 1))
            row["depths"][str(depth)] = entry
        rows.append(row)
    return {"hardware": hardware, "kv_dtype_bytes": KV_BYTES, "models": rows}


def print_table(report):
    hw = report["hardware"]
    print(f"read rate {hw['read_gb_s']} GB/s, prefill matmul rate {hw['prefill_gflops']} GFLOPS\n")
    print("| Model | Weights read / token | KV read / context token | Depth | Decode tok/s | Ceiling | % | "
          "Prefill tok/s | Ceiling | % |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["models"]:
        for depth, e in row["depths"].items():
            prefill = (f"{e['prefill_tok_s']:.0f} | {e['prefill_ceiling_tok_s']:.0f} | {e['prefill_pct']:.0f}"
                       if "prefill_pct" in e else "| |")
            print(f"| {row['label']} | {row['read_mb_per_token']:.0f} MB | {row['kv_kb_per_context_token']:.1f} KB | "
                  f"{fleet._fmt_tok(e['input_tokens'])} | {e['decode_tok_s']:.1f} | {e['decode_ceiling_tok_s']:.1f} | "
                  f"{e['decode_pct']:.0f} | {prefill} |")


def make_chart(report, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    depths = [str(d) for d in DEPTHS if any(str(d) in r["depths"] for r in report["models"])]
    alphas = dict(zip(depths, (1.0, 0.72, 0.5, 0.32)))
    has_prefill = any("prefill_pct" in e for r in report["models"] for e in r["depths"].values())
    panels = [("decode", "Decode: tok/s as a share of the memory-traffic ceiling")]
    if has_prefill:
        panels.append(("prefill", "Prefill: tok/s as a share of the matmul ceiling"))
    sections = fleet.fleet_groups({r["key"]: (fleet.MODELS[r["key"]], r) for r in report["models"] if r["depths"]})
    n_rows = sum(len(group) for _, group in sections)
    fig, axes = plt.subplots(1, len(panels), figsize=(8.2 * len(panels), 0.62 * n_rows + 0.5 * len(sections) + 2.4),
                             sharey=True, squeeze=False)
    height = 0.8 / len(depths)
    for ax, (kind, title) in zip(axes[0], panels):
        y, ticks, labels = 0.0, [], []
        for section, group in sections:
            ax.axhline(y, color=fleet.EDGE, linewidth=1.0, zorder=1)
            if ax is axes[0][0]:
                ax.text(-0.012, y, section, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=11.5, fontweight="bold", color="#f0f6fc")
            y += 0.8
            for _, meta, row in group:
                color = meta["color"]
                for slot, depth in enumerate(depths):
                    e = row["depths"].get(depth)
                    if not e or f"{kind}_pct" not in e:
                        continue
                    yy = y + (slot - (len(depths) - 1) / 2) * height
                    pct = e[f"{kind}_pct"]
                    ax.barh(yy, pct, height * 0.86, color=color, alpha=alphas[depth], zorder=5)
                    got, top = e[f"{kind}_tok_s"], e[f"{kind}_ceiling_tok_s"]
                    digits = 0 if top >= 40 else 1
                    ax.text(max(pct, 100) + 1.5 if pct > 84 else pct + 1, yy, f"{got:.{digits}f} of {top:.{digits}f}",
                            va="center", fontsize=7.5, color=fleet.INK)
                ticks.append(y)
                labels.append(row["label"])
                y += 1.0
        ax.axvline(100, color=fleet.INK, linewidth=1.2, linestyle="--", zorder=4)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9.5, color=fleet.INK)
        ax.set_ylim(y - 0.4, -0.7)
        ax.set_xlim(0, 122)
        ax.set_xticks(range(0, 101, 20))
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("% of ceiling (bar label: measured of ceiling, tok/s)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
    fig.legend(
        handles=[Patch(facecolor=fleet.MUTED, alpha=alphas[d], label=f"{fleet._fmt_tok(int(d))} input") for d in depths],
        loc="lower right", ncol=len(depths), framealpha=0.6, edgecolor=fleet.EDGE, facecolor=fleet.PANEL, fontsize=9,
        bbox_to_anchor=(0.99, 0.055))
    hw = report["hardware"]
    note = (f"Ceiling: decode = read rate over (weights read per token + cached keys and values at that depth), "
            f"at {hw['read_gb_s']:.0f} GB/s; prefill = matmul rate over (2 FLOPs per active weight + attention over "
            f"the prefix)" + (f", at {hw['prefill_gflops']:.0f} GFLOPS" if hw["prefill_gflops"] else "")
            + ". Rates: best GPU streaming read and best matmul of mlx_hardware_ceiling_probe.py on this M4 Pro.")
    fig.text(0.01, 0.005, "\n".join(__import__("textwrap").wrap(note, 105 * len(panels))), fontsize=8,
             color=fleet.MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hardware", nargs="*", default=[], help="mlx_hardware_ceiling_probe.py receipts")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--chart", help="write the chart here")
    args = ap.parse_args()
    report = build(hardware_rates(args.hardware))
    print_table(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
            fh.write("\n")
    if args.chart:
        make_chart(report, args.chart)


if __name__ == "__main__":
    main()
