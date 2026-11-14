#!/usr/bin/env python3
"""Benchmark charts for the M4 Pro rig: decode speed against context and streams.

Fleet charts (benchmarks/all_models_*.png) draw one panel per model class. A
model's context curve is the rows the README decode table publishes:
benchmarks/baselines.json (the regression instrument at 1K / 8K / 32K), or the
same rungs of its results.json when it has no baseline, then the cold-request
receipts it declares for deeper input. A stream curve is the stream receipts
a model declares (scripts/bench/concurrent_decode_probe.py starts N requests
together); the throughput_sweep rows of a results.json are not drawn, because
a row labelled c sent 4c requests. Per-model context charts draw the model's
own results.json sweep. A results.json flagged legacy_depth_suspect is never
drawn.

  python scripts/bench/generate_charts.py                    # MoE / dense panels
  python scripts/bench/generate_charts.py --group-by quant   # 8-bit / 4-bit panels
  python scripts/bench/generate_charts.py --group-by none    # one panel
"""
import argparse
import json
import math
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO, "benchmarks")

FIGURE, PANEL, EDGE, RULE = "#0d1117", "#161b22", "#30363d", "#484f58"
INK, MUTED = "#c9d1d9", "#8b949e"

plt.rcParams.update({
    "figure.facecolor": FIGURE,
    "axes.facecolor": PANEL,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#21262d",
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "font.family": "sans-serif",
    "font.size": 11,
    "hatch.linewidth": 0.8,
})

# The palette of the R9700 fleet charts. Ten colours do not all separate from
# each other, so a class takes a subset in which every pair does (the contract
# test measures it): blue, green, yellow, pink, white, tan.
FLEET_PALETTE = {
    "blue": "#4ea1ff", "orange": "#ff9a3c", "green": "#3ddc84", "red": "#ff5c5c",
    "violet": "#b48cff", "yellow": "#ffe45e", "cyan": "#3fd6e0", "pink": "#ff79c6",
    "white": "#f0f6fc", "tan": "#b89b7a",
}

MODEL_CLASSES = {"moe": "MoE", "dense": "Dense"}
MODEL_QUANTS = {"8bit": "8-bit", "4bit": "4-bit"}
# --group-by value -> (model field, {field value: panel title}); None is one panel.
FLEET_GROUPINGS = {"class": ("cls", MODEL_CLASSES), "quant": ("quant", MODEL_QUANTS), "none": None}
FLEET_DEFAULT_GROUPING = "class"

_FLINT = "perf/2026-11-10/flint"
_STREAMS = "perf/2026-11-10/concurrent"
_STAGEB = "perf/2026-11-12/stage-b"
_RERUN = "perf/2026-11-13/rerun"


def _model(label, cls, quant, color, marker, baseline=None, cold=(), streams=(), results=None):
    """One model: its look in every chart, and where its numbers are read from.

    baseline is its preset in baselines.json; cold and streams are receipts
    under benchmarks/ of cold requests and of concurrent requests (files of one
    model are averaged per depth or per stream count); results is the directory
    under benchmarks/ with its results.json and per-model charts.
    """
    return {"label": label, "cls": cls, "quant": quant, "color": FLEET_PALETTE[color], "marker": marker,
            "baseline": baseline, "cold": tuple(cold), "streams": tuple(streams), "results": results}


# Colour and marker are unique inside a class, and no two models share a marker.
MODELS = {
    "qwen36": _model(
        "Qwen3.6-35B-A3B 4-bit (MoE + DeltaNet)", "moe", "4bit", "blue", "o", baseline="qwen36",
        cold=[f"{_FLINT}/quiet-rearm/q36-deep-preset-a.json", f"{_FLINT}/quiet-rearm/q36-deep-preset-b.json"],
        streams=[f"{_RERUN}/qwen36.streams-p1.json", f"{_RERUN}/qwen36.streams-p2.json"],
        results="qwen36-35b-a3b-4bit"),
    "coder-30b": _model("Coder-30B-A3B 4-bit DWQ (MoE)", "moe", "4bit", "green", "^",
                        baseline="coder-30b", results="coder-30b-4bit", cold=[f"{_RERUN}/coder.json"],
                        streams=[f"{_RERUN}/coder.streams-p1.json", f"{_RERUN}/coder.streams-p2.json"]),
    "qwen3-moe": _model("Qwen3-30B-A3B 4-bit DWQ (MoE)", "moe", "4bit", "pink", "s",
                        baseline="qwen3-moe", results="qwen3-30b-moe-4bit", cold=[f"{_RERUN}/moe.json"],
                        streams=[f"{_RERUN}/moe.streams-p1.json", f"{_RERUN}/moe.streams-p2.json"]),
    "nemotron-30b": _model("Nemotron-3-Nano-30B 4-bit (MoE + Mamba2)", "moe", "4bit", "white", "p",
                           cold=[f"{_RERUN}/nemotron.json"],
                           streams=[f"{_RERUN}/nemotron.streams-p1.json", f"{_RERUN}/nemotron.streams-p2.json"]),
    "flint-oq8e": _model(
        "Flint oQ8e (MoE, SWA + sinks)", "moe", "8bit", "tan", ">",
        cold=[f"{_FLINT}/quiet-settings/flint-oq8e-a.json", f"{_FLINT}/quiet-settings/flint-oq8e-b.json"],
        streams=[f"{_RERUN}/flint.streams-p1.json", f"{_RERUN}/flint.streams-p2.json"]),
    "flint-int4": _model("Flint int4 (MoE, SWA + sinks)", "moe", "4bit", "yellow", "D",
                         cold=[f"{_FLINT}/quiet/ladder.json"], streams=[f"{_FLINT}/submit/sub5-b.streams.json"]),
    "qwen35-9b-8bit": _model("Qwen3.5-9B 8-bit (DeltaNet)", "dense", "8bit", "pink", "h",
                             baseline="qwen35-9b-8bit", results="qwen35-9b-8bit"),
    "devstral": _model("Devstral-24B 4-bit (Dense)", "dense", "4bit", "yellow", "X",
                       baseline="devstral", results="devstral-24b-4bit",
        streams=[f"{_STAGEB}/ab-devstral-base-a.streams.json", f"{_STAGEB}/ab-devstral-base-b.streams.json"]),
    "qwen36-27b": _model(
        "Qwen3.6-27B 4-bit (DeltaNet)", "dense", "4bit", "blue", "P", baseline="qwen36-27b",
        streams=[f"{_STREAMS}/batched/d27-serial-p1.json", f"{_STREAMS}/batched/d27-serial-p2.json"],
        results="qwen36-27b-4bit"),
    "qwopus38": _model("Qwopus3.8-27B-Coder-Flash 4-bit (DeltaNet)", "dense", "4bit", "green", "v",
                       results="qwopus3.8-27b-coder-flash"),
    "qwen35": _model(
        "Qwen3.5-27B 4-bit (DeltaNet)", "dense", "4bit", "white", "<", baseline="qwen35",
        streams=[f"{_STREAMS}/landed/q35-batched.json", f"{_STREAMS}/landed/q35-batched2.json"],
        results="qwen35-27b-4bit"),
    "qwen3-32b": _model("Qwen3-32B 4-bit DWQ (Dense)", "dense", "4bit", "tan", "*",
                        baseline="qwen3-32b", results="qwen3-32b-4bit",
        streams=[f"{_STAGEB}/ab-qwen3-32b-base-a.streams.json", f"{_STAGEB}/ab-qwen3-32b-base-b.streams.json"]),
}

# Context axis: every power of two from 128 to 256K; per-model charts share one range.
UNIFIED_XLIM = (96, 300_000)
UNIFIED_XTICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
# The rungs of the README decode table.
FLEET_RUNGS = (1024, 8192, 32768)
STD_CONC = [1, 2, 4, 8, 16, 32]
# A cold receipt within this factor of the deepest rung is the same depth on
# another instrument; the rung is drawn.
SAME_DEPTH = 1.05
# A decode bar is hatched when its deepest measured input is short of this.
DEEP_INPUT_TOKENS = 128_000
# The fastest point is labelled with its input only when that input is past this.
SHORT_INPUT_TOKENS = 1_100

CONTEXT_NOTE = (
    "Decode-only tok/s (1000 / TPOT), one request at a time, SGLang + MLX on an M4 Pro 64 GB. Points to 32K are "
    "the benchmarks/baselines.json rungs, or the same rungs of a model's results.json (64-token replies); deeper "
    "points, and models with neither, come from cold chat requests (256-token replies).")
STREAM_NOTE = (
    "Aggregate decode tok/s over N requests started together (about 250 input tokens each, replies of 64 to 200 "
    "tokens, greedy), SGLang + MLX on an M4 Pro 64 GB. Rows are each model's stream receipts under benchmarks/perf/, "
    "the arm nearest its preset's flags.")
SCALE_NOTE = "Each panel has its own y scale"


def fmt_ctx(x, _=None):
    return f"{x / 1024:.0f}K" if x >= 1024 else f"{x:.0f}"


def _fmt_tok(n):
    """A token count for a label: the axis's binary K near a power of two, else decimal thousands."""
    n = int(n)
    if n < 1000:
        return str(n)
    power = 1 << round(math.log2(n))
    if abs(n - power) <= power * 0.005:
        return f"{power // 1024}K"
    return f"{round(n / 1000)}K"


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _valid(rows):
    return [p for p in rows if "error" not in p and p.get("tok_per_sec", 0) > 0]


def _input_len(point):
    return point.get("input_len", point.get("actual_input_tokens", point["context"]))


def load_results(results_dir, bench_dir=BENCH_DIR):
    """A model's results.json; None when missing, unreadable or legacy-suspect."""
    data = _read_json(os.path.join(bench_dir, results_dir, "results.json")) if results_dir else None
    if not data or data.get("legacy_depth_suspect"):
        return None
    return data


def cold_points(paths, bench_dir=BENCH_DIR):
    """Decode points of cold-request receipts: one per depth, the mean over the files."""
    by_target = {}
    for rel in paths:
        receipt = _read_json(os.path.join(bench_dir, rel)) or {}
        for row in receipt.get("cold", []):
            by_target.setdefault(row["target"], []).append(
                (row["cold"]["prompt_tokens"], row["cold"]["decode_tok_s"]))
    points = []
    for target, hits in sorted(by_target.items()):
        points.append({
            "context": target,
            "input_len": round(sum(h[0] for h in hits) / len(hits)),
            "tok_per_sec": round(sum(h[1] for h in hits) / len(hits), 2),
        })
    return points


def stream_points(paths, bench_dir=BENCH_DIR):
    """Aggregate tok/s of stream receipts: one row per stream count, the mean over the files."""
    by_count = {}
    for rel in paths:
        receipt = _read_json(os.path.join(bench_dir, rel)) or {}
        for row in receipt.get("summary", []):
            by_count.setdefault(row["streams"], []).append(row["aggregate_tok_s"])
    return [{"concurrency": count, "tok_per_sec": round(sum(hits) / len(hits), 2)}
            for count, hits in sorted(by_count.items())]


def _path_dates(paths):
    return re.findall(r"\d{4}-\d{2}-\d{2}", " ".join(paths))


def fleet_results(meta, baselines=None, bench_dir=BENCH_DIR):
    """A model's fleet curve and stream rows, in results.json shape, from its sources."""
    own = load_results(meta["results"], bench_dir)
    rows, context_dates = [], []
    preset = (baselines or {}).get(meta["baseline"]) or {}
    for label, row in preset.items():
        if label.isdigit():
            rows.append({"context": int(label), "input_len": row["actual_input_tokens"],
                         "tok_per_sec": row["tok_per_sec"]})
    if rows:
        context_dates.append(baselines["_meta"]["saved"])
    elif own:
        rows = [{"context": p["context"], "input_len": _input_len(p), "tok_per_sec": p["tok_per_sec"]}
                for p in _valid(own.get("context_sweep", [])) if p["context"] in FLEET_RUNGS]
        if rows:
            context_dates.append(own["timestamp"][:10])
    deepest = max((r["input_len"] for r in rows), default=0)
    deep = [p for p in cold_points(meta["cold"], bench_dir) if p["input_len"] > deepest * SAME_DEPTH]
    if deep:
        context_dates += _path_dates(meta["cold"])
    streams = stream_points(meta["streams"], bench_dir)
    return {
        "context_sweep": sorted(rows + deep, key=_input_len),
        "throughput_sweep": streams,
        "context_dates": sorted(set(context_dates)),
        "stream_dates": sorted(set(_path_dates(meta["streams"]))) if streams else [],
    }


def load_fleet(bench_dir=BENCH_DIR):
    """{model key: (model, fleet results)} for every model, in MODELS order."""
    baselines = _read_json(os.path.join(bench_dir, "baselines.json"))
    return {key: (meta, fleet_results(meta, baselines, bench_dir)) for key, meta in MODELS.items()}


def _measured(all_data, field):
    dates = sorted({d for _, results in all_data.values() for d in results.get(field, [])})
    if not dates:
        return ""
    return f" Measured {dates[0]}." if len(dates) == 1 else f" Measured {dates[0]} to {dates[-1]}."


# --- Per-model charts ---

def make_context_chart(model_key, meta, results, out_dir):
    """One model's decode tok/s against input length, on the shared context axis."""
    sweep = _valid(results.get("context_sweep", []))
    if not sweep:
        print(f"  SKIP {model_key} context chart (no valid data)")
        return None
    xs = [_input_len(p) for p in sweep]
    ys = [p["tok_per_sec"] for p in sweep]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, ys, "-", color=meta["color"], linewidth=2, marker=meta["marker"], markersize=7,
            markeredgecolor=PANEL, markeredgewidth=0.8, zorder=5)
    ax.fill_between(xs, ys, alpha=0.12, color=meta["color"])
    peak = max(range(len(ys)), key=ys.__getitem__)
    ax.annotate(f"{ys[peak]:.1f}", (xs[peak], ys[peak]), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, fontweight="bold", color=INK)
    if peak != len(ys) - 1:
        ax.annotate(f"{ys[-1]:.1f} @{_fmt_tok(xs[-1])}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), ha="left", va="center", fontsize=9, color=MUTED)

    ax.set_xscale("log", base=2)
    ax.set_xlim(*UNIFIED_XLIM)
    ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS[::2]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("input tokens")
    ax.set_ylabel("decode tok/s, one request")
    ax.set_title(meta["label"], loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_title(_provenance(results, "context_dates"), loc="right", fontsize=8, color=MUTED, pad=10)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ys) * 1.18)

    fig.tight_layout()
    path = os.path.join(out_dir, "context_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


def make_concurrency_chart(model_key, meta, results, out_dir):
    """One model's aggregate tok/s against concurrent requests."""
    measured = {p["concurrency"]: p["tok_per_sec"] for p in _valid(results.get("throughput_sweep", []))}
    levels = sorted(set(STD_CONC) & set(measured))
    if not levels:
        print(f"  SKIP {model_key} concurrency chart (no valid data)")
        return None
    values = [measured[c] for c in levels]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(values)), values, color=meta["color"], alpha=0.9, width=0.34, zorder=5)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("aggregate tok/s")
    ax.set_title(meta["label"], loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_title(_provenance(results, "stream_dates"), loc="right", fontsize=8, color=MUTED, pad=10)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()
    path = os.path.join(out_dir, "concurrency_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


def _provenance(results, dates_field):
    if "timestamp" in results:
        return " · ".join(s for s in (results.get("sglang_version", ""), results["timestamp"][:10]) if s)
    dates = results.get(dates_field, [])
    return " to ".join(dict.fromkeys([dates[0], dates[-1]])) if dates else ""


# --- Fleet charts ---

def fleet_groups(all_data, group_by=FLEET_DEFAULT_GROUPING):
    """[(panel title, [(key, model, results), ...])] in panel order, empty panels dropped."""
    members = [(key, meta, results) for key, (meta, results) in all_data.items()]
    grouping = FLEET_GROUPINGS[group_by]
    if grouping is None:
        return [(None, members)] if members else []
    field, titles = grouping
    groups = [(title, [m for m in members if m[1][field] == value]) for value, title in titles.items()]
    return [(title, group) for title, group in groups if group]


def fleet_line_style(meta, rank=0):
    """Colour and marker name the model; fill carries the class, dash the quantization.

    rank is the curve's place in its panel's legend. Neighbours take opposite
    dash phases, so two curves that coincide show both colours.
    """
    dense = meta["cls"] == "dense"
    return {
        "color": meta["color"],
        "linewidth": 2.0,
        "linestyle": "-" if meta["quant"] == "8bit" else (3.5 * (rank % 2), (5, 2)),
        "marker": meta["marker"],
        "markersize": 9.5 if meta["marker"] in ("*", "P", "X", "h") else 7.5,
        "markerfacecolor": PANEL if dense else meta["color"],
        "markeredgecolor": meta["color"] if dense else PANEL,
        "markeredgewidth": 1.6 if dense else 0.8,
        "zorder": 5 - 0.01 * rank,
    }


def _fleet_style_key(fig, y):
    fig.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=7.5, markerfacecolor=INK,
               markeredgecolor=PANEL, label="filled marker: MoE"),
        Line2D([], [], linestyle="none", marker="o", markersize=7.5, markerfacecolor=PANEL,
               markeredgecolor=INK, markeredgewidth=1.6, label="hollow marker: dense"),
        Line2D([], [], color=INK, linewidth=2.0, linestyle="-", label="solid line: 8-bit"),
        Line2D([], [], color=INK, linewidth=2.0, linestyle=(0, (5, 2)), label="dashed line: 4-bit"),
    ], loc="upper center", bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, ncol=4, fontsize=9.5,
        frameon=False, handlelength=3.2)


def _fleet_line_chart(all_data, group_by, series, title, xlabel, ylabel, out_path, setup_axis, note, one_axis=None):
    panels = []
    for group_title, members in fleet_groups(all_data, group_by):
        curves = [(meta,) + tuple(series(results)) for _, meta, results in members]
        curves = sorted((c for c in curves if c[1]), key=lambda c: c[2][0], reverse=True)
        if curves:
            panels.append((group_title, curves))
    if not panels:
        print(f"  SKIP {os.path.basename(out_path)} (no fleet data)")
        return None

    all_xs = [x for _, curves in panels for _, xs, _ in curves for x in xs]
    tallest = max(len(curves) for _, curves in panels)
    if len(panels) > 1:
        note = f"{note} {SCALE_NOTE}{'; ' + one_axis if one_axis else ''}."
    note_lines = textwrap.wrap(note, 190)
    # Inches kept under the panels for the style key and the footnote. A save
    # with a tight bounding box would move a figure legend, so none is used.
    footer = 0.5 + 0.15 * len(note_lines)
    height = max(3.8, 0.27 * tallest + 1.1) * len(panels) + 1.0 + footer
    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False, layout="constrained",
                             figsize=(12, height))
    fig.get_layout_engine().set(rect=(0.004, footer / height, 0.99, 1 - footer / height - 0.006))
    for ax, (group_title, curves) in zip(axes[:, 0], panels):
        for rank, (meta, xs, ys) in enumerate(curves):
            ax.plot(xs, ys, label=meta["label"], **fleet_line_style(meta, rank))
        setup_axis(ax, all_xs)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="both")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(y for _, _, ys in curves for y in ys) * 1.12)
        if group_title:
            ax.set_title(f"{group_title} · {len(curves)} models", loc="left", fontsize=11.5, fontweight="bold")
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=9,
                           handlelength=3.4, title="fastest at the left edge first", title_fontsize=8,
                           framealpha=0.5, edgecolor=EDGE, facecolor=PANEL)
        legend.get_title().set_color(MUTED)
    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _fleet_style_key(fig, y=footer / height)
    fig.text(0.012, (footer - 0.4) / height, "\n".join(note_lines), va="top", fontsize=8, color=MUTED,
             linespacing=1.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path}")
    return out_path


def _context_series(results):
    sweep = _valid(results.get("context_sweep", []))
    return [_input_len(p) for p in sweep], [p["tok_per_sec"] for p in sweep]


def _context_axis(ax, xs):
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(xs) / 2 ** 0.5, max(max(xs), UNIFIED_XTICKS[-1]) * 1.15)
    ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def make_combined_context_chart(all_data, group_by=FLEET_DEFAULT_GROUPING, out_path=None):
    """Every model's decode tok/s against input length, one panel per group."""
    return _fleet_line_chart(
        all_data, group_by, _context_series,
        title="Decode speed against context · SGLang + MLX on an M4 Pro 64 GB",
        xlabel="input tokens (server-counted)", ylabel="decode tok/s, one request",
        out_path=out_path or os.path.join(BENCH_DIR, "all_models_context.png"),
        setup_axis=_context_axis, note=CONTEXT_NOTE + _measured(all_data, "context_dates"),
        one_axis="all_models_decode.png puts every model on one axis")


def _concurrency_series(results):
    sweep = sorted(_valid(results.get("throughput_sweep", [])), key=lambda p: p["concurrency"])
    return [p["concurrency"] for p in sweep], [p["tok_per_sec"] for p in sweep]


def _concurrency_axis(ax, xs):
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(xs) / 1.25, max(xs) * 1.25)
    ax.xaxis.set_major_locator(ticker.FixedLocator(STD_CONC))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def make_combined_concurrency_chart(all_data, group_by=FLEET_DEFAULT_GROUPING, out_path=None):
    """Every model's aggregate tok/s against concurrent requests, one panel per group."""
    return _fleet_line_chart(
        all_data, group_by, _concurrency_series,
        title="Throughput against concurrent requests · SGLang + MLX on an M4 Pro 64 GB",
        xlabel="concurrent requests", ylabel="aggregate tok/s",
        out_path=out_path or os.path.join(BENCH_DIR, "all_models_concurrency.png"),
        setup_axis=_concurrency_axis, note=STREAM_NOTE + _measured(all_data, "stream_dates"))


def decode_rows(all_data):
    """Per model: its fastest point and its deepest measured input, with the tokens of each."""
    rows = []
    for key, (meta, results) in all_data.items():
        sweep = _valid(results.get("context_sweep", []))
        if len(sweep) < 2:
            continue
        peak = max(sweep, key=lambda p: p["tok_per_sec"])
        deep = max(sweep, key=_input_len)
        rows.append({
            "key": key, "meta": meta,
            "peak": peak["tok_per_sec"], "peak_in": _input_len(peak),
            "deep": deep["tok_per_sec"], "deep_in": _input_len(deep),
            "reaches": _input_len(deep) >= DEEP_INPUT_TOKENS,
        })
    return rows


def make_decode_bar_chart(all_data, group_by=FLEET_DEFAULT_GROUPING, out_path=None):
    """One row per model: fastest point above its decode speed at the deepest measured input."""
    by_key = {row["key"]: row for row in decode_rows(all_data)}
    sections = []
    for group_title, members in fleet_groups(all_data, group_by):
        picked = sorted((by_key[key] for key, _, _ in members if key in by_key),
                        key=lambda row: row["peak"], reverse=True)
        if picked:
            sections.append((group_title, picked))
    out_path = out_path or os.path.join(BENCH_DIR, "all_models_decode.png")
    if not sections:
        print(f"  SKIP {os.path.basename(out_path)} (no fleet data)")
        return None

    count = sum(len(picked) for _, picked in sections)
    titled = sum(1 for group_title, _ in sections if group_title)
    top = max(row["peak"] for _, picked in sections for row in picked)
    fig, ax = plt.subplots(figsize=(11, 0.46 * count + 0.45 * titled + 2.0))
    h, y, ticks, labels = 0.37, 0.0, [], []
    for group_title, picked in sections:
        if group_title:
            ax.axhline(y, color=RULE, linewidth=0.8)
            ax.text(-0.012, y, f"{group_title} ({len(picked)})", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=10.5, fontweight="bold", color=INK)
            y += 0.85
        for row in picked:
            color = row["meta"]["color"]
            marked = f" @{_fmt_tok(row['peak_in'])}" if row["peak_in"] > SHORT_INPUT_TOKENS else ""
            ax.barh(y - h / 2, row["peak"], h, color=color, alpha=0.45, zorder=3)
            ax.text(row["peak"] + top * 0.008, y - h / 2, f"{row['peak']:.0f}{marked}", va="center",
                    fontsize=9, fontweight="bold", color=INK)
            ax.barh(y + h / 2, row["deep"], h, color=color, alpha=0.95, zorder=3,
                    hatch=None if row["reaches"] else "////", edgecolor=PANEL, linewidth=0.6)
            ax.text(row["deep"] + top * 0.008, y + h / 2, f"{row['deep']:.0f} @{_fmt_tok(row['deep_in'])}",
                    va="center", fontsize=9, color=INK)
            ticks.append(y)
            labels.append(row["meta"]["label"])
            y += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.35, -0.75)
    ax.set_xlim(0, top * 1.16)
    ax.tick_params(axis="x", top=True, labeltop=True)
    ax.set_xlabel("decode tok/s, one request")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Decode speed: fastest point and deepest measured input · SGLang + MLX on an M4 Pro 64 GB",
                 fontsize=13, fontweight="bold", pad=26)
    legend = ax.legend(handles=[
        Patch(facecolor=INK, alpha=0.45, label="fastest point of the curve (1K input unless marked @)"),
        Patch(facecolor=INK, alpha=0.95, label="deepest measured input (@ tokens)"),
        Patch(facecolor=INK, alpha=0.95, hatch="////", edgecolor=PANEL,
              label=f"hatched: deepest point is short of {_fmt_tok(DEEP_INPUT_TOKENS)}"),
    ], loc="lower right", fontsize=9, framealpha=0.6, edgecolor=EDGE, facecolor=PANEL)
    legend.set_zorder(6)
    note = CONTEXT_NOTE + _measured(all_data, "context_dates")
    ax.annotate(textwrap.fill(note, 150), xy=(0, 0), xycoords="axes fraction", xytext=(0, -40),
                textcoords="offset points", ha="left", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")
    return out_path


# SWE-bench chart: one row per archived 26-instance cell, a bar per count the cell has a receipt for.
SWEBENCH_SERIES = (
    ("non_empty", "non-empty patch", FLEET_PALETTE["blue"]),
    ("resolved_local", "resolved, local scorer", FLEET_PALETTE["green"]),
    ("resolved_docker", "resolved, official Docker harness (3090 rig)", FLEET_PALETTE["yellow"]),
)
SWEBENCH_NOTE = (
    "26 SWE-bench Lite instances across 12 repositories, one attempt each, SGLang + MLX on an M4 Pro 64 GB. "
    "A cell draws only the counts it has a receipt for: the opencode dense 27B cells are unscored, and the Docker "
    "count belongs to the export the 3090 rig scored. Per-cell summaries: benchmarks/quality/bakeoff-*.json.")


def swebench_cells(bench_dir=BENCH_DIR):
    quality = os.path.join(bench_dir, "quality")
    names = sorted(n for n in os.listdir(quality) if n.startswith("bakeoff-") and n.endswith(".json"))
    cells = [_read_json(os.path.join(quality, n)) for n in names]
    return sorted((c for c in cells if c), key=lambda c: c["order"])


def make_swebench_chart(out_path=None, bench_dir=BENCH_DIR):
    cells = swebench_cells(bench_dir)
    if not cells:
        print("  SKIP SWE-bench chart (no cells)")
        return None
    total = max(c["total_predictions"] for c in cells)
    height = 0.26
    fig, ax = plt.subplots(figsize=(11, 0.95 * len(cells) + 2.3))
    for row, cell in enumerate(cells):
        mine = [(field, color) for field, _, color in SWEBENCH_SERIES if cell.get(field) is not None]
        for slot, (field, color) in enumerate(mine):
            y = row + (slot - (len(mine) - 1) / 2) * height
            value = cell[field]
            ax.barh(y, value, height * 0.86, color=color, zorder=5)
            ax.text(value + total * 0.008, y, f"{value}/{cell['total_predictions']}", va="center",
                    fontsize=9, color=INK)
    labels = [c["label"] + (f"\n{c['wall_min']} min" if c.get("wall_min") else "") for c in cells]
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(len(cells) - 0.45, -0.55)
    ax.set_xlim(0, total * 1.07)
    ax.set_xticks(range(0, total + 1, 2 if total <= 30 else 50))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"instances of {total} (higher is better)")
    ax.set_title("SWE-bench Lite, 26-instance cells: patches written and patches that resolve",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in SWEBENCH_SERIES],
              loc="lower right", framealpha=0.6, edgecolor=EDGE, facecolor=PANEL, fontsize=9)
    ax.text(0.0, -0.78 / max(len(cells), 1), textwrap.fill(SWEBENCH_NOTE, 150), transform=ax.transAxes,
            va="top", fontsize=8, color=MUTED)
    out_path = out_path or os.path.join(bench_dir, "bakeoff_swebench_lite.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate the benchmark charts under benchmarks/.")
    parser.add_argument("--group-by", choices=sorted(FLEET_GROUPINGS), default=FLEET_DEFAULT_GROUPING,
                        help="panels of the fleet charts: model class (default), quantization, or one panel")
    args = parser.parse_args()

    all_data = load_fleet()
    for key, (meta, fleet) in all_data.items():
        if not meta["results"]:
            continue
        print(f"\n{meta['label']}:")
        out_dir = os.path.join(BENCH_DIR, meta["results"])
        own = load_results(meta["results"])
        has_sweep = bool(own and _valid(own.get("context_sweep", [])))
        make_context_chart(key, meta, own if has_sweep else fleet, out_dir)
        if meta["streams"]:
            make_concurrency_chart(key, meta, fleet, out_dir)

    print("\nFleet:")
    make_combined_context_chart(all_data, args.group_by)
    make_combined_concurrency_chart(all_data, args.group_by)
    make_decode_bar_chart(all_data, args.group_by)
    make_swebench_chart()


if __name__ == "__main__":
    main()
