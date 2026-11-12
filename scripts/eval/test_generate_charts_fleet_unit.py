#!/usr/bin/env python3
"""Contract test for the fleet charts of scripts/bench/generate_charts.py.

No server and no GPU: metadata, grouping and data sources are checked on
fixtures in a temp dir, and each chart is rendered once into it.

  1. metadata    every model names a class, a quantization, a colour and a
                 marker; colours are unique inside a class, markers fleet-wide;
                 every receipt a model declares exists
  2. palette     the colours of one class separate from each other for normal
                 and colour-deficient vision and read on the panel
  3. grouping    class / quant / none panels
  4. line style  fill carries the class, dash the quantization, neighbours in a
                 legend take opposite dash phases
  5. labels      token counts
  6. sources     baselines before results.json, cold receipts only past the
                 deepest rung, stream rows only from stream receipts, a
                 legacy-suspect results.json never
  7. decode rows fastest point and deepest measured input
  8. render      panels, titles, legend order, hatching, charts without data

    python scripts/eval/test_generate_charts_fleet_unit.py
"""
import importlib.util
import itertools
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "generate_charts", os.path.join(REPO, "scripts", "bench", "generate_charts.py"))
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)

count, fails = 0, []


def check(name, cond, detail=""):
    global count
    count += 1
    if not cond:
        fails.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{' -- ' + str(detail) if detail and not cond else ''}")


# --- Colour difference: OKLab distance x100, Machado 2009 matrices at severity 1.0 ---

CVD = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
}
# Under these a full-colour reader, or a colour-deficient one, cannot tell a pair apart.
NORMAL_FLOOR, CVD_FLOOR, CONTRAST_FLOOR = 15.0, 8.0, 3.0


def linear_rgb(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]


def oklab(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def delta_e(hex_a, hex_b, kind=None):
    def seen(hex_colour):
        rgb = linear_rgb(hex_colour)
        if kind:
            rgb = [min(1.0, max(0.0, sum(w * c for w, c in zip(row, rgb)))) for row in CVD[kind]]
        return oklab(rgb)
    return 100 * sum((x - y) ** 2 for x, y in zip(seen(hex_a), seen(hex_b))) ** 0.5


def contrast(hex_a, hex_b):
    def luminance(hex_colour):
        r, g, b = linear_rgb(hex_colour)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, lo = sorted((luminance(hex_a), luminance(hex_b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# --- Fixtures ---

def write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def sweep(points):
    return [{"context": c, "actual_input_tokens": c, "tok_per_sec": t} for c, t in points]


def cold_receipt(rows):
    return {"cold": [{"target": t, "cold": {"prompt_tokens": n, "decode_tok_s": s}} for t, n, s in rows]}


def stream_receipt(rows):
    return {"summary": [{"streams": n, "aggregate_tok_s": s} for n, s in rows]}


def fixture_fleet(bench):
    """Four models: a deep MoE, a baseline-less MoE, a dense with streams, a legacy-suspect dense."""
    write(os.path.join(bench, "baselines.json"), {
        "_meta": {"saved": "2026-01-05"},
        "deep": {"1024": {"actual_input_tokens": 1024, "tok_per_sec": 90.0},
                 "32768": {"actual_input_tokens": 32768, "tok_per_sec": 60.0},
                 "note": "not a rung"},
        "flat": {"1024": {"actual_input_tokens": 1024, "tok_per_sec": 14.0},
                 "32768": {"actual_input_tokens": 32767, "tok_per_sec": 12.0}},
    })
    write(os.path.join(bench, "perf/2026-01-06/a.json"), cold_receipt([(32768, 32799, 70.0), (131072, 129900, 40.0)]))
    write(os.path.join(bench, "perf/2026-01-06/b.json"), cold_receipt([(131072, 129800, 44.0)]))
    write(os.path.join(bench, "perf/2026-01-07/s1.json"), stream_receipt([(1, 14.0), (2, 25.0)]))
    write(os.path.join(bench, "perf/2026-01-07/s2.json"), stream_receipt([(1, 15.0), (2, 27.0), (4, 36.0)]))
    write(os.path.join(bench, "own/results.json"), {
        "timestamp": "2026-01-03T10:00:00", "sglang_version": "v1",
        "context_sweep": sweep([(128, 80.0), (1024, 75.0), (8192, 50.0), (32768, 30.0)])
        + [{"context": 65536, "error": "timeout_or_crash"}],
        "throughput_sweep": [{"concurrency": 1, "tok_per_sec": 99.0}, {"concurrency": 2, "tok_per_sec": 140.0}],
    })
    write(os.path.join(bench, "suspect/results.json"), {
        "timestamp": "2025-12-01T10:00:00", "legacy_depth_suspect": True,
        "context_sweep": sweep([(1024, 500.0), (32768, 400.0)]),
    })
    models = {
        "deep": charts._model("Deep 4-bit (MoE)", "moe", "4bit", "blue", "o", baseline="deep",
                              cold=["perf/2026-01-06/a.json", "perf/2026-01-06/b.json"]),
        "own": charts._model("Own 8-bit (MoE)", "moe", "8bit", "green", "^", results="own"),
        "flat": charts._model("Flat 4-bit (Dense)", "dense", "4bit", "blue", "P", baseline="flat",
                              streams=["perf/2026-01-07/s1.json", "perf/2026-01-07/s2.json"], results="own"),
        "suspect": charts._model("Suspect 4-bit (Dense)", "dense", "4bit", "green", "v", results="suspect"),
    }
    baselines = charts._read_json(os.path.join(bench, "baselines.json"))
    return {key: (meta, charts.fleet_results(meta, baselines, bench)) for key, meta in models.items()}


print("1. metadata")
models = charts.MODELS
check("every class and quantization is a known one",
      all(m["cls"] in charts.MODEL_CLASSES and m["quant"] in charts.MODEL_QUANTS for m in models.values()))
check("the class follows the label", all(("(MoE" in m["label"]) == (m["cls"] == "moe") for m in models.values()))
check("the quantization follows the label",
      all(("8-bit" in m["label"] or "oQ8e" in m["label"]) == (m["quant"] == "8bit") for m in models.values()))
check("every colour is a palette hex", all(m["color"] in charts.FLEET_PALETTE.values() for m in models.values()))
for cls in charts.MODEL_CLASSES:
    colours = [m["color"] for m in models.values() if m["cls"] == cls]
    check(f"colours are unique inside {cls}", len(colours) == len(set(colours)), colours)
markers = [m["marker"] for m in models.values()]
check("no two models share a marker", len(markers) == len(set(markers)), markers)
baselines = charts._read_json(os.path.join(charts.BENCH_DIR, "baselines.json"))
check("every declared baseline is a preset of baselines.json",
      all(m["baseline"] in baselines for m in models.values() if m["baseline"]))
missing = [rel for m in models.values() for rel in list(m["cold"]) + list(m["streams"])
           if not os.path.isfile(os.path.join(charts.BENCH_DIR, rel))]
check("every declared receipt exists", not missing, missing)
missing = [m["results"] for m in models.values()
           if m["results"] and not os.path.isfile(os.path.join(charts.BENCH_DIR, m["results"], "results.json"))]
check("every declared results directory holds a results.json", not missing, missing)

print("2. palette")
check("the measure tells the palette's known weak pair",
      delta_e(charts.FLEET_PALETTE["blue"], charts.FLEET_PALETTE["violet"], "protan") < 4
      and delta_e(charts.FLEET_PALETTE["orange"], charts.FLEET_PALETTE["tan"]) < NORMAL_FLOOR)
for cls in charts.MODEL_CLASSES:
    colours = sorted({m["color"] for m in models.values() if m["cls"] == cls})
    pairs = list(itertools.combinations(colours, 2))
    weak = [(a, b, round(delta_e(a, b), 1)) for a, b in pairs if delta_e(a, b) < NORMAL_FLOOR]
    check(f"{cls}: every colour pair separates for full-colour vision", not weak, weak)
    for kind in CVD:
        weak = [(a, b, round(delta_e(a, b, kind), 1)) for a, b in pairs if delta_e(a, b, kind) < CVD_FLOOR]
        check(f"{cls}: every colour pair separates for {kind} vision", not weak, weak)
dim = [(m["label"], round(contrast(m["color"], charts.PANEL), 2)) for m in models.values()
       if contrast(m["color"], charts.PANEL) < CONTRAST_FLOOR]
check("every colour reads on the panel", not dim, dim)

print("3. grouping")
with tempfile.TemporaryDirectory() as bench:
    fleet = fixture_fleet(bench)
check("the default grouping is by class", charts.FLEET_DEFAULT_GROUPING == "class")
groups = charts.fleet_groups(fleet, "class")
check("class panels are MoE then dense",
      [(title, [key for key, _, _ in members]) for title, members in groups]
      == [("MoE", ["deep", "own"]), ("Dense", ["flat", "suspect"])])
groups = charts.fleet_groups(fleet, "quant")
check("quant panels are 8-bit then 4-bit",
      [(title, [key for key, _, _ in members]) for title, members in groups]
      == [("8-bit", ["own"]), ("4-bit", ["deep", "flat", "suspect"])])
groups = charts.fleet_groups(fleet, "none")
check("no grouping is one untitled panel", len(groups) == 1 and groups[0][0] is None and len(groups[0][1]) == 4)
only_moe = {key: value for key, value in fleet.items() if value[0]["cls"] == "moe"}
check("an empty panel is dropped", [title for title, _ in charts.fleet_groups(only_moe, "class")] == ["MoE"])
check("an empty fleet has no panels", charts.fleet_groups({}, "none") == [])
try:
    charts.fleet_groups(fleet, "size")
    check("an unknown grouping is rejected", False)
except KeyError:
    check("an unknown grouping is rejected", True)
real = {key: (meta, {}) for key, meta in models.items()}
for grouping in charts.FLEET_GROUPINGS:
    placed = [key for _, members in charts.fleet_groups(real, grouping) for key, _, _ in members]
    check(f"every model lands in one {grouping} panel", sorted(placed) == sorted(models), placed)

print("4. line style")
moe, dense = fleet["deep"][0], fleet["flat"][0]
style = charts.fleet_line_style(moe)
check("a MoE marker is filled with the model colour",
      style["markerfacecolor"] == moe["color"] and style["markeredgecolor"] == charts.PANEL)
style = charts.fleet_line_style(dense)
check("a dense marker is hollow, ringed with the model colour",
      style["markerfacecolor"] == charts.PANEL and style["markeredgecolor"] == dense["color"])
check("an 8-bit line is solid", charts.fleet_line_style(fleet["own"][0], rank=1)["linestyle"] == "-")
first, second = charts.fleet_line_style(moe, rank=0), charts.fleet_line_style(moe, rank=1)
check("a 4-bit line is dashed", first["linestyle"][1] == (5, 2) and second["linestyle"][1] == (5, 2))
check("legend neighbours take opposite dash phases", first["linestyle"][0] != second["linestyle"][0])
check("an earlier legend entry draws on top", first["zorder"] > second["zorder"])
check("the style names the model", first["color"] == moe["color"] and first["marker"] == moe["marker"])

print("5. labels")
for tokens, label in [(959, "959"), (1023, "1K"), (1025, "1K"), (8202, "8K"), (32767, "32K"), (30422, "30K"),
                      (64958, "65K"), (128000, "128K"), (129887, "130K"), (262144, "256K")]:
    check(f"{tokens} tokens reads {label}", charts._fmt_tok(tokens) == label, charts._fmt_tok(tokens))
check("axis ticks are binary", charts.fmt_ctx(131072) == "128K" and charts.fmt_ctx(512) == "512")

print("6. sources")
with tempfile.TemporaryDirectory() as bench:
    fleet = fixture_fleet(bench)
    cold = charts.cold_points(["perf/2026-01-06/a.json", "perf/2026-01-06/b.json", "perf/absent.json"], bench)
    streams = charts.stream_points(["perf/2026-01-07/s1.json", "perf/2026-01-07/s2.json"], bench)
    suspect = charts.load_results("suspect", bench)
    absent = charts.load_results("absent", bench)
check("cold receipts average per depth",
      cold == [{"context": 32768, "input_len": 32799, "tok_per_sec": 70.0},
               {"context": 131072, "input_len": 129850, "tok_per_sec": 42.0}], cold)
check("stream receipts average per stream count",
      streams == [{"concurrency": 1, "tok_per_sec": 14.5}, {"concurrency": 2, "tok_per_sec": 26.0},
                  {"concurrency": 4, "tok_per_sec": 36.0}], streams)
check("a legacy-suspect or missing results.json loads as nothing", suspect is None and absent is None)
deep = fleet["deep"][1]
check("baseline rungs are drawn, a cold point only when it is deeper than the deepest rung",
      [(p["input_len"], p["tok_per_sec"]) for p in deep["context_sweep"]]
      == [(1024, 90.0), (32768, 60.0), (129850, 42.0)], deep["context_sweep"])
check("context dates are the baseline save and the receipts' path dates",
      deep["context_dates"] == ["2026-01-05", "2026-01-06"], deep["context_dates"])
own = fleet["own"][1]
check("without a baseline the results.json rungs are drawn",
      [(p["input_len"], p["tok_per_sec"]) for p in own["context_sweep"]]
      == [(1024, 75.0), (8192, 50.0), (32768, 30.0)], own["context_sweep"])
check("the results.json date is the curve's date", own["context_dates"] == ["2026-01-03"])
check("results.json load rows are not a stream curve", own["throughput_sweep"] == [] and own["stream_dates"] == [])
flat = fleet["flat"][1]
check("a baseline wins over the model's results.json",
      [p["tok_per_sec"] for p in flat["context_sweep"]] == [14.0, 12.0], flat["context_sweep"])
check("stream rows and their date come from the declared receipts",
      [p["tok_per_sec"] for p in flat["throughput_sweep"]] == [14.5, 26.0, 36.0]
      and flat["stream_dates"] == ["2026-01-07"], flat)
check("a legacy-suspect results.json is never drawn",
      fleet["suspect"][1]["context_sweep"] == [] and fleet["suspect"][1]["throughput_sweep"] == [])
check("the footnote names the measurement span",
      charts._measured(fleet, "context_dates") == " Measured 2026-01-03 to 2026-01-06."
      and charts._measured(fleet, "stream_dates") == " Measured 2026-01-07."
      and charts._measured({}, "context_dates") == "")

print("7. decode rows")
rows = {row["key"]: row for row in charts.decode_rows(fleet)}
check("a model with fewer than two points has no row", sorted(rows) == ["deep", "flat", "own"], sorted(rows))
check("the fastest point and the deepest input carry their tokens",
      (rows["deep"]["peak"], rows["deep"]["peak_in"], rows["deep"]["deep"], rows["deep"]["deep_in"])
      == (90.0, 1024, 42.0, 129850), rows["deep"])
check("only a curve past the deep threshold reaches it",
      rows["deep"]["reaches"] and not rows["flat"]["reaches"] and not rows["own"]["reaches"])
check("the input length is the server's count", rows["flat"]["deep_in"] == 32767, rows["flat"]["deep_in"])
errors_only = {"x": (moe, {"context_sweep": [{"context": 1024, "error": "timeout"},
                                             {"context": 8192, "tok_per_sec": 10.0}]})}
check("error rows are not points", charts.decode_rows(errors_only) == [])

print("8. render")
captured = []
real_close = charts.plt.close
charts.plt.close = captured.append


def render(make, *args, **kwargs):
    captured.clear()
    path = make(*args, **kwargs)
    figure = captured[-1] if captured else None
    return path, figure


def panel_legends(figure):
    return [[text.get_text() for text in ax.get_legend().get_texts()] for ax in figure.axes]


with tempfile.TemporaryDirectory() as out:
    context_png = os.path.join(out, "context.png")
    path, fig = render(charts.make_combined_context_chart, fleet, "class", context_png)
    check("the context chart is written", path == context_png and os.path.getsize(context_png) > 10_000)
    check("one panel per class, titled with its model count",
          [ax.get_title(loc="left") for ax in fig.axes] == ["MoE · 2 models", "Dense · 1 models"],
          [ax.get_title(loc="left") for ax in fig.axes])
    check("a legend lists its panel fastest at the left edge first",
          panel_legends(fig) == [["Deep 4-bit (MoE)", "Own 8-bit (MoE)"], ["Flat 4-bit (Dense)"]], panel_legends(fig))
    check("the style key is a figure legend of four entries",
          len(fig.legends) == 1 and len(fig.legends[0].get_texts()) == 4)
    check("the footnote says the panels differ in scale and names the one-axis chart",
          any("own y scale; all_models_decode.png" in " ".join(t.get_text().split()) for t in fig.texts))
    real_close(fig)

    path, fig = render(charts.make_combined_context_chart, fleet, "none", context_png)
    check("no grouping draws one untitled panel",
          len(fig.axes) == 1 and fig.axes[0].get_title(loc="left") == "" and len(panel_legends(fig)[0]) == 3)
    check("one panel carries no scale note", not any("own y scale" in t.get_text() for t in fig.texts))
    real_close(fig)

    stream_png = os.path.join(out, "streams.png")
    path, fig = render(charts.make_combined_concurrency_chart, fleet, "class", stream_png)
    check("the stream chart draws only models with stream receipts",
          path == stream_png and panel_legends(fig) == [["Flat 4-bit (Dense)"]], panel_legends(fig))
    real_close(fig)

    decode_png = os.path.join(out, "decode.png")
    path, fig = render(charts.make_decode_bar_chart, fleet, "class", decode_png)
    ax = fig.axes[0]
    row_labels = [t.get_text() for t in ax.get_yticklabels()]
    check("decode rows are sectioned by class, fastest first",
          row_labels == ["Deep 4-bit (MoE)", "Own 8-bit (MoE)", "Flat 4-bit (Dense)"], row_labels)
    section_titles = [t.get_text() for t in ax.texts if t.get_fontweight() == "bold" and "(" in t.get_text()]
    check("each section is titled with its model count", section_titles == ["MoE (2)", "Dense (1)"], section_titles)
    bars = [p for p in ax.patches if p.get_width() > 0]
    hatched = [p for p in bars if p.get_hatch()]
    check("two bars per model", len(bars) == 6, len(bars))
    check("a deepest point short of the threshold is hatched", len(hatched) == 2 and
          sorted(p.get_width() for p in hatched) == [12.0, 30.0], [p.get_width() for p in hatched])
    labels = [t.get_text() for t in ax.texts]
    check("bars are labelled with the speed and the deepest input",
          "42 @130K" in labels and "12 @32K" in labels and "90" in labels, labels)
    real_close(fig)

    for make in (charts.make_combined_context_chart, charts.make_combined_concurrency_chart,
                 charts.make_decode_bar_chart):
        path, fig = render(make, {}, "class", os.path.join(out, "empty.png"))
        check(f"{make.__name__} without data draws nothing",
              path is None and fig is None and not os.path.exists(os.path.join(out, "empty.png")))

    meta, results = fleet["flat"]
    path, fig = render(charts.make_context_chart, "flat", meta, results, out)
    check("a per-model context chart is written", path == os.path.join(out, "context_vs_toks.png")
          and os.path.isfile(path))
    real_close(fig)
    path, fig = render(charts.make_concurrency_chart, "flat", meta, results, out)
    check("a per-model stream chart has one bar per stream count",
          os.path.isfile(path) and len(fig.axes[0].patches) == 3)
    real_close(fig)
    path, fig = render(charts.make_concurrency_chart, "own", *fleet["own"], out)
    check("a model without stream receipts has no stream chart", path is None and fig is None)

charts.plt.close = real_close

print(f"\n{count - len(fails)}/{count} PASS")
sys.exit(1 if fails else 0)
