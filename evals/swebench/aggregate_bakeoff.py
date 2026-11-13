#!/usr/bin/env python3
"""Per-cell SWE-bench Lite summaries for the README chart.

Each cell is one archived 26-instance run. Counts come from the run's own
files: non-empty patches from predictions.jsonl, the local scorer's resolved
count from scores.jsonl (score_local.py), the official Docker count from the
summary the 3090 rig exported. A count a run has no receipt for stays null and
the chart draws no bar for it.

  python3 evals/swebench/aggregate_bakeoff.py     # writes benchmarks/quality/bakeoff-*.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "benchmarks", "quality")

# key, chart label, preset, scaffold, run dir (or predictions export), extras.
# local_resolved is given where the score is recorded in another cell's README
# instead of a scores.jsonl; docker is a summary under exports/.
CELLS = [
    {"key": "qwen36-omp", "label": "qwen36, omp", "preset": "qwen36", "scaffold": "omp",
     "run": "runs/qwen36-omp-final-26inst-2026-11-12", "wall_min": 81},
    {"key": "qwen36-opencode", "label": "qwen36, opencode", "preset": "qwen36", "scaffold": "opencode",
     "run": "runs/qwen36-26inst-p023-2026-10-31", "wall_min": 87, "local_resolved": 5,
     "local_source": "runs/qwen36-26inst-omp-2026-11-08/README.md"},
    {"key": "qwen36-opencode-docker", "label": "qwen36, opencode (Docker-scored export)", "preset": "qwen36",
     "scaffold": "opencode", "predictions": "exports/qwen36-predictions.jsonl",
     "docker": "exports/qwen36-docker-summary.json"},
    {"key": "qwopus38-opencode", "label": "Qwopus3.8-27B-Coder-Flash, opencode", "preset": "qwen36-27b",
     "scaffold": "opencode", "run": "runs/qwopus38-27b-coder-flash-26inst-sealed-2026-11-02"},
    {"key": "qwen38-opencode", "label": "Qwen3.8-27B, opencode", "preset": "qwen36-27b", "scaffold": "opencode",
     "run": "runs/qwen38-27b-community-26inst-sealed-2026-11-02"},
    {"key": "flint-omp", "label": "Flint oQ8e, omp bash-only", "preset": "flint", "scaffold": "omp",
     "run": "runs/flint-oQ8e-26inst-omp-bashonly-remap-sealed-2026-11-08", "wall_min": 780},
]


def _jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(cell):
    preds_path = cell.get("predictions") or os.path.join(cell["run"], "predictions.jsonl")
    preds = _jsonl(os.path.join(HERE, preds_path))
    out = {"key": cell["key"], "label": cell["label"], "preset": cell["preset"], "scaffold": cell["scaffold"],
           "total_predictions": len(preds),
           "non_empty": sum(1 for p in preds if (p.get("model_patch") or "").strip()),
           "resolved_local": cell.get("local_resolved"), "resolved_docker": None,
           "wall_min": cell.get("wall_min"),
           "source": "evals/swebench/" + (cell.get("run") or cell["predictions"])}
    if "local_source" in cell:
        out["local_source"] = "evals/swebench/" + cell["local_source"]
    scores = os.path.join(HERE, cell.get("run", ""), "scores.jsonl")
    if cell.get("run") and os.path.exists(scores):
        out["resolved_local"] = sum(1 for s in _jsonl(scores) if s.get("resolved"))
    if "docker" in cell:
        with open(os.path.join(HERE, cell["docker"])) as fh:
            out["resolved_docker"] = json.load(fh)["resolved"]
        out["docker_source"] = "evals/swebench/" + cell["docker"]
    return out


def main():
    for order, cell in enumerate(CELLS):
        row = summarize(cell)
        row["order"] = order
        path = os.path.join(OUT, f"bakeoff-{cell['key']}.json")
        with open(path, "w") as fh:
            json.dump(row, fh, indent=2)
            fh.write("\n")
        print(f"{row['label']}: non-empty {row['non_empty']}/{row['total_predictions']}, "
              f"local {row['resolved_local']}, docker {row['resolved_docker']}")


if __name__ == "__main__":
    main()
