#!/usr/bin/env python3
"""Aggregate SWE-bench Lite rollout artifacts into a single leaderboard.

Scans `evals/swebench/runs/*/{predictions.jsonl, *.jsonl}` files and
produces:

  1. Per-run summary (model + instance + wall + diff bytes + tool calls
     if a log is colocated)
  2. Per-model totals (success rate, mean wall, etc)
  3. Per-ecosystem totals
  4. Markdown leaderboard suitable for pasting into the README

Usage:
    python evals/swebench/aggregate.py
    python evals/swebench/aggregate.py --runs /tmp/swebench-*  # override location
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Map instance prefix → repo nickname for the per-ecosystem rollup.
def _ecosystem(instance_id: str) -> str:
    prefix = instance_id.split("__", 1)[0]
    return {
        "astropy": "astropy",
        "django": "django",
        "matplotlib": "matplotlib",
        "sympy": "sympy",
        "psf": "requests",
        "pylint-dev": "pylint",
        "pytest-dev": "pytest",
        "scikit-learn": "scikit-learn",
        "sphinx-doc": "sphinx",
        "pydata": "xarray",
        "mwaskom": "seaborn",
        "pallets": "flask",
    }.get(prefix, prefix)


def _count_tool_calls(log_path: Path) -> dict[str, int]:
    if not log_path.exists():
        return {}
    counts: dict[str, int] = defaultdict(int)
    pat = re.compile(r'"tool":"([a-z_]+)"')
    try:
        for line in log_path.read_text().splitlines():
            if '"type":"tool_use"' not in line:
                continue
            m = pat.search(line)
            if m:
                counts[m.group(1)] += 1
    except Exception:
        pass
    return dict(counts)


def _load_run(run_dir: Path) -> list[dict]:
    """Load every predictions.jsonl in run_dir (often just one)."""
    out = []
    # Pattern A: predictions.jsonl + per-instance .diff in run_dir
    pj = run_dir / "predictions.jsonl"
    if pj.exists():
        with pj.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["run_dir"] = run_dir.name
                out.append(rec)
    # Pattern B: alternate naming (4pick-scorecard had one .jsonl per model)
    # but ONLY pick records that look like predictions (have a model_patch
    # field). scores.jsonl from score_local.py shares the same .jsonl
    # extension but stores scoring records with no model_patch, no model
    # name, and `rollout_seconds: null` — those would mis-load as
    # predictions and crash the aggregator at table-print time.
    for jf in run_dir.glob("*.jsonl"):
        if jf.name in ("predictions.jsonl", "scores.jsonl"):
            continue
        with jf.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "model_patch" not in rec:
                    continue  # not a prediction record
                rec["run_dir"] = run_dir.name
                rec.setdefault("model_name_or_path", f"sglang/{jf.stem}")
                out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        default=str(Path(__file__).parent / "runs"),
        help="Path to the runs directory (default: evals/swebench/runs/)",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Also emit a Markdown leaderboard table.",
    )
    ap.add_argument(
        "--export",
        default=None,
        help=(
            "Write a consolidated predictions.jsonl to this path, deduplicated "
            "by (model_name_or_path, instance_id) — most-recent run wins on dup. "
            "Output format matches what the 3090 stack's score_docker.py expects."
        ),
    )
    ap.add_argument(
        "--model-filter",
        default=None,
        help="When exporting, only include records for this model name (e.g. 'sglang/qwen36').",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs)
    if not runs_root.exists():
        print(f"runs dir not found: {runs_root}", file=sys.stderr)
        return 1

    all_recs = []
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        # Runs with "JETSAM" in the name are quarantined (e.g. mid-sweep
        # server death contaminated the per-instance results — see
        # qwen36-missing-ecosystems-JETSAM-2026-05-18/README.md). Their
        # predictions are kept for forensics but excluded from totals.
        if "JETSAM" in sub.name:
            continue
        run_recs = _load_run(sub)
        # Try to find logs for tool-call counts: looks like <inst>.log next
        # to predictions/<inst>.diff, OR /tmp/<run-name>/logs/<inst>.log
        # The latter is gone after cleanup, so we mostly rely on archived
        # diffs not logs. Skip tool-call count if log not co-located.
        for rec in run_recs:
            inst = rec.get("instance_id", "?")
            log_candidate = sub / f"{inst}.log"
            if log_candidate.exists():
                rec["_tool_calls"] = _count_tool_calls(log_candidate)
            else:
                rec["_tool_calls"] = {}
            patch = rec.get("model_patch", "")
            rec["_diff_bytes"] = len(patch)
            rec["_has_patch"] = bool(patch.strip())
            rec["_ecosystem"] = _ecosystem(inst)
            all_recs.append(rec)

    if not all_recs:
        print("no records found", file=sys.stderr)
        return 1

    # ---- Per-run summary ----
    print(f"{'='*92}")
    print(f"SWE-bench Lite aggregate ({len(all_recs)} rollouts across {len(set(r['run_dir'] for r in all_recs))} runs)")
    print(f"{'='*92}")
    print(f"{'run':<38} {'model':<18} {'instance':<32} {'wall':<6} {'diff_B':<7} {'patch':<5}")
    print("-" * 92)
    for r in sorted(all_recs, key=lambda x: (x["run_dir"], x.get("instance_id", ""))):
        model = (r.get("model_name_or_path") or "?").replace("sglang/", "")
        inst = r.get("instance_id", "?")
        wall = f"{r.get('rollout_seconds', 0):.0f}s"
        diff_b = r["_diff_bytes"]
        ok = "Y" if r["_has_patch"] else "."
        print(f"{r['run_dir'][:37]:<38} {model[:17]:<18} {inst[:31]:<32} {wall:<6} {diff_b:<7} {ok:<5}")

    # ---- Per-model rollup ----
    by_model: dict[str, list] = defaultdict(list)
    for r in all_recs:
        model = (r.get("model_name_or_path") or "?").replace("sglang/", "")
        by_model[model].append(r)

    print()
    print("=" * 92)
    print("PER-MODEL ROLLUP")
    print("=" * 92)
    print(f"{'model':<22} {'rollouts':<10} {'patches':<10} {'rate':<8} {'mean_wall':<12} {'ecosystems'}")
    print("-" * 92)
    for model, recs in sorted(by_model.items()):
        patches = sum(1 for r in recs if r["_has_patch"])
        rate = f"{patches}/{len(recs)}"
        rate_pct = (patches / len(recs) * 100) if recs else 0
        mean_wall = sum(r.get("rollout_seconds", 0) for r in recs) / max(1, len(recs))
        ecos = sorted(set(r["_ecosystem"] for r in recs))
        print(f"{model:<22} {len(recs):<10} {patches:<10} {rate:<8} {mean_wall:>8.0f}s   {','.join(ecos)}")

    # ---- Export consolidated predictions.jsonl ----
    if args.export:
        # Dedup by (model, instance) — prefer the record with a non-empty
        # patch when one exists, otherwise fall back to the most-recent run.
        # Without this preference, alphabetic last-write-wins picks 0-byte
        # diagnostic runs (e.g. `qwen36-thinking-*`) over real runs for
        # the same instance.
        by_key: dict[tuple[str, str], dict] = {}
        for r in sorted(all_recs, key=lambda x: x.get("run_dir", "")):
            model = r.get("model_name_or_path", "?")
            inst = r.get("instance_id", "?")
            if args.model_filter and model != args.model_filter:
                continue
            key = (model, inst)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = r
                continue
            # Prefer records with non-empty patches; among equals, newer wins
            # (sort order already enforces "newer last", so simple replace
            # is correct when the incoming record is at least as good).
            if r["_has_patch"] and not existing["_has_patch"]:
                by_key[key] = r
            elif r["_has_patch"] == existing["_has_patch"]:
                by_key[key] = r  # newer run wins ties
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for (model, inst), r in sorted(by_key.items()):
                # Strip aggregator-internal fields before writing.
                rec = {
                    k: v for k, v in r.items()
                    if not k.startswith("_") and k != "run_dir"
                }
                fh.write(json.dumps(rec) + "\n")
        with_patch = sum(1 for r in by_key.values() if r["_has_patch"])
        print()
        print(f"Wrote {len(by_key)} unique predictions to {out_path} "
              f"({with_patch} with non-empty patches)")
        if args.model_filter:
            print(f"  filtered to model={args.model_filter}")
        print(f"  next: scp {out_path} to the 3090 stack, then run "
              f"`python evals/swebench/score_docker.py --predictions <path>`")

    # ---- Markdown emission ----
    if args.markdown:
        print()
        print("=" * 92)
        print("MARKDOWN (paste into README)")
        print("=" * 92)
        print()
        print("| Model | Rollouts | Patches | Rate | Mean wall | Ecosystems |")
        print("|-------|:--------:|:-------:|:----:|:---------:|------------|")
        for model, recs in sorted(by_model.items(), key=lambda kv: -sum(1 for r in kv[1] if r["_has_patch"])):
            patches = sum(1 for r in recs if r["_has_patch"])
            rate = f"{patches}/{len(recs)}"
            mean_wall = sum(r.get("rollout_seconds", 0) for r in recs) / max(1, len(recs))
            ecos = ", ".join(sorted(set(r["_ecosystem"] for r in recs)))
            print(f"| `{model}` | {len(recs)} | {patches} | **{rate}** | {mean_wall:.0f}s | {ecos} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
