#!/usr/bin/env python3
"""Audit archived SWE-bench rollouts for answer-leak channels.

Scans every `runs/<run>/logs/<instance>.log` (opencode JSON event stream)
for the three ways a session could obtain the upstream fix instead of
writing one:

  web      — opencode `webfetch` / `websearch` tool calls
  network  — bash commands that reach the internet (curl, wget, pip
             download / non-editable install, git fetch/pull/ls-remote/clone,
             package managers, HTTP client one-liners)
  history  — git commands that read refs past the base commit (`--all`,
             `branch -a`, `--remotes`, `origin/…`, `tags`, `show <sha>`)

The sealed harness (ancestor-only checkout, localhost-only sandbox, web
tools off) closes all three for new runs; this script is the evidence for
the archive. Prints a markdown table of every session with a hit and the
size of the patch that session produced, so a reader can see whether a
scored prediction came out of a leaking session.

    python3 evals/swebench/audit_rollouts.py                 # all runs
    python3 evals/swebench/audit_rollouts.py --runs runs/qwen36-26inst-p023-2026-10-31
"""
import argparse
import json
import re
from pathlib import Path

NETWORK = re.compile(
    r"\b(curl|wget|aria2c|nc|ncat|telnet)\b"
    r"|\bpip3?\s+download\b"
    r"|\bpip3?\s+install\s+(?!-e\b|--editable\b|\.\b|-r\b)"
    r"|\buv\s+pip\s+install\s+(?!-e\b|\.\b)"
    r"|\bgit\s+(fetch|pull|ls-remote|clone|remote\s+add|submodule\s+update)\b"
    r"|\b(conda|brew|npm|apt(-get)?)\s+install\b"
    r"|\bgh\s+(api|pr|issue|repo)\b"
    r"|urllib\.request\.urlopen|urlopen\(|httpx\.(get|post|Client)|http\.client\.HTTPS?Connection"
)
# A URL string literal is not a network call: django settings and the
# requests test suite carry http://example.com-style values in commands
# that never leave the process, so bare URLs are not matched.
HISTORY = re.compile(
    r"\bgit\s+(log|show|diff|branch|tag|rev-list|reflog|name-rev|describe|cherry|for-each-ref)\b[^|;&]*"
    r"(--all\b|-a\b|--remotes\b|\borigin/|\brefs/|\btags?\b)"
    r"|\bgit\s+show\s+[0-9a-f]{7,40}\b"
    r"|\bgit\s+(branch|tag)\s*(-l|--list)?\s*($|[|;&])"
)


def events(path: Path):
    for line in path.open(errors="replace"):
        if '"tool"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        p = e.get("part")
        if isinstance(p, dict) and p.get("type") == "tool":
            yield p


def patch_size(run: Path, inst: str) -> int:
    pred = run / "predictions.jsonl"
    if not pred.exists():
        return -1
    for line in pred.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("instance_id") == inst:
            return len(r.get("model_patch") or "")
    return -1


def audit(run: Path):
    rows = []
    for log in sorted((run / "logs").glob("*.log")) if (run / "logs").is_dir() else []:
        inst = log.stem
        web, net, hist = [], [], []
        for p in events(log):
            tool = p.get("tool")
            inp = p.get("state", {}).get("input", {}) or {}
            if tool in ("webfetch", "websearch"):
                web.append(inp.get("url") or inp.get("query") or "?")
            elif tool == "bash":
                cmd = inp.get("command", "") or ""
                if NETWORK.search(cmd):
                    net.append(cmd.strip().replace("\n", " ")[:100])
                if HISTORY.search(cmd):
                    hist.append(cmd.strip().replace("\n", " ")[:100])
        if web or net or hist:
            rows.append((inst, web, net, hist, patch_size(run, inst)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs (default: every runs/*)")
    ap.add_argument("--verbose", action="store_true", help="print the matching commands")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent / "runs"
    runs = [Path(r) for r in a.runs] if a.runs else sorted(d for d in root.iterdir() if d.is_dir())
    total = hits = 0
    print("| Run | Instance | web | network | history | patch bytes |")
    print("|---|---|:-:|:-:|:-:|--:|")
    for run in runs:
        logs = list((run / "logs").glob("*.log")) if (run / "logs").is_dir() else []
        total += len(logs)
        for inst, web, net, hist, size in audit(run):
            hits += 1
            print(f"| `{run.name}` | `{inst}` | {len(web)} | {len(net)} | {len(hist)} | {size} |")
            if a.verbose:
                for u in web:
                    print(f"|  | webfetch | `{u}` | | | |")
                for c in net:
                    print(f"|  | network | `{c}` | | | |")
                for c in hist:
                    print(f"|  | history | `{c}` | | | |")
    print(f"\n{total} sessions scanned, {hits} with at least one hit.")


if __name__ == "__main__":
    main()
