#!/usr/bin/env python3
"""Audit archived SWE-bench rollouts for answer-leak channels.

Scans every archived session — `runs/<run>/logs/<instance>.log` (the
opencode JSON event stream or omp's `--mode json` stream) and
`runs/<run>/sessions/<instance>/*.jsonl` (omp's pi-row session store) — for
the three ways a session could obtain the upstream fix instead of writing
one:

  web      — opencode `webfetch` / `websearch`, omp `web_search` and
             `github` tool calls, an omp `read` whose path is a URL
  network  — bash commands that reach the internet (curl, wget, pip
             download / non-editable install, git fetch/pull/ls-remote/clone,
             package managers, HTTP client one-liners); omp `eval` code that
             calls fetch / requests / urllib
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


JS_FETCH = re.compile(r"\b(?:fetch|https?\.(?:get|request)|axios(?:\.(?:get|post))?|got|undici\.request)\(")
PY_FETCH = re.compile(r"\b(?:requests\.(?:get|post|request|Session)|urlopen|httpx\.|http\.client\.|urllib3)")


def events(path: Path):
    """Yield (tool, input) pairs from an archived session file.

    Two shapes: opencode's event stream (`part.type == "tool"`, input under
    `part.state.input`) and the pi row format omp writes (an assistant
    `message.content` list with `toolCall` blocks, arguments under
    `arguments`), the latter both in omp's `--mode json` stdout log and in
    its session jsonl.
    """
    for line in path.open(errors="replace"):
        if '"tool"' not in line and '"toolCall"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        p = e.get("part")
        if isinstance(p, dict) and p.get("type") == "tool":
            yield p.get("tool"), (p.get("state", {}).get("input", {}) or {})
            continue
        m = e.get("message")
        if isinstance(m, dict) and m.get("role") == "assistant":
            for c in m.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "toolCall":
                    yield c.get("name"), (c.get("arguments") or {})


def session_files(run: Path, inst: str) -> list[Path]:
    files = []
    log = run / "logs" / f"{inst}.log"
    if log.exists():
        files.append(log)
    sess = run / "sessions" / inst
    if sess.is_dir():
        files.extend(sorted(sess.glob("*.jsonl")))
    return files


def instances(run: Path) -> list[str]:
    names = set()
    if (run / "logs").is_dir():
        names.update(p.stem for p in (run / "logs").glob("*.log") if not p.name.endswith(".env.log"))
    if (run / "sessions").is_dir():
        names.update(p.name for p in (run / "sessions").iterdir() if p.is_dir())
    return sorted(names)


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


def classify(tool, inp, web, net, hist):
    if tool in ("webfetch", "websearch", "web_search", "github"):
        web.append(inp.get("url") or inp.get("query") or inp.get("op") or "?")
    elif tool == "read" and re.match(r"https?://", str(inp.get("path") or ""), re.I):
        web.append(str(inp.get("path")))
    elif tool == "bash":
        cmd = inp.get("command", "") or ""
        if NETWORK.search(cmd):
            net.append(cmd.strip().replace("\n", " ")[:100])
        if HISTORY.search(cmd):
            hist.append(cmd.strip().replace("\n", " ")[:100])
    elif tool == "eval":
        code = str(inp.get("code") or "")
        lang = str(inp.get("language") or "js").lower()
        pat = PY_FETCH if lang in ("py", "python") else JS_FETCH
        if pat.search(code) or NETWORK.search(code):
            net.append(f"eval[{lang}]: " + code.strip().replace("\n", " ")[:90])


def audit(run: Path):
    rows = []
    for inst in instances(run):
        web, net, hist = [], [], []
        seen = set()
        for f in session_files(run, inst):
            for tool, inp in events(f):
                # the stdout log and the session store carry the same calls
                key = (tool, json.dumps(inp, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
                classify(tool, inp, web, net, hist)
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
        total += len(instances(run))
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
