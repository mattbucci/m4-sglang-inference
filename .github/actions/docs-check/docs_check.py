#!/usr/bin/env python3
"""Docs alignment check shared by the three SGLang rig repos.

One rule set, read from rules.json next to this file, applied to the repo at
--root. Every rig repo runs the same pinned copy through the composite action
in this directory, so a rule change lands in all three at once.

Checks:
  skeleton   README.md H2 headings equal the shared skeleton, in order
  budget     byte / line caps on README.md and the agent files
  longline   no prose line in README.md over the character cap
  banned     no dates, status markers, history framing or host paths in
             README.md / CLAUDE.md / rules-for-agents.md (code fences, inline
             code, link targets and short "quoted" words are ignored, so receipt
             paths may carry dates and a rule may quote what it bans; a line
             ending in <!-- docs-check:allow --> is skipped)
  links      relative links in those files resolve to a file in the repo
  generated  <!-- GENERATED:name cmd="..." --> blocks match their command's output
  schema     benchmarks/*/results.json carry the shared result fields

Exit status is 0 unless --strict is given and a violation was found.
Standard library only, so CI needs no install step.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")
HTML_COMMENT = re.compile(r"<!--.*?-->")
QUOTED = re.compile(r'"[^"\n]{1,40}"|\u201c[^\u201d\n]{1,40}\u201d')
ALLOW = "<!-- docs-check:allow -->"
GEN_OPEN = re.compile(r'^<!-- GENERATED:(?P<name>[\w.-]+) cmd="(?P<cmd>[^"]+)" -->\s*$')
GEN_CLOSE = "<!-- /GENERATED:{} -->"


class Report:
    def __init__(self, gha):
        self.gha = gha
        self.items = []

    def add(self, rule, path, line, msg):
        self.items.append((rule, path, line, msg))
        if self.gha:
            loc = f"file={path}" + (f",line={line}" if line else "")
            print(f"::warning {loc},title=docs-check {rule}::{msg}")
        else:
            where = f"{path}:{line}" if line else path
            print(f"{rule:9} {where}: {msg}")


def read(root, rel):
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read()


def prose_lines(text):
    """(line_no, line) for lines outside fenced code blocks."""
    out, fenced = [], False
    for i, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            out.append((i, line))
    return out


def check_skeleton(rep, root, rules):
    text = read(root, "README.md")
    if text is None:
        rep.add("skeleton", "README.md", 0, "README.md is missing")
        return
    for needle in rules.get("readme_must_link", []):
        if not re.search(r"\]\([^)]*" + re.escape(needle), text):
            rep.add("skeleton", "README.md", 0, f'no link to {needle}: open work and known issues live in GitHub issues')
    want = rules["readme_h2"]
    got = [(i, l[3:].strip()) for i, l in prose_lines(text) if l.startswith("## ")]
    names = [g for _, g in got]
    if names == want:
        return
    for i, name in got:
        if name not in want:
            rep.add("skeleton", "README.md", i, f'H2 "{name}" is not in the shared skeleton')
    for name in want:
        if name not in names:
            rep.add("skeleton", "README.md", 0, f'missing H2 "{name}"')
    present = [n for n in names if n in want]
    if present != [n for n in want if n in present]:
        rep.add("skeleton", "README.md", 0,
                "H2 order differs from the skeleton: " + " / ".join(want))


def check_budgets(rep, root, rules):
    for rel, cap in rules["budgets"].items():
        text = read(root, rel)
        if text is None:
            continue
        size = len(text.encode("utf-8"))
        lines = text.count("\n") + (0 if text.endswith("\n") else 1)
        if "bytes" in cap and size > cap["bytes"]:
            rep.add("budget", rel, 0,
                    f"{size} bytes > {cap['bytes']} cap; split or prune, do not compress")
        if "lines" in cap and lines > cap["lines"]:
            rep.add("budget", rel, 0,
                    f"{lines} lines > {cap['lines']} cap; split or prune, do not compress")


def check_long_lines(rep, root, rules):
    text = read(root, "README.md")
    if text is None:
        return
    cap = rules["readme_max_prose_line"]
    for i, line in prose_lines(text):
        s = line.lstrip()
        if s.startswith("|") or s.startswith("![") or s.startswith("<"):
            continue
        if len(line) > cap:
            rep.add("longline", "README.md", i, f"{len(line)} chars > {cap}; one idea per line, detail in docs/")


def check_banned(rep, root, rules):
    for rel in rules["prose_files"]:
        text = read(root, rel)
        if text is None:
            continue
        pats = [(re.compile(p["re"]), p["why"]) for p in rules["banned"]
                if rel in p.get("files", rules["prose_files"])]
        for i, line in prose_lines(text):
            if ALLOW in line:
                continue
            scrub = HTML_COMMENT.sub("", line)
            scrub = QUOTED.sub('""', scrub)
            scrub = INLINE_CODE.sub("", scrub)
            scrub = LINK_TARGET.sub("]()", scrub)
            for rx, why in pats:
                m = rx.search(scrub)
                if m:
                    rep.add("banned", rel, i, f'"{m.group(0)}": {why}')


def check_links(rep, root, rules):
    for rel in rules["prose_files"]:
        text = read(root, rel)
        if text is None:
            continue
        base = os.path.dirname(os.path.join(root, rel))
        for i, line in prose_lines(text):
            for target in LINK_TARGET.findall(INLINE_CODE.sub("", line)):
                if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                if not path:
                    continue
                full = os.path.normpath(os.path.join(root, path.lstrip("/")) if path.startswith("/")
                                        else os.path.join(base, path))
                if not os.path.exists(full):
                    rep.add("links", rel, i, f"relative link target does not exist: {target}")


def check_generated(rep, root, rules):
    for rel in rules["prose_files"]:
        text = read(root, rel)
        if text is None:
            continue
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            m = GEN_OPEN.match(lines[i])
            if not m:
                i += 1
                continue
            name, cmd = m.group("name"), m.group("cmd")
            close = GEN_CLOSE.format(name)
            try:
                j = lines.index(close, i + 1)
            except ValueError:
                rep.add("generated", rel, i + 1, f"block {name} has no closing {close}")
                break
            committed = "\n".join(lines[i + 1:j]).strip()
            try:
                out = subprocess.run(shlex.split(cmd), cwd=root, capture_output=True,
                                     text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                rep.add("generated", rel, i + 1, f"block {name}: generator failed to run: {e}")
            else:
                if out.returncode != 0:
                    rep.add("generated", rel, i + 1,
                            f"block {name}: generator exited {out.returncode}: {out.stderr.strip()[:200]}")
                elif out.stdout.strip() != committed:
                    rep.add("generated", rel, i + 1,
                            f"block {name} is stale; run `{cmd}` and commit its output between the markers")
            i = j + 1


def check_schema(rep, root, rules):
    with open(os.path.join(HERE, rules["results_fields"]), encoding="utf-8") as f:
        schema = json.load(f)
    bench = os.path.join(root, "benchmarks")
    if not os.path.isdir(bench):
        return
    for entry in sorted(os.listdir(bench)):
        rel = os.path.join("benchmarks", entry, "results.json")
        if not os.path.isfile(os.path.join(root, rel)):
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                data = json.load(f)
        except ValueError as e:
            rep.add("schema", rel, 0, f"not valid JSON: {e}")
            continue
        missing = [k for k in schema["required"] if k not in data]
        if missing:
            rep.add("schema", rel, 0, "missing top-level fields: " + ", ".join(missing))
        rows = data.get(schema["rows_key"])
        if not isinstance(rows, list) or not rows:
            continue
        row_missing = sorted({k for r in rows if isinstance(r, dict)
                              for k in schema["row_required"] if k not in r})
        if row_missing:
            rep.add("schema", rel, 0,
                    f"{schema['rows_key']} rows missing fields: " + ", ".join(row_missing))


CHECKS = {
    "skeleton": check_skeleton,
    "budget": check_budgets,
    "longline": check_long_lines,
    "banned": check_banned,
    "links": check_links,
    "generated": check_generated,
    "schema": check_schema,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="repo to check")
    ap.add_argument("--rules", default=os.path.join(HERE, "rules.json"))
    ap.add_argument("--strict", action="store_true", help="exit 1 on any violation")
    ap.add_argument("--github", action="store_true", help="emit GitHub annotations + step summary")
    ap.add_argument("--only", default="", help="comma list of checks to run")
    args = ap.parse_args()

    with open(args.rules, encoding="utf-8") as f:
        rules = json.load(f)
    rep = Report(args.github)
    only = [c for c in args.only.split(",") if c] or list(CHECKS)
    for name in only:
        CHECKS[name](rep, os.path.abspath(args.root), rules)

    counts = {}
    for rule, *_ in rep.items:
        counts[rule] = counts.get(rule, 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "clean"
    mode = "strict" if args.strict else "warn-only"
    print(f"docs-check ({mode}): {len(rep.items)} violation(s): {summary}")
    if args.github and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"### docs-check ({mode})\n\n{len(rep.items)} violation(s): {summary}\n\n")
            if rep.items:
                f.write("| rule | where | message |\n|---|---|---|\n")
                for rule, path, line, msg in rep.items[:300]:
                    where = f"{path}:{line}" if line else path
                    f.write(f"| {rule} | `{where}` | {msg.replace('|', '/')} |\n")
    return 1 if (args.strict and rep.items) else 0


if __name__ == "__main__":
    sys.exit(main())
