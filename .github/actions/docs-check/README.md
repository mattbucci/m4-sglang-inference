# docs-check

One docs rule set for the three SGLang rig repos: this one, the R9700 repo and the
3090 repo. Each calls this action, so the READMEs, agent files and result files stay
aligned. The R9700 and 3090 repos reference
`mattbucci/m4-sglang-inference/.github/actions/docs-check@<ref>`.

Run locally from any of the three repos:

```bash
python3 path/to/m4-sglang-inference/.github/actions/docs-check/docs_check.py --root .
```

| Check | Rule (values in `rules.json`) |
|---|---|
| skeleton | README H2s are exactly `Results / Presets / Quick start / Stack / Repo map`, in order, and the README links to the repo's GitHub issues |
| budget | README ≤ 20 KB / 300 lines, CLAUDE.md ≤ 12 KB / 200 lines, rules-for-agents.md ≤ 10 KB / 200 lines; split or prune, do not compress |
| longline | no README prose line over 400 characters (tables, images and code are exempt) |
| banned | no ISO dates, status markers, strikethrough, was-vs-now framing, or known-issue / next-step / status / roadmap headings in README, CLAUDE.md, rules-for-agents.md; no host paths or live-run status in README |
| links | relative links in those files resolve |
| generated | a block between `<!-- GENERATED:name cmd="..." -->` and `<!-- /GENERATED:name -->` equals the command's stdout |
| schema | `benchmarks/<preset>/results.json` carries the fields in `results-fields.json` |

Open work and known issues are GitHub issues. History is git history and dated
receipt files under `benchmarks/` and `evals/*/runs/`, which the checks ignore.
Code fences, inline code, link targets and short quoted words are not scanned, so a
receipt path may carry a date and a rule may quote what it bans. A line ending in
`<!-- docs-check:allow -->` skips the banned-pattern scan.

Callers run it report-only (`strict: "false"`). Set `strict: "true"` in a repo's
`.github/workflows/docs-check.yaml` once that repo is clean.
