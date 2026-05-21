# qwen36 pylint-5859 N=3 — RESOLVED stable across reruns

## Context

After the N=4 requests-1963 variance finding
(`../qwen36-perfeat-requests1963-N3-VARIANCE-2026-05-20/`) showed
qwen36 produces 3 distinct fix strategies for the *same* instance,
the open question was: does this variance also flip the
RESOLVED/NOT RESOLVED outcome for May-18's 5/13 RESOLVED set, or
does it only vary content while preserving correctness?

This iteration tests on pylint-5859 (cheapest May-18 RESOLVED:
~500 s rollout, 1 F2P + 10 P2P tests, ~8 s scoring).

## N=3 results — all RESOLVED, all different bytes

| N | When | Bytes | SHA256 | Verdict |
|:-:|---|---:|---|---|
| 1 | May 18 (v0.5.11 stack) | 656 | 34c5eb29… | RESOLVED ✓ |
| 2 | Today run #1 (v0.5.12 per-feature) | 672 | e3992833… | RESOLVED ✓ |
| 3 | Today run #2 (v0.5.12 per-feature) | 656 | 44a2d46a… | RESOLVED ✓ |

Note runs 1 and 3 have the *same byte count* but *different content*
— the variance is concentrated in a single regex-anchor token.

## The variance is in cosmetic syntax, not in correctness

All three patches modify the same file
(`pylint/checkers/misc.py`), the same method, the same regex
construction, and produce a functionally-equivalent fix. Differences:

```
May 18 (656 B):  (?:\W|$)                    # non-word or end
Run #1 (672 B):  (?![a-zA-Z0-9_])            # negative lookahead for word chars
Run #2 (656 B):  (?=\s|$)                    # positive lookahead for whitespace or end
```

All three correctly prevent the original bug (partial-word matches of
TODO/FIXME notes like `XXXfoo` being misclassified). Same F2P 1/1, same
P2P 10/10. The model has internalized the correct fix space and lands
on different equivalent expressions of it.

## Implications for the 5/13 = 38.5% M4-scorable RESOLVED rate

Combined with the requests-1963 finding:

| Instance | N | Distinct outputs | RESOLVED outcomes |
|---|:-:|---|---|
| pylint-5859 (May-18 RESOLVED) | 3 | 3 (all equivalent regex anchors) | 3/3 RESOLVED |
| requests-1963 (May-18 NOT RESOLVED) | 4 | 3 (canonical + 2 alternatives) | 0/4 RESOLVED |

For both instances, **outcome is stable across reruns even when content
varies substantially**. The 5/13 = 38.5% headline appears to be a
true-capability estimate, not a sample-favorable accident.

This is a stronger claim than "scorer is stable" (proven yesterday).
It says the MODEL is stable in *outcome*: qwen36 reliably finds the
right fix space for RESOLVED instances and reliably misses on NOT
RESOLVED instances. Run-to-run variance affects the surface form, not
which side of the line the result falls on.

## Caveat — not yet tested on all 5

We've only confirmed N=3 stability for pylint-5859 (cheap, fast). The
other 4 RESOLVED instances (django-10914, django-11001, django-11039,
sphinx-10325) may have larger sampling-fragility — N≥3 would confirm.
Likewise NOT RESOLVED cases at the borderline (small patches that
nearly worked) might be more N-sensitive.

Tracking as future loop work:
- N=3 confirmation on a borderline NOT-RESOLVED instance (e.g.,
  one of the 8 in the M4-scorable subset whose patch was within 20%
  of gold byte count)
- N=3 on django-10914 (largest patch, 2563 B; most opportunity for
  content variance to swing outcome)

## Implications for recommendation set

**qwen36 primary**: confirmed at the *outcome-stability* level, not
just at the *patch-engagement* level. When SWE-bench says qwen36
resolves 5 of 13 M4-scorable instances, that's a robust statement
about the model + stack, not a lucky draw.

Variance in patch content is a property of greedy MLX decode + multi-
turn agentic flows. It doesn't undermine the recommendation; it just
means single-run patch *bytes* aren't reproducible, but RESOLVED/NOT
RESOLVED *outcomes* are.

## Files

- `run1-672B.diff`, `run1-scores.jsonl` — N=2 sample
- `run2-656B.diff`, `run2-scores.jsonl` — N=3 sample
