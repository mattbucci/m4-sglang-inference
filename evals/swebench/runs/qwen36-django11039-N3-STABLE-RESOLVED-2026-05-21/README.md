# qwen36 django-11039 N=3 — RESOLVED stable, variance is comment prose

## Context

Third N=3 stability test, extending coverage of the May-18 RESOLVED
set from 2/5 to 3/5 confirmed.

## N=3 results — all RESOLVED, 2 distinct outputs

| N | When | Bytes | SHA256 | Verdict |
|:-:|---|---:|---|---|
| 1 | May 18 (v0.5.11) | 826 | 3ed39b4d… | RESOLVED ✓ |
| 2 | Today run #1 (v0.5.12 per-feature) | 812 | 5137836c… | RESOLVED ✓ |
| 3 | Today run #2 (v0.5.12 per-feature) | 812 | 5137836c… | RESOLVED ✓ (BIT-IDENTICAL to run #1) |

Today's two runs produced **byte-identical** output. The May-18
baseline differed by 14 bytes.

## The variance: just comment prose

```diff
- # Show begin/end around output only for atomic migrations on databases
- # that support transactional DDL.
+ # Show begin/end around output only for atomic migrations that support
+ # rolling back DDL.
```

Same hunk, same code change (`if migration.atomic and connection.features.can_rollback_ddl:`),
same f2p/p2p outcome. Only the model's English explanation differs.
Both are valid descriptions of the same conditional.

## Updated variance taxonomy across 4 instances

| Instance | RESOLVED? | N | Distinct outputs | Variance shape |
|---|---|:-:|---|---|
| pylint-5859 | YES | 3 | 3 | regex-anchor syntax (`\W|$` vs lookaheads) |
| django-10914 | YES | 3 | 3 | scope of inert doc edits |
| **django-11039** | **YES** | **3** | **2** | **comment prose (no code diff)** |
| requests-1963 | NO | 4 | 3 | conceptually distinct fix strategies |

This is the smallest variance shape observed yet — just English
phrasing of a comment. The model is so confident on this specific
fix that it almost converges on bit-identical output across the
v0.5.12 stack.

## Within-stack determinism is real for some instances

For django-11039, today's two runs are byte-identical (5137836c…).
For pylint-5859, today's two runs were *different* from each other.
For django-10914, today's two runs were *different* from each other.

So within-stack determinism is instance-dependent — some agentic
flows are tight enough that the model converges on a single output
across multiple runs; others have enough tool-call variance to
produce distinct surface forms.

## Tally

Three of five May-18 RESOLVED instances confirmed N=3 outcome-stable:

- ✅ pylint-5859 — 3/3 RESOLVED, 3 distinct outputs
- ✅ django-10914 — 3/3 RESOLVED, 3 distinct outputs
- ✅ django-11039 — 3/3 RESOLVED, 2 distinct outputs (today's runs identical)
- ⏳ django-11001 — pending (772 s May-18 wall → ~1500-2000 s today)
- ⏳ sphinx-10325 — pending (725 s May-18 wall → ~1500 s today)

Each subsequent confirmation strengthens the recommendation that
qwen36's 5/13 = 38.5% M4-scorable RESOLVED rate is robust.

## Files

- `run1-812B.diff`, `run1-scores.jsonl` — first N=2 sample
- `run2-812B.diff`, `run2-scores.jsonl` — second N=2 sample (bit-identical to run #1)
