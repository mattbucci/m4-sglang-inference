# qwen36 django-10914 N=3 — RESOLVED stable, variance is in doc-edit scope

## Context

Second N=3 stability test (after `../qwen36-pylint5859-N3-STABLE-RESOLVED-2026-05-21/`).
Pick: django-10914 — largest May-18 RESOLVED patch (2563 B, 2 files
touched), most opportunity for content variance to swing outcome.

## N=3 results — all RESOLVED, three distinct strategies

| N | When | Bytes | SHA256 | Files touched | Verdict |
|:-:|---|---:|---|---|---|
| 1 | May 18 (v0.5.11) | 2563 | 87fbe43e… | 2: global_settings.py + ref/settings.txt | RESOLVED ✓ |
| 2 | Today run #1 (v0.5.12 per-feature) | 3128 | dfb4943b… | 3: + howto/deployment/checklist.txt | RESOLVED ✓ |
| 3 | Today run #2 (v0.5.12 per-feature) | 3081 | 1a32b9c8… | 4+: + releases/2.2.txt | RESOLVED ✓ |

All three include the load-bearing one-line code change:

```python
# django/conf/global_settings.py
-FILE_UPLOAD_PERMISSIONS = None
+FILE_UPLOAD_PERMISSIONS = 0o644
```

The variance is in how many *documentation* files the model touches in
addition: May-18's model edited 1 doc file, today's #1 edited 2, #2 edited
3+. The F2P test only validates the code change, so the extra doc edits
are functionally inert — all three RESOLVE with F2P 1/1 and P2P 98/98.

## Variance taxonomy across instances

| Instance | Variance shape |
|---|---|
| pylint-5859 (May 18 RESOLVED) | Same fix at same site; different regex anchor syntax (`\W|$` vs negative lookahead vs positive lookahead) |
| django-10914 (May 18 RESOLVED) | Same code fix; different scope of related doc edits |
| requests-1963 (May 18 NOT RESOLVED) | Conceptually distinct fix strategies (canonical vs history-list vs prev_prepared_request) |

For RESOLVED cases, variance is in **cosmetic surface form** or **scope
of inert collateral edits** — the load-bearing fix is invariant. For NOT
RESOLVED cases, variance is in **conceptual approach** — the model tries
different strategies, none of which pass the test suite.

This matches the recommendation: qwen36 reliably finds the right fix
space for RESOLVED instances and reliably misses on NOT RESOLVED ones.
The 5/13 = 38.5% M4-scorable rate is a property of the model + dataset,
not a property of which random seed got drawn.

## Performance footnote (separate from outcome question)

Today's rollouts took 2-3× longer than May-18's:

| N | Wall | TIMEOUT | rc |
|:-:|---:|---:|:-:|
| May 18 | 328 s | (unknown) | 0 |
| Today #1 | 603 s | 600 s | **124** (timeout) |
| Today #2 | 757 s | 900 s | 0 |

The slowdown is consistent with the model producing more elaborate
patches (more doc files = more tool calls = more decode time). Worth
investigating if this generalizes — possible v0.5.12 chunked-prefill
regression at long contexts, or just consequence of the variance
sampling more verbose paths today. Either way, the OUTCOME is unchanged.

## Tally so far

Two of five May-18 RESOLVED instances confirmed N≥3 outcome-stable:

- ✅ pylint-5859 — 3/3 RESOLVED
- ✅ django-10914 — 3/3 RESOLVED
- ⏳ django-11001 — pending (772 s May-18 wall — expect ~1500-2000 s today)
- ⏳ django-11039 — pending (349 s May-18 wall — expect ~700-1000 s today)
- ⏳ sphinx-10325 — pending (725 s May-18 wall — expect ~1500-2000 s today)

The three pending instances would need TIMEOUT≥1800 and 2-3 rollouts
each. They're tractable but not in a single 30-min loop tick. Future
work.

## Files

- `run1-3128B.diff`, `run1-scores.jsonl` — N=2 sample (rc=124 timeout)
- `run2-3081B.diff`, `run2-scores.jsonl` — N=3 sample
