# May-18 RESOLVED set rescoring — 5/5 still RESOLVED, scorer stable

## Context

The previous loop iteration (commit `a6d8934`) found run-to-run variance
in qwen36 on requests-1963 and flagged a worry: "score_local.py bug
fixes since May 18 may have exposed false positives — the 5/13 = 38.5%
M4-scorable headline needs re-audit."

This iteration tests that worry directly: re-score the 5 May-18 RESOLVED
predictions against today's `score_local.py`. If scoring drifted (e.g.,
became more strict), fewer than 5 would resolve today. If stable, all 5
still resolve.

## Result: 5/5 still RESOLVED (no scorer drift)

```
[1/5] django__django-10914     ✓ resolved=True  f2p=1/1   p2p=98/98   (14.0s)
[2/5] django__django-11001     ✓ resolved=True  f2p=2/2   p2p=118/118 (3.4s)
[3/5] django__django-11039     ✓ resolved=True  f2p=1/1   p2p=88/88   (3.7s)
[4/5] pylint-dev__pylint-5859  ✓ resolved=True  f2p=1/1   p2p=10/10   (7.8s)
[5/5] sphinx-doc__sphinx-10325 ✓ resolved=True  f2p=1/1   p2p=5/5     (18.0s)
=== princeton-nlp/SWE-bench_Lite resolved=5/5 (100.0%) ===
```

Identical bytes, identical scorer outcome. The `5/13 = 38.5% M4-scorable`
headline holds.

## What this clears up

In the previous iteration I noted that today's rescoring of the canonical
521 B requests-1963 fix scored NOT RESOLVED (F2P 6/7) where May 18's
README implied it had been RESOLVED. I attributed this to scorer drift
exposing false positives.

That was a misread. Going back to the May-18 archive
(`qwen36-score-local-2026-05-18/scores.jsonl`):

```
{"instance_id": "psf__requests-1963", "resolved": false,
 "patch_applied": true, "f2p_passed": 6, "f2p_total": 7,
 "p2p_passed": 111, "p2p_total": 112, "score_seconds": 37.0}
```

requests-1963 was always F2P 6/7, NOT RESOLVED on May 18. The
`qwen36-perinst-missing-ecosystems-2026-05-18/README.md`'s "SUCCESS ✓"
mark was a patch-size proximity inference ("521 B vs gold 566 B — within
8%"), NOT an actual score. The two markers got conflated in my
analysis.

## Net picture across the last three iterations

| Claim | Status |
|---|---|
| qwen36 produces bit-identical 506 B patches across stack changes (astropy-12907) | ✅ confirmed (commit 35b1187) |
| qwen36 same instance produced 832 B vs prior 521 B → stack regression | ❌ wrong; was N=1 variance (corrected at a6d8934) |
| Stack change exposed scorer false positives (5/13 needs re-audit) | ❌ wrong; scorer is stable (corrected here) |
| qwen36 has substantial run-to-run variance in multi-turn agentic flows | ✅ confirmed (3 distinct outputs in 4 samples) |
| The 5/13 = 38.5% M4-scorable RESOLVED rate is reliable | ✅ confirmed (5/5 still resolve on today's scorer) |

## Implications for recommendations

- **qwen36 primary**: unchanged. Patches the model produces are stable
  across re-scoring; variance is between *which* patch the model
  produces, not whether the same patch passes tests.
- **5/13 headline holds**: no re-audit needed; today's `score_local.py`
  is consistent with May-18's on the same predictions.
- **Variance still matters**: qwen36's outputs vary by ~50-150%
  byte-count across reruns on the same instance + same stack. For
  borderline cases (instances that scored RESOLVED with N=1) the run-
  to-run distribution may include NOT-RESOLVED outliers. Re-running
  some of the 5 with N=3 would tell us if the 5/13 is a modal estimate
  or a sample-favorable estimate.

## Next steps tracked

1. **N=3 rerun of one of the 5 RESOLVED instances** (cheapest:
   pylint-5859 or django-10914 because they have short test suites).
   Tests whether the RESOLVED outcome is stable across N=3 or just a
   favorable sample.
2. **Stabilize qwen36**: investigate chunked-prefill scheduling as a
   variance source. `--chunked-prefill-size 2048` may produce different
   KV pool eviction order; disabling chunked prefill (or setting
   `--chunked-prefill-size 999999999`) may reduce variance.

## Files

- `predictions.jsonl` — the 5 May-18 RESOLVED predictions (verbatim)
- `scores.jsonl` — today's scoring results (5/5 RESOLVED)
