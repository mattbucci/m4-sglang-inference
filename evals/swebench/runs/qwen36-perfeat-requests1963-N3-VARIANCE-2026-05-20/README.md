# qwen36 requests-1963 N=3 within-stack — substantial run-to-run variance

## Context

Last commit (`bfcc287`, "stack change degraded fix quality") claimed
v0.5.12 + per-feature patches regressed qwen36 on requests-1963 from
the May 18 canonical 521 B fix to a worse 832 B variant. That
conclusion was **wrong** — it was based on a single sample of an
unstable distribution. This iteration ran the same instance twice
more on the same stack and found three distinct outputs.

## N=4 sample distribution (all greedy MLX decode, qwen36)

| N | When | Bytes | SHA256 | Approach | Resolved |
|:-:|---|---:|---|---|:--:|
| 1 | May 18 (v0.5.11 stack) | 521 | eef11314… | Canonical: `prepared_request = req` + `prepared_request.copy()` | YES *(per old score_local; see caveat)* |
| 2 | Today rerun #1 (v0.5.12 per-feature) | 832 | 5b51ab4a… | `history` list, copy from `history[-1]` | NO (F2P 6/7, P2P 111/112) |
| 3 | Today rerun #2 (v0.5.12 per-feature) | 521 | eef11314… | Canonical (byte-identical to May 18) | NO (F2P 6/7, P2P 112/112) |
| 4 | Today rerun #3 (v0.5.12 per-feature) | 1264 | 7c6abc58… | `prev_prepared_request` + `method = prepared_request.method` | NO (F2P 6/7, P2P 110/112) |

Three distinct fix strategies in four samples. Notably, rerun #2's
521 B output is **byte-identical** to the May-18 baseline (same
SHA256), proving the v0.5.12 + per-feature stack CAN reproduce the
canonical fix — just not every time.

## What this means for prior commits

- **bfcc287's "stack change degraded fix quality" framing was wrong.**
  The 832 B was not a stack-induced regression; it was variance.
  Sampling N=1 on a system with substantial run-to-run variance gave
  a misleading picture.
- **Greedy MLX decode is still token-deterministic given fixed input.**
  The non-determinism is upstream of the model: tool-call ordering,
  prefill scheduling under chunked-prefill, dataset cache state,
  filesystem readdir order, network timing for HF dataset fetches.
  These feed back into the next prompt and the model decodes a
  different path through the agentic flow each time.

## Score discrepancy (May 18 RESOLVED → today NOT RESOLVED on identical bytes)

The May-18 README reported the 521 B output as RESOLVED. Today the
**bit-identical bytes** score F2P 6/7 / P2P 112/112 = NOT RESOLVED. One
F2P test fails on every sample regardless of patch approach.

Likely cause: between May 18 and today, `score_local.py` got 3
silent-misreport bug fixes (`memory/project_score_local_bug_class.md`
— "test-file overlap, tox 4.x runner, pytest -rA injection plus
aggregator scores.jsonl mis-load"). Memory notes "Each hid +1 qwen36
resolved" — but that wording implies the fixes catch FALSE NEGATIVES,
i.e. things that should have resolved but didn't. The opposite case
— FALSE POSITIVES that the bugfix exposed — was not noted but is
worth checking. If May-18's `psf__requests-1963 → RESOLVED` was a
false positive masked by the pytest -rA injection bug, then both:

- The "5/13 = 38.5% M4-scorable" headline is an over-count by an
  unknown amount
- The "patch-engagement 21/26 = 80.8%" is unchanged (engagement is
  judged on patch-size > 0, not on F2P pass)

## Recommendation refinements

1. **qwen36 primary**: still primary. It still produces patches.
   Variance does not make it unreliable; modal output is still the
   canonical fix on this instance.
2. **N=1 SWE-bench numbers are noise**. Any single-run "X resolved
   of Y" should be reported as a range, not a point estimate. For
   honest A/B between stacks, N≥3 per instance with stack pinning.
3. **Score_local re-audit**: re-score the May-18 5/13 set on today's
   score_local to find any other false-positives the bugfixes
   exposed. This could shrink the headline number or confirm it.
4. **Investigate determinism root cause**: chunked-prefill scheduling
   is a likely contributor. `--chunked-prefill-size 2048` may produce
   different KV pool eviction order across runs. Test with chunked
   prefill disabled or much larger chunks.

## Files

- `rerun2-521B.diff` — N=3 sample: canonical fix, byte-identical to May 18
- `rerun3-1264B.diff` — N=4 sample: third distinct strategy
- `rerun{2,3}-scores.jsonl` — local scoring results

See `../qwen36-perfeat-requests1963-REGRESSED-2026-05-20/` for the
N=2 (832 B) sample's full archive. That directory's conclusion
("stack regression") is **superseded** by this N=4 analysis.
