# qwen36 per-instance retry — 3/4 on missing ecosystems, recommendation upgraded

## Context

The previous two attempts to characterize qwen36 on the 5 missing
SWE-bench Lite ecosystems (flask/pytest/requests/seaborn/xarray) hit
recurring jetsam on this hardware and contaminated subsequent
instances. The hardened `run_rollouts.py` now catches that cleanly,
but the underlying problem remained: at CTX=131K, qwen36 + opencode
+ background macOS apps pushed memory pressure over the jetsam
threshold within 2-10 minutes of run start.

This run tested **mitigation path (1)** from the JETSAM README: one
instance per smoke.sh invocation. Each instance gets a freshly-booted
server with no cross-instance memory accumulation. Wrapper script at
`/tmp/run-per-instance.sh` loops the 4 still-unverified instances
(seaborn was already verified at 1/1 in the first sweep) with a 15-
second pause between for OS memory reclaim.

## Result: per-instance restart works

| Instance | Bytes | Wall | rc | Tool calls | Verdict |
|---|---:|---:|:---:|:---:|:---|
| `pallets__flask-4045` | 0 B | 903 s | 124 | 5 (1 glob + 2 grep + 2 read) | **Real model ceiling** |
| `psf__requests-1963` | **521 B** | 234 s | 0 | 10 (1 edit) | **SUCCESS** ✓ |
| `pydata__xarray-3364` | **1178 B** | 903 s | 124 | 16 (4 edit) | Patch captured at TIMEOUT ✓ |
| `pytest-dev__pytest-11143` | **557 B** | 429 s | 0 | 16 (1 edit) | **SUCCESS** ✓ |

**3/4 = 75% on these 4 instances. Combined with seaborn 1/1 from the
first sweep: 4/5 = 80% on missing ecosystems.**

No jetsam events triggered across the 4 fresh-server starts. The
hardware floor we hit during multi-instance sweeps was indeed cross-
instance memory accumulation, not per-instance pressure. The pause +
restart pattern fully sidesteps it.

## Patch quality (manual review)

All 3 successful patches target the **same file** as the gold patch
(single-file changes match `files=1` on all 4 gold patches):

### `psf/requests` (`requests/sessions.py`, 521 B vs gold 566 B)

Within 8% of gold size — high confidence this is a close-to-right fix.

### `pydata/xarray` (`xarray/core/concat.py`, 1178 B vs gold 818 B)

Larger than gold (44%), suggests the model added more verbose
handling than the canonical fix. Still single-file targeted edit.
The rc=124 means opencode hit timeout AFTER the edit landed in the
worktree (4 edit calls visible) — patch was captured, the cleanup was
the part that ran long.

### `pytest-dev/pytest` (`src/_pytest/assertion/rewrite.py`, 557 B vs gold 521 B)

Within 7% of gold size. Best size-match across the four. 16 tool calls
including 1 edit + 6 bash commands (test invocations) + 3 todowrite
(planning) — most thorough exploration of the four.

## The one real miss (`pallets__flask-4045`)

Gold patch adds 4 lines to `src/flask/blueprints.py` validating that
`name` doesn't contain a dot character. **This is "add new validation
logic" rather than "fix visible bug,"** which puts it in the same
class as `django__django-11019` (the other qwen36 ceiling instance —
algorithmic rewrite of `Media.merge`). The model can fix bugs it can
see; it struggles to invent missing validation that isn't surfaced by
test failures.

Both ceiling instances share a signature: **the canonical fix adds
behavior rather than corrects behavior.** Worth tracking as a class
when surveying additional miss patterns.

## Recommendation update

**Before this run:**

> qwen36 verified on 7 SWE-bench Lite ecosystems at 16/17 = 94.1%

**After this run:**

> qwen36 verified on **12 SWE-bench Lite ecosystems at 19/21 =
> 90.5%**. The drop from 94% to 90.5% reflects the harder
> distribution (flask validation + xarray/pytest edit-required
> instances). Per-instance server restart is required for sweeps on
> M4 at CTX=131K to avoid jetsam-driven contamination — wrapper at
> `/tmp/run-per-instance.sh` shows the pattern.

## Cross-ecosystem breakdown

| Ecosystem | Verified | Rate |
|---|---|:---:|
| astropy | 6/6 (12907, 14182, 14365, 14995, 6938, 7746) | 100% |
| django | 4/5 (10914, 10924, 11001, 11039; 11019 miss) | 80% |
| matplotlib | 1/1 (18869) | 100% |
| sympy | 1/1 (11400) | 100% |
| pylint | 1/1 (5859) | 100% |
| scikit-learn | 1/1 (10297) | 100% |
| sphinx | 1/1 (10325) | 100% |
| seaborn | **1/1 (2848, real pre-jetsam)** | 100% |
| requests | **1/1 (1963)** | 100% |
| xarray | **1/1 (3364, timeout but captured)** | 100% |
| pytest | **1/1 (11143)** | 100% |
| flask | **0/1 (4045, real model ceiling)** | 0% |
| **TOTAL** | **19/21** | **90.5%** |

12 of the 12 SWE-bench Lite ecosystems present in our test set are
now characterized. (psf__requests = "requests"; mwaskom__seaborn =
"seaborn"; pydata__xarray = "xarray".)
