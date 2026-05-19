# gemma4 MoE on requests-1963 — wrote only tests, no actual fix

## Task #80 — small-codebase verification

After pylint-5859 RESOLVED at T=1800, looking for another
small-codebase data point. `requests` is one of the smallest Python
libraries in our M4-scorable set.

## Result: clean exit at 370s, 12.6 KB patch — ALL tests, NO fix

```
rc=0  elapsed=370s  diff=12591 B  tool_calls=7
  3 write, 2 read, 1 bash, 1 todowrite
```

The 12.6 KB patch consists of two new files:

```
diff --git a/repro_test.py b/repro_test.py   (128 lines, test fixture)
diff --git a/test_repro.py b/test_repro.py   (additional test)
```

**Zero modifications to `requests/sessions.py`** (where the actual
fix lives — and where qwen36's RESOLVED patch landed for related
instances). The model wrote reproductions and test fixtures but
never updated the source.

Score: **MODEL PATCH FAIL — patch_applied=false**. score_local
strips test-file diffs from model patches (per the canonical
SWE-bench harness convention). After stripping, the model patch
is EMPTY → "(model patch only touched test files; stripped)".

## A FOURTH gemma4 MoE failure mode

We now have four distinct behaviors across N=4 tests at T=1800:

| Instance | Codebase | Outcome | Behavior |
|---|---|---|---|
| pylint-5859 | small | **RESOLVED** | Full agentic: read → edit → done |
| django-11001 | moderate | Failed (0 B) | Conversational fallback: types code as text |
| matplotlib-18869 | large | Failed (0 B, TIMEOUT) | Slow decode: 0.7 tok/s, only 4 exploration calls |
| **requests-1963** | small | **Failed (12.6 KB but applied=false)** | Test-obsessed: writes only test files, never the source fix |

The 4 behaviors aren't a clean "codebase size" gradient — requests is
small (like pylint) but the model went down a test-writing rabbit
hole instead of finding the canonical fix.

## What this means for the recommendation

gemma4 MoE @ T=1800 is **1/4 verified RESOLVED** (25 %). Two of the
four failures (django, requests) had the model fully engaging with
tools — it just steered into the wrong work (chat-text in one case,
test-only files in the other). The matplotlib failure was purely
infrastructure (slow decode at 28K context).

**The reliability is too low to recommend gemma4 MoE as a general
agentic option.** Even small-codebase work isn't a guaranteed win
(requests was small and failed). The pylint success was a real model
capability, but doesn't extrapolate.

**Updated recommendation**: gemma4 MoE @ T=1800 produces RESOLVED
patches some of the time (1/4 verified). Use only when you
specifically want a Gemma-family model and can tolerate ~75 % "wrong
work" rate. Otherwise, qwen36 remains the only reliable agentic
primary.

## Why qwen36 succeeded where gemma4 didn't

qwen36 RESOLVED 5/13 M4-scorable instances. For the same instances:

- pylint-5859: qwen36 RESOLVED (656 B targeted fix to misc.py)
- django-11001: qwen36 RESOLVED (1368 B)
- requests-1963: qwen36 6/7 F2P + 1 P2P regression (close but not resolved)
- matplotlib-18869: qwen36 RESOLVED on a DIFFERENT instance
  (matplotlib-18869 itself was not in the N=21 scored set; qwen36's
  full coverage was the harder-ecosystems sweep)

qwen36's MoE+DeltaNet decode is fast enough at all context sizes to
keep the agent loop alive, AND its chat template stays on-task. Both
properties seem load-bearing.

## Files

- `run.log` — smoke.sh stdout (rc=0)
- `predictions.jsonl` — the 12.6 KB record (entirely test files)
- `psf__requests-1963.diff` — the diff (2 test files, no source)
- `psf__requests-1963.log` — opencode session (7 tool calls)
- `psf__requests-1963.env.log` — venv (requests installed cleanly)
- `score.jsonl` — applied=false (test-strip removes everything)
