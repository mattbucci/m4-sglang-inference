# gemma4 MoE on sphinx-10325 — gave up early (5/5 boundary mapping)

## Task #82 — final gemma4 boundary mapping

Five-instance characterization of gemma4 MoE @ T=1800 after patch 020:

| Instance | Codebase | Outcome | Tool calls | Wall |
|---|---|---|---|---|
| pylint-5859 | small | **RESOLVED** | 48 (1 edit + 1 write) | 1324s |
| django-11001 | moderate (19K) | Conversational fallback | 4 (all bash) | 123s |
| matplotlib-18869 | large (28K) | Slow decode TIMEOUT | 4 (3 glob + 1 read) | 1825s |
| requests-1963 | small | Test-obsessed (12.6KB but only test files) | 7 (3 write) | 370s |
| **sphinx-10325** | moderate | **Gave up early** | 5 (3 bash + 1 glob + 1 grep) | 152s |

## Result

```
rc=0  elapsed=152s  diff=EMPTY (0 bytes)  tool_calls=5
  3 bash, 1 glob, 1 grep
```

Same "gives up early without making an edit" pattern as django-11001.
Model decided after 5 exploration calls that it was done.

Score: applied=False (empty diff).

## Final gemma4 MoE @ T=1800 verdict: 1/5 RESOLVED

**The "small-codebase only" hypothesis is disproved.** Failure modes
are not cleanly correlated with codebase size:

- requests (small): FAILED with test-only patch
- sphinx (moderate): FAILED with early give-up
- pylint (small): RESOLVED

So gemma4 MoE @ T=1800 isn't reliably useful even on simple
codebases. The pylint-5859 RESOLVE remains a real proof-point
(model has the capability) but doesn't extrapolate.

**Recommendation collapse**: gemma4 MoE is **verified RESOLVED on
pylint-5859 only.** Use only if you specifically want Gemma-family
agentic, accept ~80% non-resolution rate, and ideally pick instances
similar to pylint-5859 (small Python library with focused regex/
logic bug).

qwen36 remains the only reliable cross-instance primary.

## Files

- `run.log` — smoke.sh stdout
- `predictions.jsonl` — empty-patch record
- `sphinx-doc__sphinx-10325.log` — opencode session (5 calls, no edits, reason=stop)
- `sphinx-doc__sphinx-10325.env.log` — venv setup OK (3 pkgs)
- `score.jsonl` — applied=false (empty diff)
