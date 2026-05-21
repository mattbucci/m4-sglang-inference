# qwen36 requests-1963 on v0.5.12 per-feature — REGRESSED (vs May 18 baseline)

## Result

| | May 18 (v0.5.11 + cumulative-or-individual patches) | Today (v0.5.12 + per-feature patches) |
|---|---|---|
| Patch bytes | 521 | **832** |
| Wall | 234 s | 477 s |
| Tool calls | 10 (1 edit) | (see logs) |
| F2P | (resolved) | **6/7** ✗ |
| P2P | (resolved) | **111/112** ✗ |
| Resolved | **YES** | **NO** |
| SHA256 | eef11314… | 5b51ab4a… |

The model produced a *different* fix to the same bug. Both target the same
file (`requests/sessions.py`) and the same method
(`SessionRedirectMixin.resolve_redirects`). May 18's fix was the canonical
2-line variant; today's fix is a 4-hunk variant that introduces a `history`
list and tracks redirect chain explicitly. The new fix fails 1 F2P test
and regresses 1 P2P test.

## Why this matters

This contradicts the just-claimed "qwen36 produces bit-identical patches
across stack changes" from
`evals/swebench/runs/qwen36-perfeat-regression-PASS-2026-05-20/`. That claim
was based on astropy-12907 only. Now we have a second data point that
shows the opposite for requests-1963.

**Refined claim** — see also `memory/project_m4_agentic_model_boundaries.md`:

> Greedy MLX decode is deterministic at the token level (proven by
> astropy bit-identity across three stack versions). BUT multi-turn
> agentic flows have implicit non-determinism from tool-call ordering,
> filesystem state, prefill scheduling, and dataset cache state.
> Stack changes can shift this enough to produce qualitatively
> different — and sometimes worse — patches.

This is a more honest framing of qwen36's behavior on M4. The PRIMARY
recommendation still stands (qwen36 produces a patch on this instance),
but the "scored resolved" verdict on any given instance can flip across
stack changes purely from tool-flow variance, not model regression.

## Implications for the recommendation set

- **qwen36 primary**: unchanged. Still produces patches.
- **Reproducibility caveat**: same instance, same model, same chat
  template can score differently on different stack versions. The 5/13
  M4-scorable subset number is a snapshot under one stack, not a stable
  invariant.
- **Test methodology**: a single regression run is insufficient to
  confirm a recommendation. Re-running N=3 across stack changes is
  more honest. The astropy-12907 bit-identity may be the exception
  (simpler bug, fewer tool calls, smaller tool-output variance) rather
  than the rule.

## What could be done

1. **Run requests-1963 N=3 on current stack** to test whether today's
   832B output is itself stable or whether each rerun produces something
   different. If stable → it's a stack-version-stable but worse fix.
   If unstable → there's real run-to-run non-determinism in the agentic
   flow that we should fix.
2. **Audit the chunked-prefill / scheduling diff** between v0.5.11 and
   v0.5.12 for code paths qwen36 exercises during agentic flows.
3. **Investigate whether opencode/dataset caches changed** between
   May 18 and today (huggingface dataset version, opencode binary).

Tracked as future loop iteration work.

## Files

- `psf__requests-1963.diff` — the 832 B new fix (4 hunks, history-list approach)
- `predictions.jsonl` — opencode rollout prediction record
- `scores.jsonl` — score_local result (resolved=False, 6/7 F2P, 111/112 P2P)
- `run.log` — smoke.sh stdout
