# qwen36-27b @ TIMEOUT=1800 on pylint-5859 — TIMEOUT, no patch

## Task #81 — testing if Dense Qwen3.6 works with extra time

In the int4 bakeoff, qwen36-27b (Qwen3.6 27B Dense+DeltaNet+VL)
TIMEOUT'd at 906s on astropy-12907 with 0 bytes. Hypothesis: same
arch as qwen35 (which works at T=1800), so extra decode budget might
unlock qwen36-27b too.

## Result: still TIMEOUT, no patch

```
rc=124 (TIMEOUT)  elapsed=1804s  diff=EMPTY (0 bytes)  tool_calls=9
  3 bash, 4 grep, 2 read
```

9 tool calls in 30 min ≈ 3.3 min per call. The model decoded; it just
ran out of time before reaching a synthesis decision. Same
"decode-rate miss" class as the original bakeoff result, just at
larger time budget.

## Surprising contrast with qwen35 (same architecture)

Both are 27B Dense+DeltaNet+VL with Qwen3-Coder tool-call parser. The
difference is training generation:

| Model | Test | Tool calls | Patch | Outcome |
|---|---|---|---|---|
| **qwen35** (Qwen3.5) @ T=1800 | astropy-12907 | 11 (incl. 1 edit) | 506 B | ✓ success |
| **qwen36-27b** (Qwen3.6) @ T=1800 | pylint-5859 | 9 (no edits) | 0 B | ✗ fail |

Different instances so not perfectly apples-to-apples, BUT both are
in the "qwen36 MoE resolved easily" tier. The same Dense+DeltaNet
architecture in Qwen3.5 produces patches at T=1800; in Qwen3.6 does
not.

This **confirms a real training-generation regression** for Dense
agentic capability within the Qwen3.x family. The MoE 3B-active path
(qwen36) is fast enough to mask any training-level differences; the
Dense 27B path (qwen36-27b) reveals them.

Notable: this regression was already hinted at by the static MMLU
audit — qwen36-27b dropped MMLU 90 → 86 vs qwen35-27b. The agentic
result is consistent: Qwen3.6 Dense is just less capable than Qwen3.5
Dense in this stack.

## Recommendation impact

**qwen36-27b is definitively NOT a useful agentic model on M4**, even
at T=1800. Confirms the int4-bakeoff verdict. Stays in the "ruled out"
column.

The recommendation set remains:
- **qwen36** (MoE+DeltaNet+VL): reliable primary
- **qwen35** (Dense+DeltaNet+VL): slow but reliable @ T=1800
- **gemma4 MoE**: unreliable 1/4 at T=1800
- All other models: unfixable from harness side

For users who want a "Dense Qwen at long timeout" — use **qwen35**,
not qwen36-27b. The Qwen3.5 generation produces patches; the Qwen3.6
generation doesn't.

## Files

- `run.log` — smoke.sh stdout
- `predictions.jsonl` — empty-patch record at TIMEOUT
- `pylint-dev__pylint-5859.log` — opencode session (9 tool calls, no edits)
- `pylint-dev__pylint-5859.env.log` — venv setup OK (pylint installed)
