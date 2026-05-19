# qwen35-9b-8bit agentic smoke — engages but can't synthesize

## Why this test

Last opencode-configured model still untested in the agentic loop.
qwen35-9b-8bit is the smaller (9B) higher-precision (8-bit) variant of
qwen35. ~10 GB resident. Theoretically a "memory-constrained agentic"
candidate if it works.

## Result

```
rc=124 (TIMEOUT)  elapsed=904s  diff=EMPTY (0 bytes)  tool_calls=13
  11 bash
   2 read
   0 edit  ← key failure
```

The model **engaged with the agent loop** (13 tool calls in 15 min,
much more activity than coder-30b's 1 glob or devstral's 0 calls) but
ran 11 bash and 2 read calls without ever using `edit`. Exploratory
paralysis: extensive investigation, no synthesis decision.

## Comparison with qwen35 family on the same instance

| Model | Wall | Tool calls | Patch | Verdict |
|---|---:|---:|---:|---|
| `qwen36` (35B-A3B MoE) | 125 s | 6 incl. 1 edit | 506 B ✓ | Best — fast + decisive |
| `qwen35` (27B Dense, T=1800) | 1810 s | 11 incl. 1 edit | 506 B ✓ | Same patch, 15× wall |
| `qwen35-9b-8bit` (9B, T=900) | 904 s | 13 incl. 0 edit | 0 B ✗ | **Explores but can't synthesize** |

The 9B-parameter ceiling is the issue here, not quantization (this is
8-bit, the higher-precision variant) or decode rate (9B is faster than
27B). The smaller model has the tool-use loop down but lacks the
synthesis capability to decide when its exploration is "enough" to
write a fix.

## Recommendation impact

**No change**: qwen35-9b-8bit doesn't add agentic value. Stick with
qwen36 (best in class) or qwen35 (if 27B-DeltaNet specifically
needed). Smaller-model users without a 30+ GB-resident option don't
have a working agentic pick on M4.

This closes the opencode-configured int4/8-bit set: 11 of 11
alternatives to qwen36 are now ruled out with direct evidence (no
gating artifacts).

## Files

- `run.log` — smoke.sh stdout
- `predictions.jsonl` — empty-patch record at 904 s timeout
- `astropy__astropy-12907.log` — opencode session showing 13 tool calls without edit
- `astropy__astropy-12907.env.log` — astropy venv install fail (irrelevant — fell back to no-venv)
