# qwen35 with TIMEOUT=1800 — capable, but 15× slower than qwen36

Followup to the 2026-05-18 4-pick scorecard, where qwen35 made 6 tool
calls and identified the correct bug line in its final text turn but
ran out of 600 s budget before reaching `edit`. Question: does qwen35
actually solve when given more time?

**Answer: yes.** Given 1800 s budget, qwen35 produced the **identical
506-byte patch** as qwen36 on `astropy__astropy-12907`:

```diff
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

| Metric | qwen35 (1800s) | qwen36 (600s) |
|--------|:--------------:|:-------------:|
| Wall time | 1810 s (hit timeout, rc=124) | 122 s |
| Tool calls | 11 (5R+2bash+2g+1grep+1edit) | 6 (3R+2g+1edit) |
| Diff bytes | 506 (same patch) | 506 (same patch) |
| Patch correct | ✓ same canonical fix | ✓ |

**qwen35 is ~15× slower** for the same output. Practical recommendation
stays at qwen36 for agentic coding. qwen35's static benchmarks (MMLU 90 /
HE 100 / Needle 100) lead the M4 set; it just can't make decisions fast
enough through the opencode agent loop.

The DeltaNet 27B dense decode is the bottleneck. qwen36 (35B-A3B MoE +
DeltaNet) has 3B active params and decodes much faster while still
benefiting from the linear-attention layers at long context.

## Why this matters

The original scorecard verdict "qwen35: RIGHT BUG, TOO SLOW" turned out
to be exactly right — it WAS the right bug, and 600s WAS too slow. With
3× the budget, qwen35 reaches the same answer. That validates the
scorecard's methodology: tool-call telemetry + final-text content
matters more than just diff-byte count when ranking models.

For workflows where wall time is irrelevant (overnight runs, batch
research), qwen35 is an equivalent agentic-coding option to qwen36 on
this instance. For interactive use, qwen36's 15× speedup is decisive.
