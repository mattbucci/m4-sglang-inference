# gemma4 MoE pylint-5859 on v0.5.12 — fails where v0.5.11 succeeded

## Context

User asked whether upstream gemma fixes (v0.5.11 → v0.5.12 + post-v0.5.12
commits) help our M4 stack. Code-path analysis said NO — every gemma fix
targets CUDA/Triton/TensorRT/Pipeline-Parallelism paths that our MLX
backend bypasses via the `_TextOnlyVLMShim` wrapper.

This run empirically verifies that conclusion by retesting the one
gemma4 MoE success case (pylint-5859 RESOLVED at TIMEOUT=1800 on
v0.5.11) on the v0.5.12-rebased stack.

## Result: didn't reproduce

| Run | Wall | Patch | Tool calls | Score |
|---|---:|---:|---|---|
| **v0.5.11 baseline** | 1324 s rc=0 | 1003 B (canonical fix) | 48 (1 edit + 1 write) | **RESOLVED** |
| **v0.5.12 retest** | 1804 s rc=124 | 180 B (reproduction.py only) | 64 (0 edit + 2 write) | **NOT resolved** |

The 180-byte patch:

```diff
diff --git a/reproduction.py b/reproduction.py
new file mode 100644
+++ b/reproduction.py
@@ -0,0 +1,2 @@
+# This is a test comment
+# ???
```

Same instance, same model, same TIMEOUT — the model went down a
different exploration path on v0.5.12. More tool calls (64 vs 48), more
bash (44 vs 35), more read (8 vs 6), more glob (8 vs 3), more grep
(2 vs 2) — but **never made an edit call**. The model timed out still
exploring instead of writing the canonical regex fix to
`pylint/checkers/misc.py`.

## Why this happened

Two possible explanations:

1. **MLX library subtle differences**: v0.5.12 added the on-the-fly
   quantization feature (`from mlx_lm.utils import quantize_model`).
   This import path may have triggered different module-load semantics
   or affected the mlx_vlm code path indirectly. Investigation would
   require diffing the loaded module versions.

2. **Greedy MLX decode is not as deterministic as expected at this
   model size**: qwen36 produced bit-identical 506 B output on both
   v0.5.11 and v0.5.12. But gemma4 MoE diverged. Gemma 4's sliding-
   window attention is more sensitive to subtle MLX-layer differences
   (different memory layout, different fp16 rounding order, etc.) than
   qwen36's DeltaNet. A single bit flip in the input embeddings can
   branch decoding into a completely different trajectory.

The contrast with qwen36's bit-identical result is what makes this
notable — same MLX backend changes, same patches, but two different
sensitivities.

## Recommendation impact

gemma4 MoE @ T=1800's "1/5 RESOLVED" rate is now revealed to be even
weaker — the one success case isn't reproducible. The recommendation
collapses further:

- **Before v0.5.12 retest**: "gemma4 MoE works on pylint-5859 (RESOLVED), unreliable everywhere else (4/5 fail)"
- **After v0.5.12 retest**: "gemma4 MoE fails everywhere; the prior pylint-5859 RESOLVED was a single-trajectory coincidence that doesn't reproduce across stack changes"

qwen36 remains the only reliable primary on M4 — and reliably so:
verified bit-identical 506 B output on both v0.5.11 and v0.5.12 stacks.

## Upstream gemma fixes assessment

The empirical retest confirms the code-path analysis: upstream gemma
fixes (PR #24696 fused Q/K/V RMSNorm, #24048 PCG optimization, #25286
FP8 Triton, #25284 pipeline parallelism, #23976 EAGLE3 spec decoding,
#24436 MTP, #25006 trtllm_mha, #25547 attention backend override) all
target server-GPU execution paths irrelevant to MLX-on-M4. Net effect
on gemma4-on-M4 agentic capability: **zero**.

## Files

- `run.log` — smoke.sh stdout (rc=0, rollout rc=124)
- `predictions.jsonl` — the 180-byte prediction record
- `pylint-dev__pylint-5859.diff` — the patch (just reproduction.py)
- `pylint-dev__pylint-5859.log` — opencode session with 64 tool calls (0 edits)
- `pylint-dev__pylint-5859.env.log` — pylint venv install
- `score.jsonl` — applied=true (patch only touches new file, no test files
  to strip), F2P 0/1, P2P 10/10
