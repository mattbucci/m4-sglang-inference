# qwen36 at CTX=131072 — long-context claim validated

## Result

| Config | Wall | Diff | KV pool | mlx_used |
|--------|:----:|:----:|---------|----------|
| CTX=32768 (baseline) | 122 s | 506 B | (smaller) | (smaller) |
| **CTX=131072 (this run)** | **123 s** | **506 B** | **322,430 slots / 6.9 GB turboquant** | **19 GB** |

Same instance, same proxy, same model — **identical patch produced
in identical wall time** when context is sized 4× larger. The
README's "long-context flagship" claim is validated: DeltaNet
decode stays flat regardless of allocated context size.

## Why this matters

Agentic coding sessions accumulate context with each tool call:

  - astropy__astropy-12907 single-instance smoke: ~5K-token prompt
  - After 3-5 reads: 30-50K-token effective context
  - Multi-turn opencode sessions with retained history: 50-100K+

At our prior preset (CTX=32768) those longer sessions would hit
the context ceiling and either truncate or refuse. At CTX=131072
the model has 4× the headroom for tool-output accumulation without
paying a decode-speed penalty.

Memory budget at 131K turboquant:
  - mlx_used: 19 GB (weights + activation + KV pool)
  - sys_available: 33 GB (post-allocation system memory)
  - kv_budget: 6.92 GB (322K slots × 23 KB)

Plenty of headroom on the 64 GB M4 Pro. Could push to CTX=262144
with the same turboquant config but the marginal value is small
unless we're actually filling 256K of context.

## Operational recipe

```bash
CTX=131072 EXTRA_ARGS="--disable-radix-cache --kv-cache-dtype turboquant \
    --chunked-prefill-size 2048 --mem-fraction-static 0.5 \
    --enable-multimodal --tool-call-parser qwen3_coder" \
    bash scripts/launch.sh qwen36
```

Plus the no_thinking_proxy (started automatically by smoke.sh).
