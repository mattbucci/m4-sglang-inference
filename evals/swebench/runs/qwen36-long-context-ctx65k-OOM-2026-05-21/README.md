# qwen36 long-context: dropping CTX to 65536 also OOMs at 60K input

## Hypothesis under test

Yesterday's #1 future idea: SGLang auto-sizes the KV pool based on
`--context-length`. Dropping CTX from 131072 → 65536 should halve
the upfront KV pool reservation and leave more headroom for prefill.

## Result

```
60K tokens   ERROR: Remote end closed connection without response
post-boot:   7 GB free
post-OOM:    1 GB free
```

Same OOM behavior as yesterday. Server died mid-prefill.

## What the allocator log told us

The smoking gun is in SGLang's startup log:

```
Wired memory limit set to 51.8 GB
Auto-sized KV pool (mode=turboquant):
    sys_available=33.41 GB
    mlx_limit=51.8 GB
    mlx_used=19.00 GB           (model weights — 35B at 4-bit)
    kv_budget=1.73 GB           (remaining for KV cache)
    bytes_per_slot=23040 (0.562 B/elem)
    pool_size=80838 tokens
max_total_num_tokens=80838
chunked_prefill_size=2048
context_len=65536
available_gpu_mem=32.80 GB
```

Key insights:

1. **Wired memory limit = 51.8 GB.** Not 64 GB × 0.4 = 25.6 GB as I'd
   expected. MLX's "wired memory" on macOS is a different beast than
   `mem_fraction_static` — the latter controls KV pool budget; the
   former controls how much physical RAM MLX can pin.
2. **kv_budget = 1.73 GB**, leading to pool_size = 80,838 tokens.
   That fits one 65K-token request (with 15K headroom). So the KV
   POOL sizing isn't the OOM cause.
3. **mlx_used = 19 GB** for model weights. **sys_available = 33.41 GB**
   (system free at boot time). 51.8 - 19 = 32.8 GB unused budget...
   but kv_budget only 1.73 GB? Must be that SGLang reserves the
   remaining ~31 GB for activation memory during prefill.

## So why does prefill still OOM?

If kv_budget=1.73 GB and pool fits 80K tokens, the OOM isn't in the
KV cache. It must be in **activation memory** during chunked prefill.

At chunked-prefill-size 2048, per-chunk activations for qwen36 (35B
MoE, 64 layers, hidden_dim ~6144, 32 KV heads × 128 head_dim) include:

- Q × K^T attention scores: chunk_size × full_context = 2048 × 65536 × 32 heads × 4 bytes ≈ 17 GB **per layer** (when materialized)
- MoE expert routing: 8 active experts × FFN ≈ few GB transient

If the implementation materializes the full attention matrix (no
FlashAttention), one layer's attention alone exceeds the ~31 GB
activation budget. Multi-layer is impossible.

This points to:

**The bottleneck is per-layer attention activation memory at long
context, not KV pool sizing.** Lowering CTX from 131072 to 65536 did
nothing useful because the per-chunk activation cost scales with the
*total context* (the K dimension in Q × K^T), not the CTX setting.

## Actionable

1. **Try `--chunked-prefill-size 1024`** to halve per-chunk activation:
   `2048 × 65K → 1024 × 65K` ≈ 8.5 GB per-layer attention. Still too
   big to fit one layer easily, let alone 64 layers, but might be
   enough at 32K context.

2. **The real fix would be a flash-attention-like fused attention
   kernel for MLX** that doesn't materialize the (chunk × context)
   matrix. That's a separate-month-of-work project, not a config
   tweak.

3. **Document the M4 long-context ceiling**: ~32K input is the
   practical limit for qwen36 (35B) on M4 today. Updating CLAUDE.md's
   "256K primary target" to "32K is what the hardware can actually
   do" would be honest. The 128K/256K aspiration needs flash-attention
   in MLX.

## What still works

- Decode at low context (≤16K): healthy ~60 tok/s
- Decode at higher context (16-32K): tested today, healthy
- Single-request prefill at ≤32K: works at mf=0.4

The agentic workload (SWE-bench Lite, 8-25K typical per turn) is well
under 32K. The recommendation set (qwen36 primary) is unaffected by
the long-context investigation.

## Files

- `sglang-allocator-log.txt` — the auto-sizer report at boot
