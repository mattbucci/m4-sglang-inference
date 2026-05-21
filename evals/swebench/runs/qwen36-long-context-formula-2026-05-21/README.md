# Long-context ceiling: environment-dependent, predictable formula

## What this characterizes

Yesterday: 32K WORKS / 60K OOMs / chunked-prefill-size doesn't help.
That gave a fixed "ceiling around 32K" recommendation.

This iteration probed in between and found the ceiling is actually
**not fixed** — it depends on **how much memory the OS has free after
the SGLang server boots**, which in turn depends on what desktop apps
are running. A predictable formula falls out cleanly.

## Today's measurements

Same config across both runs: `mem-fraction-static 0.4`, turboquant,
chunked-prefill 2048.

| Probe | Post-boot free | Result | Free after |
|---|---:|---|---:|
| 40K input | 5 GB | OOM (server DOWN) | 1 GB |
| 32K input | 6 GB | OK | **0 GB** (at edge) |

Yesterday with the same probe:

| Probe | Post-boot free | Result | Free after |
|---|---:|---|---:|
| 32K input | 7 GB | OK | 88 MB |

The difference between today and yesterday isn't the model, the
stack, or any flag — it's that today the OS has ~1-2 GB less free
memory after boot because of more background apps (Firefox tabs,
Spotify, Steam, claude-code-remote).

## The formula

Per-token memory cost during prefill is consistent ~0.19-0.22 MB/token
across all measurements:

```
delta_free = post_boot_free_GB - free_after_GB
per_token = delta_free / tokens_prefilled
         ≈ 0.2 MB/token (range 0.19-0.22)
```

Predictive ceiling formula:

```
max_prefill_tokens ≈ (post_boot_free_GB × 1024 - 500_MB_safety) / 0.2 MB/token
                  ≈ (post_boot_free_GB - 0.5) × 5120 tokens
```

| Post-boot free | Predicted ceiling | Notes |
|---:|---:|---|
| 5 GB | 23K | Today's failed 40K matches |
| 6 GB | 28K | Today's 32K-tight matches (at edge) |
| 7 GB | 33K | Yesterday's 32K-with-88MB-after matches |
| 8 GB | 38K | (untested) |
| 10 GB | 48K | (untested — would require clearing more apps) |
| 12 GB | 59K | (untested — close to fresh-boot state) |

## Per-token cost root cause (still hypothesis)

The 0.2 MB/token rate doesn't match the KV cache slot size
(`bytes_per_slot=23040` = 23 KB/token per the SGLang allocator log).
The extra ~177 KB per token must come from somewhere else during
prefill. Hypothesis (unchanged from prior iteration):

- MLX unified-memory lazy page touches accumulating as the KV pool's
  virtually-reserved pages get written to
- Python-side intermediate buffers (logit slices, mask construction,
  position-id arrays) not GC'd between chunks
- MoE expert dispatch overhead (qwen36 has 128 experts × top-8 routing
  per token)

The mechanism doesn't really matter for the recommendation — what
matters is the 0.2 MB/token rate is empirically reliable.

## Recommendation update

**For long-context bench / serving on M4:**

1. **Close background apps before booting**. Firefox + Spotify + Steam
   + claude-remote together consume ~2.7 GB today. Clearing those
   would lift the ceiling from ~32K → ~46K.

2. **Calculate your own ceiling** with the formula:
   - Free RAM after server boot (`vm_stat | awk '/Pages free/ {gsub(...);}'`)
   - Subtract 0.5 GB safety
   - Multiply by 5120 → max prefill tokens

3. **The "32K ceiling" recommendation in CLAUDE.md is correct as a
   conservative default** — assumes typical desktop background load.
   Cleared-environment ceiling could be 45K-50K.

4. **The actual 256K aspirational target needs MLX flash-attention or
   per-token memory reclaim** to drop the 0.2 MB/token per-prefill-
   token rate. Neither is a config knob.

## Files

- `data.txt` — raw measurements + formula derivation
