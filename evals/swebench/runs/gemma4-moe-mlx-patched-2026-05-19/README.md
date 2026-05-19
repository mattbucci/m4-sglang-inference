# gemma4 (26B-A4B MoE) after MLX patches — engages with agent loop now

## Task #75 — second MoE-rescue MLX patch

In the earlier `gemma4-moe-int4-2026-05-18` run, SGLang crashed ~22 s
into the first opencode chat completion:

```
AttributeError: 'RotatingKVCache' object has no attribute 'write_token'
File ".../mlx/kv_cache/attention_wrapper.py", line 186, in _batched_decode
```

Gemma 4's heterogeneous attention layout mixes ContiguousKVCache (full
attention) with RotatingKVCache (sliding-window) per layer. The MLX
batched-decode wrapper assumed the ContiguousKVCache API
(`write_token` + `get_kv`) which doesn't exist on RotatingKVCache.

## Patches landed today (patch 020)

Two interacting fixes were needed:

### 1. `attention_wrapper.py` — use `update_and_fetch`

Replaced `write_token(k, v)` + `get_kv()` with the single canonical
`update_and_fetch(keys, values)` call. This API exists on every
mlx_lm cache type (ContiguousKVCache, RotatingKVCache, MambaCache,
ArraysCache, etc.) and returns the full valid K/V slice — same
semantics for ContiguousKVCache, the rotated window for RotatingKVCache.

### 2. `model_runner.py` — non-hybrid path serial fallback

After fix #1, gemma4 hit a NEW crash:

```
ValueError: [concatenate] dimensions must match... shapes
(1,8,7426,256), (1,8,1024,256)
```

The non-hybrid batched path's `pad_sizes` math derives a single
`seq_lens` per request from `caches[i][0].offset`, then assumes ALL
layers' K/V have that length after `update_and_fetch`. RotatingKVCache
returns at most `max_size` (1024 for Gemma 4 sliding-window), so per-
request K/V across layers have different shapes and `mx.concatenate`
ValueErrors at the batch step.

Fix: detect `RotatingKVCache` in `caches[0]` BEFORE entering the
batched path and fall back to serial-per-request decode. Same pattern
as the hybrid-path partial-cache fallback from patch 019.

## Result: model engages, but doesn't synthesize

```
rc=0  elapsed=108 s  diff=EMPTY (0 bytes)  tool_calls=4
  1 glob, 1 read, 2 bash
opencode step_finishes: 4× reason=tool-calls, 2× reason=stop
```

**This is qualitatively different from prior gemma4 behavior** — the
chat-template DOES engage with opencode's tool-call protocol on the
MoE variant (gemma4-31b Dense still emits 0 tokens). gemma4 MoE made
4 tool calls (glob → read → 2× bash) over 108 s, then stopped without
making an `edit` call. Same failure class as `qwen35-9b-8bit`:
"engages with the loop but can't synthesize an edit decision."

| Metric | Pre-patch | Post-patch |
|---|---|---|
| Server crash | YES (~22s in) | NO ✓ |
| First decode succeeds | NO | YES ✓ |
| Tool calls | 0 (crash before any) | 4 |
| Edit decisions | 0 | 0 |
| Final patch | empty | empty |
| Failure class | MLX backend bug | Model synthesis ceiling |

## Recommendation impact

**No change to primary**: qwen36 stays primary; gemma4 MoE doesn't
produce patches.

**But the failure-mode UPGRADE matters** for the broader narrative:
- Gemma 4 family is no longer stack-blocked on M4
- Future Gemma 4 variants (with better agentic training) would now
  work through the harness without additional patches
- The "Gemma 4 MoE: gives up after 4 tool calls" diagnosis is real
  data, not inferred from a backend crash

## Files

- `run.log` — smoke.sh stdout
- `predictions.jsonl` — empty-patch record
- `astropy__astropy-12907.log` — opencode session with 4 tool calls + step_finishes
- `astropy__astropy-12907.env.log` — astropy venv install failure (irrelevant)
