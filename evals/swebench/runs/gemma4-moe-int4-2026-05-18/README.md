# gemma4 (26B-A4B MoE) agentic smoke — MLX RotatingKVCache bug, not model failure

## Goal

The smaller Gemma 4 variant (MoE not Dense) was excluded from every prior
bakeoff. Different architecture from gemma4-31b (which emits 0 tokens
under tool prompts). gemma4 uses `--tool-call-parser gemma4` per
launch.sh; might engage with opencode where the Dense variant doesn't.
This was Task #72 from the user-requested MoE rescue queue.

## Result: crashed at first decode (MLX backend bug)

```
[00:41:11] opencode session starts, sends first chat completion
[00:41:33] SGLang crashes with AttributeError, SIGQUIT received
[00:56:18] opencode wall TIMEOUT after 900s (server been dead for 15 min)
```

The crash trace:

```
File ".../components/sglang/python/sglang/srt/hardware_backend/
      mlx/kv_cache/attention_wrapper.py", line 186, in _batched_decode
    layer_caches[i].write_token(keys[i : i + 1], values[i : i + 1])
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'RotatingKVCache' object has no attribute 'write_token'
```

Gemma 4's heterogeneous attention layout uses **sliding-window
attention** for most layers, which is backed by mlx_lm's
`RotatingKVCache`. The MLX batched-decode wrapper at
`attention_wrapper.py:186` calls `write_token()` on the per-layer
cache — a method present on the standard caches but **not on
RotatingKVCache**. First chat completion that requires decode causes
the AttributeError; SGLang crashes; opencode hangs until smoke.sh's
SIGKILL.

## Result tag: stack-level MLX bug (same class as nemotron-omni)

| Metric | Value |
|---|---|
| Patch | 0 B (model never produced output) |
| Wall | 904 s (opencode timeout — server died at ~22 s in) |
| Tool calls | 0 |
| MLX server uptime before crash | ~22 s |
| Failure class | **MLX backend bug** (not model capability) |

The preflight DID pass (113 B content) — so a simple non-decode-batched
request worked. The crash happens specifically in batched-decode for
RotatingKVCache layers. This is patchable in principle.

## Recommendation impact

**No change to qwen36's primary status.** gemma4 MoE moves from
"untested" to "stack-level bug — needs MLX patch." The Gemma 4 family
verdict remains the same:
- gemma4-31b Dense: emits 0 tokens (chat-template gap, model-side)
- gemma4 26B-A4B MoE: RotatingKVCache.write_token missing (stack-side)

Neither is currently usable for agentic on this M4 stack.

## Fix sketch (for a future patch)

Need to either:
1. Add a `write_token(keys, values)` shim to RotatingKVCache that
   appends to the rotating buffer the same way standard caches do
2. Have `_batched_decode` detect RotatingKVCache and use a different
   write code path (e.g. iterate `update_and_fetch` per layer)

Either approach is ~1 hour of MLX backend work. Combined with the
nemotron-omni `is_full_attn` patch, this would be a single "fix MLX
backend for hybrid attention models" patch covering both Gemma 4 and
NemotronH. Filed as future patch 019 candidate alongside nemotron-omni.

## Files

- `run.log` — smoke.sh stdout (preflight OK, then long silence + TIMEOUT)
- `predictions.jsonl` — empty patch record
- `astropy__astropy-12907.log` — opencode session log (only `step_start` event before SIGKILL)
- `crash.log` — extracted AttributeError + SIGQUIT trace from launch.log
