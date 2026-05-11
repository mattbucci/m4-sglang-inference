# Hybrid concurrent prefill — trace plan

DeltaNet hybrid models (Qwen3.5/3.6, Coder-Next) crash on `MAX_RUNNING>1` during
**prefill** with a broadcast shape mismatch. Decode-path serial fallback (lines
959-972 of model_runner.py) protects the decode side, but prefill goes through
the inner mlx_lm forward directly and crashes when concurrent requests have
different prefill lengths.

## Failure signature (from 2026-05-11 reproduction)

```
ValueError: [broadcast_shapes] Shapes (1,24,108,64) and (1,1,38,64) cannot be broadcast.
```

Decoded:
- `(1, 24, 108, 64)` — one tensor with batch=1, heads=24, seq=108, head_dim=64
- `(1, 1, 38, 64)`  — another with batch=1, heads=1, seq=38, head_dim=64

The 108 and 38 are sequence lengths from two different requests. The mismatch
implies one request's tensor is being broadcast against another's in a single
op.

## Confirmed NOT the cause

1. **Per-request cache state** — `_acquire_cache` returns `make_cache()`
   directly for hybrids (model_runner.py:182-183), each invocation yields a
   fresh list of `ArraysCache(size=2)` + `KVCache()` instances with unique
   `id()`. Verified by inserting `assert id(cache_A[0]) != id(cache_B[0])`.

2. **BatchedDecodeContext thread-local leakage** — context is only set in
   `decode_batch_start` (line 984) and cleared in `finally` (line 994). Prefill
   path never sets it; `MLXAttentionWrapper.__call__` delegates to inner
   module when context is None (attention_wrapper.py:72-73).

3. **mlx_lm RoPE module state** — all rope variants in `rope_utils.py` precompute
   `_freqs` at `__init__` and never mutate in `__call__`. `mx.fast.rope` is
   stateless.

4. **`gated_delta_kernel`** — pure functional kernel, no module state. Both
   ops-based and metal-kernel paths take (q, k, v, g, beta, state) and return
   (y, new_state).

5. **`ArraysCache.lengths`** — only set by `prepare(lengths=...)` (cache.py:647).
   We never call `prepare()` in our prefill path; lengths stays None throughout.

## Hypothesis A — lazy graph interference

Two prefill requests submitted within one scheduling step:

```python
# req A (108 tokens)
cache_A = self._build_fresh_cache()         # fresh
model_output_A = self.model(input_ids_A, cache=cache_A)  # lazy graph A
lazy_token_A = mx.argmax(...)

# req B (38 tokens) — immediately after, A not yet evaluated
cache_B = self._build_fresh_cache()         # fresh
model_output_B = self.model(input_ids_B, cache=cache_B)  # lazy graph B
lazy_token_B = mx.argmax(...)

# both materialised when caller invokes mx.async_eval / mx.eval
```

If `self.model` mutates any module-level state during the second call that
the first call's still-lazy graph reads, broadcast shapes diverge. The model
parameters are read-only, but module instance attributes (`_inner`, `_layer_idx`
on MLXAttentionWrapper) are set once at init — no per-call mutation.

The remaining candidate is mlx_lm's `Qwen3NextAttention.__call__` line 153:
```python
output = scaled_dot_product_attention(
    queries, keys, values, cache=cache, scale=self.scale, mask=mask
)
```
where `mask` was computed by `create_attention_mask(hidden_states, cache[self.fa_idx])`
at the model-call boundary (qwen3_5.py:268). If the mask object reference is
shared somehow (it shouldn't be — cache_A is different from cache_B), this
breaks.

## Hypothesis B — packed-prefill batch from scheduler

SGLang's scheduler may concatenate two concurrent prefill requests into one
batch of shape `(1, 146)` instead of two `(1, 108)` and `(1, 38)` calls. If so,
the inner mlx_lm model would treat them as one 146-token sequence with full
causal attention — but somewhere in the cache update path the 108-segment and
38-segment get split and broadcast incorrectly.

Check: instrument `prefill_start` to log `len(new_token_ids)` per call. If MR=4
serial-bench-serving fires it once with 146 tokens, hypothesis B. If twice with
108 and 38, hypothesis A.

## Hypothesis C — chunked-prefill partial cache

With `--chunked-prefill-size N`, a request larger than N is split into chunks.
`prefill_start` handles chunk 1 (offset 0 → N), `extend_start` handles chunk 2
(offset N → 2N). cache.offset advances between calls. If two concurrent
requests both have chunked prefill, the scheduler interleaves chunks: A1, B1,
A2, B2... If chunks A2 and B2 see different cache.offset values due to bookkeeping
state in `self._req_caches`, broadcast fails.

Check: smoke-test pattern from memory is `MAX_RUNNING=2 EXTRA_ARGS="--disable-radix-cache"
bash scripts/launch.sh qwen35` + `sglang.bench_serving --dataset-name random
--num-prompts 8 --request-rate inf`. The `--disable-radix-cache` flag enables
the early-return branch (model_runner.py:799-812) where prefill_start does
single-call full-prompt prefill. If the bug still fires with
`--disable-radix-cache`, hypothesis C is ruled out.

## Concrete trace patch (env-flag guarded)

Add to model_runner.py at the top of `prefill_start`, `extend_start`,
`decode_batch_start`:

```python
if os.environ.get("SGLANG_MLX_TRACE_PREFILL"):
    print(f"[TRACE prefill_start] req={req_id[:8]} "
          f"new_tokens={len(new_token_ids)} "
          f"prefix_slot_ids={len(prefix_slot_ids)} "
          f"is_hybrid={self._is_hybrid_model}", flush=True)
```

Add to `attention_wrapper.py` MLXAttentionWrapper.__call__:

```python
if os.environ.get("SGLANG_MLX_TRACE_ATTN"):
    print(f"[TRACE attn layer={self._layer_idx}] x.shape={x.shape} "
          f"ctx={'set' if ctx is not None else 'none'}", flush=True)
```

Add to model_runner.py `_build_fresh_cache` or similar at make_cache time:

```python
print(f"[TRACE make_cache] returning {len(cache)} layers, "
      f"types={[type(c).__name__ for c in cache[:3]]}...", flush=True)
```

## Repro recipe

1. `pkill -9 -f sglang`
2. `cd /Users/letsrtfm/AI/m4-sglang-inference`
3. Edit `scripts/launch.sh` qwen35 preset: `MAX_RUNNING=2`
4. `SGLANG_MLX_TRACE_PREFILL=1 SGLANG_MLX_TRACE_ATTN=1 EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh qwen35 2>&1 | tee /tmp/qwen35-trace.log`
5. Wait for server ready
6. In another shell: `python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 23334 --dataset-name random --num-prompts 8 --request-rate inf --random-input-len 300 --random-output-len 100`
7. When the crash fires, the last few `[TRACE ...]` lines pinpoint the layer and shapes.

## Expected diagnostic outcomes

- If `prefill_start` fires twice with new_tokens=108 then new_tokens=38, and
  no `extend_start` between — hypothesis A (lazy graph).
- If `prefill_start` fires once with new_tokens=146 — hypothesis B (packed).
- If `prefill_start` then multiple `extend_start` for the same req_id —
  hypothesis C (chunked).

Once the failing layer index is known from `[TRACE attn layer=N]`, look at the
inner Qwen3NextAttention.__call__ flow for that layer index. If layer N is a
DeltaNet linear_attn (not full attention), the issue is in `GatedDeltaNet`
which uses different shapes — the (1,24,...) tensor may not even be n_heads=24
in that case; it could be `num_v_heads` or `key_dim`.

## Until fixed

Production presets keep `MAX_RUNNING=1` for all hybrid models. Smoke-test that
2-way decode works (same prefill length) but acknowledge mixed-length concurrent
prefill is broken.
