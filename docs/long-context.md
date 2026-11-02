# Long context on 64 GB unified memory

## The 256K recipe

qwen36 serves a 256K single-user context: in=251,659 server-verified tokens prefill in
~23 min, and the multi-needle recall ladder scores 6/6 at every rung from 9.6K to 245,656
realized tokens (`benchmarks/longctx-bisect/`, `benchmarks/quality/depth-recall/`).

```bash
bash scripts/common/oom_guard.sh &          # mandatory for ≥32K
CTX=<label+64> MEM_FRAC=0.5 CHUNKED=1024 EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh qwen36
```

Four settings carry the ceiling:

- **MLX buffer-cache cap** (patch 008, `SGLANG_MLX_CACHE_LIMIT_GB`, default 4): uncapped,
  chunked prefill retains ~0.6 MB/token and jetsams around 30K.
- **Cache pre-sizing** (patch 015): the per-request attention cache is sized to the request,
  so no power-of-two doubling overshoot at depth.
- **Exact pool sizing** (`CTX = label + 64`): a 192K label at CTX=210K dies where CTX=196608
  completes; overallocating the pool costs the last 2–3 GB.
- **1024-token chunks**: the 4096 default's transients swing free memory 1–11 GB per chunk and
  kill deep prefills at ~100–113K; 1024 buys ~4.3 GB of transient margin over 2048 at no speed
  cost. Chunk size sets the transient peak, not the steady per-token growth.

There is no decode wall: streamed decode is 61.7 tok/s at 32K, 36.0 at 128K and 30.2 at 151K
(`benchmarks/longctx-bisect/decode-curve/DECISION.md`). Bench tooling needs a long urllib
timeout for deep runs. `--kv-cache-dtype` is inert on this backend (bf16 KV in every run), so
the levers that move memory are the four above plus `--max-total-tokens`, `MAX_RUNNING` and
request `max_tokens`. Models with native context below the requested `CTX` get linear RoPE
interpolation (logged at boot).

## Memory budget at 256K

| Model class | Weights | KV @256K (bf16) | Fits? |
|-------|:-------:|:---------------:|:------|
| MoE 3B-active (Coder-30B, Qwen3-30B-DWQ) | 16 GB | 24 GB | yes, with the recipe above |
| MoE+DeltaNet (Qwen3.6-35B-A3B) | 17 GB | full-attention layers only | yes (256K validated) |
| Flint 40B-A2.5B (oQ8e) | 42 GB | 7.5 GB (14 full layers, 4 KV heads) | 32K preset validated; deeper runs need headroom measured per depth |
| Dense+VL (Devstral, Qwen3.5-27B) | 14–15 GB | 42 GB | no; the dense attention ceiling applies first |
| Dense (Qwen3-32B) | 18 GB | 66 GB | no |

**MoE wins at long context.** Each decode token reads the model weights and the whole KV
cache. At short context weight reads dominate (MoE 1.5 GB vs dense 14 GB). At 256K the KV
read grows to several GB, so keeping the weight term small matters more, not less.
**DeltaNet hybrids** (Qwen3.5/3.6/3.8) alternate full attention with linear attention whose
cost does not grow with context, which is why they hold decode flat at depth; the
full-attention layers still pay O(n).

## Radix prefix cache and agentic turns

Agentic workloads re-send the same system prompt + history every turn; the radix cache
reuses the shared-prefix KV so follow-up turns prefill only their new tokens. It is on for
every preset, including hybrids (DeltaNet/Mamba2 state snapshotted per prefix) and Flint
(patch 021). Greedy output is deterministic on both sides of the cache, and a hit differs
from a cold run only as much as two cold runs on different chunk sizes do: over 24 positions
of a 16.6K-token agent request the top token matches at 20–21 of 24 with a median top-1
shift of 0.02–0.05 nats (`benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md`).

A follow-up turn re-rendered through the Qwen3.x template diverges 1–2 tokens before the
primed prompt end, so it is served from the 64-token-grid DeltaNet checkpoint (patch 026)
with `AUX_SLOTS=64` checkpoint slots per hybrid preset: the first follow-up on Qwen3.8-27B
recomputes 55–77 tokens (0.9–1.1 s TTFT at 8.7K–52K) and on qwen36 21–73 tokens (0.2–0.5 s).
Appending a tool result to a cached prefix costs the model's own short-prefill rate on the
recomputed tokens (~35 ms + ~1.2 ms per token: the suffix plus the remainder past the last
64-token checkpoint), 80 ms at 8.6K and 260 ms at 34.5K to the first token, with 5 ms / 35 ms
of hit-path overhead on top. Patch 027 cuts the chat-prompt re-tokenization to 4–9 ms at 250K.

Radix-on sessions hold flat memory for 150 turns under a 32K-rolling agentic load
(free+inactive 28.4 → 26.2 GB with a 24.7 GB trough, no guard warning, 2.8 s median TTFT with
cache hits to the end, 56 tok/s decode); `/flush_cache` is a reset, not a cadence
(`benchmarks/session-endurance/VERDICT.md`). Deep one-shot prefills (32K+ with a big pool)
use the radix-off recipe above; `EXTRA_ARGS="--disable-radix-cache"` restores the uncached
path for A/B runs.
