# Decode-at-depth: the wall dissolves — decode-topk port is NO-GO

Instrument: `scripts/bench/bench_long_context.py` streaming (TTFT and
per-token ITL separated; the old whole-request metric is renamed
`amortized_s_per_token`). Serve: qwen36, CTX=175000, MEM_FRAC=0.5,
CHUNKED=2048, radix off, turboquant. Server-verified depth per rung;
streamed decode cross-checked against server-log gen-throughput (128K:
streamed 36.0 vs server-log 36.78 tok/s — 2.1% agreement).

## The true decode curve (steady state, out=256)

| depth (server-verified) | decode tok/s | ITL p50 | TTFT | old "amortized" reading |
|---:|---:|---:|---:|---:|
| 31,458 | 61.7 | 16 ms | 51 s | 1.59 s/tok |
| 62,916 | 50.5 | 20 ms | 127 s | 4.15 s/tok |
| 94,374 | 41.7 | 24 ms | 244 s | 7.69 s/tok |
| 125,830 | **36.0** | 27 ms | 399 s | 12.57 s/tok |
| 150,997 (out=32 arm) | 30.2 | 31 ms | 553 s | 17.31 s/tok |

(The out=32 arm agrees within ~10% at every depth — post-prefill transients
are small; the 153K out=256 rung dropped the connection and is published
from the out=32 arm.)

## Verdict

- **The "13-19 s/token decode wall" never existed.** It was whole-request
  elapsed ÷ output tokens at out=32 — ~99.7% prefill amortization at 128K.
  True decode at 128K is 36.0 tok/s (27 ms/token), ~350× faster than the
  cited number.
- **128K decode is 1.7× the 32K rate** (61.7 → 36.0) — inside the spec's
  ~2× "wall dissolves" threshold. The hybrid architecture explains the
  gentle slope: DeltaNet layers carry recurrent state (O(1) in depth); only
  the few full-attention layers pay the KV read.
- **Decode-topk port (would-be patch 017): NO-GO.** The sister-rig ports
  (R9700 1.77×, 3090 2.03× at 256K-class depth) attack full-attention KV
  read walls that this hybrid does not have; the entire available win at
  128K is bounded by the 1.7× depth ratio, most of which lives in DeltaNet
  layers a topk port does not touch.
- **Radix-on/off decode attribution at 32K** (from `benchmarks/radix-ab/`):
  radix-off 57.3 tok/s vs radix-on 48.6 — the overlap-schedule delta, not a
  depth effect; consistent with this curve's 61.7 (different serve CTX).
- **Doc 08 phase 2 is closed**: capacity reached 256K with configuration
  (exact-CTX pool + CHUNKED=1024 + patch 015), and decode at depth is not a
  wall. The long-context campaign's remaining open threads live in
  agentic-endurance (session-scale retention) and the 153K out=256
  connection drop (single occurrence, unreproduced).

Real user experience at 128K: ~6.7 min to first token (prefill), then
36 tok/s — interactive-grade decode at depths the amortized metric painted
as unusable.
