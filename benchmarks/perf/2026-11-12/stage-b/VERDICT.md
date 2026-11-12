# Stage B: decode-flag ladders on every preset but qwen36 and flint

Run 2026-11-11 to 2026-11-12, display asleep, frozen tree at patch 041 (v0.5.23-18). Each preset ran its
ladder forward (`-a`) and reversed (`-b`): 74 arms. Serving recipe: `CTX=36000 MEM_FRAC=0.45
SGLANG_MLX_CACHE_LIMIT_GB=2 --disable-radix-cache`. Instruments: `published_shape_sweep.py` (cold and
repeat decode at 1K / 8K / 30K, 4-5 runs sampled and greedy at 1K) and `concurrent_decode_probe.py`.
`analyze.py` reads the receipts in this directory and writes `analysis.json`.

Rule, written before the run: a rung becomes a preset default when it wins at least 2 % at one depth in
both orders, loses nowhere by more than the 1 % noise floor, and raises the wired peak by no more than
2 GB. The preset's capability gates then run with it set.

| Preset | Rung | Decode 1K | Decode 8K | Decode 30K | Top stream count | Prefill | Wired | Disposition |
|---|---|---|---|---|---|---|---|---|
| coder-30b | router + 5 parts | +6.5 / +7.2 % | +3.4 / +7.1 % | +2.9 / +3.6 % | +5.9 / +6.0 % | +1.0 to +2.7 % | +0.2 GB | **default** |
| coder-30b | + SwiGLU fusion | +6.3 / +6.8 % | +3.8 / +5.0 % | +3.4 / +3.5 % | +5.2 / +6.1 % | -14 to -32 % | +0.1 GB | rejected (prefill) |
| qwen3-moe | router + 5 parts | +6.6 / +6.3 % | +2.6 / +5.7 % | +2.6 / +4.6 % | +5.2 / +5.3 % | -0.1 to +2.2 % | +0.1 GB | **default** |
| qwen3-moe | + 200 MB buffers | +9.8 / +9.7 % | +6.5 / +8.2 % | +2.2 / +4.2 % | +4.7 / +5.2 % | +0.1 to +3.2 % | +0.6 GB | opt-in; the `-a` arm overlapped an active session |
| qwen3-32b | every rung | within 1.5 % | within 1.5 % | n/a | within 0.7 % | level | up to +1.4 GB | neutral |
| devstral | every rung | within 1.4 % | within 1.2 % | n/a | within 1.1 % | level | up to +0.7 GB | neutral |
| qwen36-27b | front + linears + 5 parts | +2.2 / +0.9 % | +1.7 / +0.9 % | n/a | n/a | level | +0.6 GB | under the bar |
| qwen36-27b | + 200 MB buffers | +2.6 / +1.9 % | +2.5 / +2.3 % | n/a | n/a | -2.9 / +1.3 % at 8K | +6.9 GB | fails wired limit |
| qwen35 | front + linears + 5 parts | +1.0 / +0.9 % | +0.9 / +0.4 % | n/a | +0.6 / +1.0 % | level | +0.0 GB | under the bar |
| qwen35 | + 200 MB buffers | +1.9 / +2.0 % | +1.8 / +0.8 % | n/a | +0.6 / +0.6 % | level | +5.0 GB | fails wired limit |
| qwen35-9b-8bit | + 200 MB buffers (best) | +2.7 / +1.9 % | +2.2 / +1.2 % | +0.5 / +1.3 % | n/a | -5.0 / -0.7 % at 30K | +3.1 GB | fails wired limit |
| nemotron-30b | 5 parts + 200 MB buffers | +0.1 / +2.5 % | +4.2 / +1.6 % | +0.9 / -1.1 % | n/a | level | +0.0 GB | fails (loss at 30K in one order) |

Cells are the two orders. Chained commit (`-cc`) never separated from its parent rung. Five-part
submission alone is level for one stream and +3.5 to +4.2 % aggregate at 8 streams on the two MoE presets.

Overlapped arms (session active while they ran): `qwen3-moe-router-sub5-buf-a` and
`qwen35-front-lin-sub5-b`. Neither changes a disposition; the first is why the buffer rung stays opt-in.

Single-shot cold prefill moves 2 to 6 % between the two base arms of some presets, so several hybrid
rungs trip the 1 % loss clause in one order only. The rule was applied as written; no verdict depends on it.

Gates with the new defaults on the live tree (`gate-*.txt`): both presets pass `validate_capabilities` 3/3,
codegen STRONG, default sampling 8/8; qwen3-moe thinking VERIFIED (coder-30b is not a thinking model).
Baselines re-armed: coder-30b 88.3 / 57.5 / 33.2 tok/s and qwen3-moe 88.4 / 57.7 / 32.5 at 1K / 8K / 32K.
