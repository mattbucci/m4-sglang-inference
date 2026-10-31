# Flint validation receipts — 2026-10-31

Stack: SGLang v0.5.23 + 10 patches (019 flint, 020 last-token prefill logits), mlx-lm git 736ab0d, MLX 0.32.4, env sglang-0523.
Preset `flint`: oQ8e default, CTX 32768, MAX_RUNNING 4, CHUNKED 1024, 2 GB MLX buffer cache, radix off, `--sampling-defaults model`. Probes at temperature 0.

## Gates per quantization

| Variant | validator | codegen (400-tok default) | codegen (4,096-tok budget) | thinking | greedy vs mlx-lm, pre-020 server | mlx-lm smoke: tok/s / peak GB / load s |
|---|---|---|---|---|---|---|
| oQ8e | --- 3/3 passed in 8.0s --- | VERDICT: STRONG | VERDICT: STRONG | THINKING VERIFIED | identical=False common_prefix_chars=197/500 | 64.6 / 42.32 / 10.4 |
| oQ6e | --- 3/3 passed in 8.1s --- | VERDICT: PARTIAL | VERDICT: STRONG | THINKING VERIFIED | identical=True common_prefix_chars=486/486 | 68.0 / 32.87 / 5.1 |
| oQ4e | --- 3/3 passed in 5.7s --- | VERDICT: PARTIAL | VERDICT: STRONG | THINKING VERIFIED | identical=True common_prefix_chars=521/521 | 73.0 / 23.45 / 3.6 |

The 400-token codegen PARTIALs are `finish=length` truncations (Flint reasons first); the 4,096-token column is the sweep configuration.

## Server bench, first pass (preset knobs of that run: CHUNKED 4096 / 4 GB cache; 192 output tokens, temperature 0)

**oQ8e**
- depth~64: prompt_tokens=83 ttft=0.26s prefill_tps=316 completion_tokens=192 decode_tps=61.9 wall=3.3s

**oQ6e**
- depth~64: prompt_tokens=83 ttft=0.29s prefill_tps=290 completion_tokens=192 decode_tps=64.8 wall=3.2s
- depth~4096: prompt_tokens=3495 ttft=4.60s prefill_tps=760 completion_tokens=192 decode_tps=61.2 wall=7.7s
- depth~16384: prompt_tokens=13863 ttft=21.72s prefill_tps=638 completion_tokens=192 decode_tps=56.0 wall=25.1s

**oQ4e**
- depth~64: prompt_tokens=83 ttft=0.22s prefill_tps=381 completion_tokens=192 decode_tps=70.5 wall=2.9s
- depth~4096: prompt_tokens=3495 ttft=4.93s prefill_tps=708 completion_tokens=192 decode_tps=67.5 wall=7.7s
- depth~16384: prompt_tokens=13863 ttft=23.29s prefill_tps=595 completion_tokens=192 decode_tps=61.5 wall=26.3s

## oQ8e envelope + post-patch-020 checks (chain5.log, final.log)

**A2 sampled: oQ8e CHUNKED=1024 CACHE=2 MR=4 (patch 020)  Sat Oct 31 01:16:37 PDT 2026**
- READY in 29s (CHUNKED=1024 CACHE=2GB MR=4)
- depth~4096: prompt_tokens=3495 ttft=3.78s prefill_tps=926 completion_tokens=96 decode_tps=59.1 wall=5.4s
- depth~16384: prompt_tokens=13863 ttft=17.14s prefill_tps=809 completion_tokens=96 decode_tps=54.4 wall=18.9s
- depth~32000: prompt_tokens=27039 ttft=42.62s prefill_tps=634 completion_tokens=96 decode_tps=49.3 wall=44.5s
- server alive after bench
- min memorystatus level GB: 17.3 at 01:17:34.884 (level=27% pressure=1 free=0.1 inactive=10.4 spec=0.0)
- min free+inactive GB: 10.3 at 01:18:16.096 (level=27% pressure=1 free=0.2 inactive=10.1 spec=0.0)
- min free GB: 0.0 at 01:17:00.736 (level=87% pressure=1 free=0.0 inactive=27.0 spec=0.5)
- max vm pressure level: 1

**B2 sampled: oQ8e CHUNKED=2048 CACHE=4 MR=4 (patch 020)  Sat Oct 31 01:18:24 PDT 2026**
- READY in 31s (CHUNKED=2048 CACHE=4GB MR=4)
- depth~4096: prompt_tokens=3495 ttft=3.78s prefill_tps=924 completion_tokens=96 decode_tps=59.1 wall=5.4s
- depth~16384: prompt_tokens=13863 ttft=18.82s prefill_tps=736 completion_tokens=96 decode_tps=54.0 wall=20.6s
- depth~32000: prompt_tokens=27039 ttft=46.20s prefill_tps=585 completion_tokens=96 decode_tps=49.1 wall=48.1s
- server alive after bench
- min memorystatus level GB: 15.4 at 01:20:09.850 (level=24% pressure=1 free=0.1 inactive=8.5 spec=0.0)
- min free+inactive GB: 8.5 at 01:20:09.850 (level=24% pressure=1 free=0.1 inactive=8.5 spec=0.0)
- min free GB: 0.0 at 01:18:49.702 (level=90% pressure=1 free=0.0 inactive=28.3 spec=0.4)
- max vm pressure level: 1

**greedy equivalence after patch 020: oQ6e vs mlx-lm ref, oQ8e divergence gap  Sat Oct 31 01:20:18 PDT 2026**
- identical=True common_prefix_chars=486/486
- server greedy tokens=96 ref=96 first_divergence_index=None
- IDENTICAL token stream
- identical=True common_prefix_chars=500/500
- server greedy tokens=96 ref=96 first_divergence_index=None
- IDENTICAL token stream

**FINAL guarded envelope (explicit knobs 1024/2/4, new guard metric)  Sat Oct 31 01:23:16 PDT 2026**
- READY in 29s  (CHUNKED=1024 CACHE=2GB MAX_RUNNING=4)
- depth~4096: prompt_tokens=3495 ttft=3.76s prefill_tps=931 completion_tokens=96 decode_tps=58.6 wall=5.4s
- depth~16384: prompt_tokens=13863 ttft=17.21s prefill_tps=805 completion_tokens=96 decode_tps=54.9 wall=18.9s
- depth~32000: prompt_tokens=27039 ttft=42.69s prefill_tps=633 completion_tokens=96 decode_tps=49.7 wall=44.6s
- guard: 0 WARN lines, 0 CRITICAL; min free seen:  GB
- server alive after bench

**plain preset boot check + validator  Sat Oct 31 01:25:04 PDT 2026**
- 'model_path': 'mlx-community/flint-oQ8e'
- 'chunked_prefill_size': 1024
- MLX buffer cache limit set to 2.0 GB
-   [PASS] basic     finish=stop answer='paris'
-   [PASS] tool_call finish=tool_calls name='get_weather' args='{"location": "Paris"}'
-   [PASS] thinking  reasoning_seen terminated answer_ok           (307 tok, finish=stop)
- --- 3/3 passed in 7.7s ---
- depth~64: prompt_tokens=83 ttft=0.27s prefill_tps=307 completion_tokens=192 decode_tps=62.6 wall=3.3s
- guard: 0 CRITICAL, 0 WARN

## Other artifacts

- `serve-flint-session-oQ8e--*.log`: streaming tool-call, 4-way concurrency, 4,096-token codegen on oQ8e (chain4.log has the summary lines).
- `env-*--mem.log`: 0.5 s samples of kern.memorystatus_level / vm pressure / vm_stat during load + 4K/16K/27K prefill.
- `scan-*.log`: per-layer quantization scans (486/486 healthy). `pipeline2.log`: the first server pass for all three variants plus qwen36/coder-30b regression checks.
