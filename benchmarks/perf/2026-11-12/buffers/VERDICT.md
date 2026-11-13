# 200 MB command buffers, mirrored reruns on `qwen3-moe` and `nemotron-30b`

`MLX_MAX_MB_PER_BUFFER=200` against the same preset without it, four mirrored pairs each
(reference, arm, arm, reference, twice), display asleep, `published_shape_sweep.py`. The
`qwen3-moe` reference is the preset default (fused router, five-part submission); both
`nemotron-30b` arms set `SGLANG_MLX_DECODE_SUBMIT_PARTS=5`. Change per pair, arm over reference:

| Preset | Metric | Reference | Arm | Mean | Pairs |
|---|---|---:|---:|---:|---|
| qwen3-moe | decode 1K, tok/s | 87.2 | 88.6 | +1.6 % | +1.1 +2.7 +1.0 +1.6 |
| qwen3-moe | decode 8K | 68.0 | 68.9 | +1.4 % | +1.8 +2.1 -0.3 +2.0 |
| qwen3-moe | decode 32K | 37.7 | 38.5 | +2.1 % | +3.2 +1.5 +2.2 +1.4 |
| qwen3-moe | prefill 32K | 343.9 | 349.0 | +1.5 % | +3.2 -0.2 +0.1 +2.9 |
| nemotron-30b | decode 1K, 5 greedy runs | 94.0 | 95.9 | +2.0 % | +1.9 +1.8 +1.7 +2.7 |
| nemotron-30b | decode 8K | 89.3 | 90.7 | +1.6 % | +0.8 +1.2 +2.9 +1.4 |
| nemotron-30b | decode 32K | 77.6 | 78.3 | +0.8 % | +1.2 +1.3 +0.9 +0.0 |
| nemotron-30b | prefill 32K | 637.7 | 639.7 | +0.3 % | +0.0 +0.6 +1.1 -0.4 |

Wired peak: qwen3-moe 34.0 to 34.6 GB, nemotron-30b level at 32.2 GB.

Verdict: a consistent gain of 1 to 2 % with no loss past the noise floor on either preset; the
single-order loss at 30K on `nemotron-30b` in stage B does not repeat. The promotion bar
(2 % at one depth in both orders) is not met in every pair, so the variable stays opt-in.
