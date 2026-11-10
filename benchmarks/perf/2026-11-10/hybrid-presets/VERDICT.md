# Hybrid presets: pipelined decode loop under the radix cache (patch 028)

Run date 2026-11-10. Stack: SGLang v0.5.23 + patches 002-028, MLX 0.32.4,
M4 Pro 64 GB. One gate per preset, same shape as the qwen36 gate in
`../qwen36-hybrid-overlap/`. Serve recipe: `CTX=36000 MEM_FRAC=0.45
scripts/launch.sh <preset>` with the OOM guard; flint runs its preset
context (`CTX=32768 MEM_FRAC=0.7`).

Arms: `off` passes `SGLANG_MLX_HYBRID_OVERLAP=0 SGLANG_MLX_COMMIT_CHAINED=0`
(normal scheduler loop), `on` passes both `=1`.

## Decode A/B (order off on on off)

`scripts/bench/decode_ab.py`, 3 runs per depth, `ignore_eos`, temperature 0;
256 tokens on the MoE presets and the 9B, 128 on the dense 27B presets.
Pooled median ms per token / tok/s over both arms of each side.

| preset | prompt tokens | off | on | on vs off | same text hashes |
|---|---|---|---|---|---|
| nemotron-30b | 948 | 13.08 / 76.4 | 11.10 / 90.1 | +17.9% | yes |
| nemotron-30b | 7,572 | 13.65 / 73.2 | 11.62 / 86.0 | +17.5% | yes |
| nemotron-30b | 30,358 | 15.36 / 65.1 | 13.22 / 75.7 | +16.2% | yes |
| flint | 957 | 20.18 / 49.5 | 18.15 / 55.1 | +11.2% | yes |
| flint | 7,599 | 21.12 / 47.3 | 19.04 / 52.5 | +10.9% | yes |
| flint | 22,801 | 23.43 / 42.7 | 21.19 / 47.2 | +10.6% | yes |
| qwen35-9b-8bit | 948 | 39.67 / 25.2 | 37.42 / 26.7 | +6.0% | yes |
| qwen35-9b-8bit | 7,583 | 40.91 / 24.4 | 39.41 / 25.4 | +3.8% | yes |
| qwen36-27b | 948 | 72.21 / 13.8 | 68.75 / 14.5 | +5.0% | yes |
| qwen36-27b | 7,583 | 74.12 / 13.5 | 70.13 / 14.3 | +5.7% | yes |
| qwen35 | 948 | 72.88 / 13.7 | 69.24 / 14.4 | +5.3% | yes |
| qwen35 | 7,583 | 74.48 / 13.4 | 71.03 / 14.1 | +4.9% | yes |

The loop saves a fixed 2-3.5 ms of host time per step, so the gain scales
with how short the step is: +16-18% on a 13 ms step, +5% on a 70 ms step.
Wired memory after each arm is equal between sides (nemotron-30b 23.5-23.8
GB, qwen35-9b-8bit 20.1, flint 46.9-47.0, both 27B presets 26.2-26.4). No
tracebacks in any server log. Repeat-prompt TTFT is equal between sides
except qwen36-27b, where a hit answers 0.06-0.07 s later on the overlap
loop (0.73 / 0.50 s against 0.67 / 0.44 s at 948 / 7,583 tokens), about one
decode step; flint is 0.02 s later.

## Finish-state gate

`scripts/eval/probe_finish_state_resume.py`, prompt depth 1,500, finishes
on length and on a stop token. Generation lengths 2,3,4,5,9,17,33,64,65 (14
cases) on nemotron-30b, flint and the 9B; 2,3,5,17,64,65 (9 cases) on the
27B presets. `resume-ref.json` is the normal loop, `resume-on.json` the
overlap loop checked against it with `--ref`.

| preset | leaf hits (normal / overlap) | resumed ids equal to the normal loop | max logprob gap to the normal loop | max hit-versus-cold gap, normal loop |
|---|---|---|---|---|
| nemotron-30b | 14 / 14 | 14 of 14 | 0.0 | 0.0 |
| flint | 14 / 14 | 14 of 14 | 0.0 | 0.77 |
| qwen35-9b-8bit | 14 / 14 | 14 of 14 | 0.0 | 0.41 |
| qwen36-27b | 9 / 9 | 9 of 9 | 0.0 | 1.80 |
| qwen35 | 9 / 9 | 9 of 9 | 0.0 | 0.33 |

The overlap loop resumes bit-equal to the normal loop from every finish
shape on every preset. On qwen36-27b the probe's standalone threshold
(resumed tail against a cold run of the same tokens) fails 2 of 9 cases on
both loops with identical numbers: a leaf hit differs from cold by up to
1.80 nats on these word-salad prompts. That is the normal loop's
hit-versus-cold noise, not a property of the patch.

## Capability validator

`validate_capabilities` on the overlap loop: nemotron-30b 3/3, flint 3/3,
qwen35-9b-8bit 3/3, qwen36-27b 3/3, qwen35 2/3. The qwen35 miss is the
greedy thinking check running to its 2,048-token cap; the normal loop
returns the same 2/3 with the same miss (`qwen35/validate-normal-loop.log`).

## qwen36 as shipped

`qwen36-as-shipped/`: `scripts/launch.sh qwen36` with no flag in the
environment (`disable_overlap_schedule: False` in the server arguments),
against both flags `=0`, order off default default off, 256 tokens.

| prompt tokens | off | default | default vs off | same text hashes |
|---|---|---|---|---|
| 948 | 15.55 / 64.3 | 12.38 / 80.8 | +25.5% | yes |
| 7,583 | 16.28 / 61.4 | 13.09 / 76.4 | +24.3% | yes |
| 30,391 | 18.65 / 53.6 | 15.24 / 65.6 | +22.4% | yes |

Gates on the default launch: validator 3/3, `probe_thinking` VERIFIED,
`probe_vision` STRONG, `probe_codegen` STRONG, the 2,048-token repeat test
4/4, finish-state probe 13/13 equal to the normal-loop reference at a 0.0
gap, 0 tracebacks, 28.5 GB wired after the sweep. Repeat-prompt TTFT is
0.01 s later on the default (0.18 / 0.15 / 0.30 s against 0.17 / 0.14 /
0.29 s). `probe_multiturn_hit_determinism` exits 1 with the same rows as
both loops in `../qwen36-hybrid-overlap/gates/multiturn-*.json`.

## Disposition

Every hybrid preset passes, so `scripts/launch.sh` exports both flags as 1
for hybrid presets when the radix cache is on;
`SGLANG_MLX_HYBRID_OVERLAP=0` restores the normal loop. Radix-off launches
already run the overlap loop and are left untouched.
