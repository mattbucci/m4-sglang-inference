# qwen36: pipelined decode loop under the radix cache (patch 028)

Run date 2026-11-10. Stack: SGLang v0.5.23 + patches 002-028, MLX 0.32.4,
`mlx-community/Qwen3.6-35B-A3B-4bit`, M4 Pro 64 GB. Serve recipe for every
arm: `CTX=36000 MEM_FRAC=0.45 scripts/launch.sh qwen36` with the OOM guard.

Arms: `off` is the normal scheduler loop (radix cache on, overlap schedule
disabled by upstream's `no_buffer` resolution). `on` adds
`SGLANG_MLX_HYBRID_OVERLAP=1 SGLANG_MLX_COMMIT_CHAINED=1`.

## Decode A/B (`ab/`, order on off off on)

`scripts/bench/decode_ab.py`, 256 tokens, 3 runs per depth, `ignore_eos`,
temperature 0. Median ms per token / tok/s.

| prompt tokens | off-1 | on-1 | off-2 | on-2 | on vs off |
|---|---|---|---|---|---|
| 948 | 15.38 / 65.0 | 12.24 / 81.7 | 15.53 / 64.4 | 12.40 / 80.7 | +25.7% / +25.3% |
| 7,583 | 16.04 / 62.4 | 12.99 / 77.0 | 16.12 / 62.0 | 13.11 / 76.3 | +23.5% / +23.0% |
| 30,391 | 18.48 / 54.1 | 15.10 / 66.2 | 18.27 / 54.7 | 15.01 / 66.6 | +22.4% / +21.7% |

Text hashes are identical between arms at every depth (cold run and hit
runs each). Repeat-prompt TTFT 0.18 / 0.15 / 0.29 s on both arms. Wired
memory after the arm 26.6-26.7 GB on both. `radix-off-comparator.json` is
the radix-off launch (overlap loop, no cache): 12.11 / 12.81 / 14.91 ms,
so the `on` arm decodes at the radix-off rate and keeps the cache.

The census (`../census-qwen36.json`) puts the raw pipelined loop at 11.15 /
12.15 / 14.86 ms and the raw sync loop at 13.86 / 14.75 / 17.76 ms for
depths 24 / 8,192 / 32,768.

## Finish-state gate (`resume/`)

`scripts/eval/probe_finish_state_resume.py`, prompt depth 1,500, generation
lengths 2,3,4,5,9,17,33,64,65, finishes on length and on a stop token, 13
cases. Each case finishes a request, then resumes from `prompt + reply`.

| arm | leaf hits | resumed ids vs reference | max logprob gap vs reference | verdict |
|---|---|---|---|---|
| normal loop (reference) | 13/13 | n/a | n/a | PASS |
| overlap + commit | 13/13 | 13/13 equal | 0.0 | PASS |
| overlap, no commit | 2/13 (2-token replies), 11 resume from the 1,472 grid state | 2/2 equal | 0.0 | no wrong state donated |
| negative control (live state stored one token ahead) | 13/13 | 9/13 differ | 0.02-2.68 | FAIL 13/13 |

The standalone thresholds catch only 4 of the 13 negative-control cases:
hit-versus-cold noise on these word-salad prompts is already 0.24 on a leaf
hit and 0.81 from a grid state, so the comparison against the normal-loop
reference is the conclusive check.

## Capability gates on the overlap arm (`gates/`)

`validate_capabilities` 3/3, `probe_thinking` VERIFIED, `probe_vision`
pass, `probe_codegen` STRONG, `test_radix_cache_repeat --max-tokens 2048`
4/4 "Tokyo", 0 tracebacks, wired 27.9-28.5 GB after the sweep.
`validate-final-code.log` and `repeat-2048.log` are from the committed
patch; the other three logs are from the same patch with the commit step
calling `alloc_token_slots`, which the committed version replaces by its two
constituent calls so that an exhausted pool is a quiet fallback.

`probe_multiturn_hit_determinism` returns the same rows on both arms
(`multiturn-*.json`): at depth 8,192 two hits diverge from each other at
token 24, at 24,576 they agree and diverge from the cold run at token 55.
That is the normal loop's behaviour and is unchanged by the patch.
