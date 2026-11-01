# Quantization fidelity: community vs in-house 4-bit against the 8-bit reference

## Qwen3.6-35B-A3B (`qwen36`)

`scripts/eval/quant_fidelity.py` through the serving stack (`scripts/launch.sh
qwen36` defaults, radix on): 8 held-out documents (4 Salesforce/wikitext test
paragraphs, 4 scripts from this repo), 1,524 positions (532 prose, 992 code),
stride 4, top-20 next-token logprobs per position. KL is KL(reference ‖ model)
over the reference's top-20 support, both renormalized, with the model's
logprobs for those tokens scored exactly. `scripts/eval/probe_loops.py`: 40
fixed prompts, greedy, 2,048-token cap, thinking on (the preset default).

| Arm | PPL all / prose / code | NLL ± se | top-1 | KL to 8-bit, mean ± se (all / prose / code) | KL median | argmax agree | loops: degenerate / cap / unclosed |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3.6-35B-A3B-8bit` (reference) | 4.80 / 6.75 / 4.00 | 1.569 ± 0.059 | 0.643 | — | — | — | 3 / 34 / 20 |
| `mlx-community/Qwen3.6-35B-A3B-4bit` (incumbent) | 5.14 / 7.25 / 4.28 | 1.637 ± 0.061 | 0.634 | 0.0671 ± 0.0035 / 0.0739 / 0.0634 | 0.030 | 0.898 | 2 / 35 / 24 |
| in-house 4-bit (gates + vision BF16) | 5.14 / 7.22 / 4.28 | 1.636 ± 0.061 | 0.626 | 0.0690 ± 0.0035 / 0.0778 / 0.0642 | 0.028 | 0.904 | 2 / 34 / 20 |

Paired at identical positions (incumbent minus in-house): NLL +0.0008 ±
0.0035 nats (z 0.2); KL −0.0019 ± 0.0014 (z −1.4; prose −0.0039 ± 0.0017,
z −2.3; code −0.0008 ± 0.0019). The in-house build is closer to the
reference at 738 of 1,524 positions.

The reference shares the incumbent's 8-bit router rounding (both are
mlx-community recipes), so the KL metric favours the incumbent at near-tie
routing positions where a BF16 router picks differently; the reference-free
teacher-forced loss is the unbiased comparison, and it is a tie. The loop
probe does not separate the arms either: with thinking on, 34–35 of 40
greedy answers run to the cap on all three arms, the 8-bit reference
included, and 2–3 per arm degenerate into repetition.

Verdict: no measurable advantage for the in-house build on fidelity,
reference distance or long-generation behaviour; it is not promoted or
uploaded and stays an opt-in local build (`MODEL=~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse`).

## Files (qwen36)

- `qwen36-{8bit,incumbent,inhouse}-dump-2026-10-31.json` — per-position true-token logprob and top-20
- `qwen36-{incumbent,inhouse}-vs8bit-2026-10-31.json` — the reference's top-20 scored on each 4-bit arm
- `qwen36-*-loops-2026-10-31.json` — per-prompt finish reason, tokens, repetition metrics

## Qwen3.8-27B (`qwen36-27b` preset, `MODEL=` override)

Same protocol, served through `scripts/launch.sh qwen36-27b` (`CHUNKED=2048`,
radix on) with the in-house tokenizer for all three arms (identical vocab).
The 8-bit reference is served from a mirror of the `mlx-community/Qwen3.8-27B-8bit`
snapshot without its `processor_config.json` (see README Known Issues).

| Arm | PPL all / prose / code | NLL ± se | top-1 | KL to 8-bit, mean ± se (all / prose / code) | KL median | argmax agree | loops: degenerate / cap / unclosed |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3.8-27B-8bit` (reference) | 3.80 / 6.79 / 2.78 | 1.335 ± 0.055 | 0.690 | — | — | — | 0 / 20 / 18 |
| `mlx-community/Qwen3.8-27B-4bit` (community) | 3.87 / 6.83 / 2.86 | 1.354 ± 0.054 | 0.677 | 0.0341 ± 0.0018 / 0.0436 / 0.0291 | 0.012 | 0.913 | 1 / 24 / 20 |
| in-house 4-bit (gates + vision BF16) | 3.87 / 6.83 / 2.85 | 1.353 ± 0.054 | 0.673 | 0.0337 ± 0.0017 / 0.0443 / 0.0280 | 0.012 | 0.911 | 2 / 26 / 21 |

Paired at identical positions (community minus in-house): NLL +0.0006 ±
0.0014 nats (z 0.4; prose −0.0003 ± 0.0020, code +0.0010 ± 0.0018); KL
+0.0004 ± 0.0006 (z 0.6; prose −0.0007 ± 0.0004, z −1.7; code +0.0010 ±
0.0009, z 1.1). The two 4-bit builds pick the same greedy token at 1,506 of
1,524 positions, and the in-house build is closer to the reference at 790.

Downstream on the same stack: MMLU 86/100 vs 84, HumanEval 18/20 both,
needle 4/4 both, LabBench 0.375 vs 0.387 (`benchmarks/quality/Qwen38-27B-{inhouse,community}-2026-10-31.json`),
SWE-bench 3-instance smoke 2/3 non-empty patches both (the astropy-12907
rollout times out at 30 min on both; `evals/swebench/runs/qwen38-27b-*-2026-10-31/`),
decode 14.1 tok/s both. With thinking on (the 3.8 chat template ignores
`enable_thinking`), 24–26 of 40 greedy answers run to the 2,048-token cap on
the 4-bit arms and 20 on the reference, so the reference is marginally more
decisive but the two 4-bit arms do not separate.

Verdict: the in-house Qwen3.8-27B build is indistinguishable from the
community 4-bit on fidelity, reference distance, downstream quality and
long-generation behaviour; it is not uploaded and stays an opt-in local
build (`MODEL=~/AI/models/Qwen3.8-27B-4bit-inhouse scripts/launch.sh qwen36-27b`).

## Files (qwen38)

- `qwen38-{8bit,community,inhouse}-dump-2026-10-31.json` — per-position true-token logprob and top-20
- `qwen38-{community,inhouse}-vs8bit-2026-10-31.json` — the reference's top-20 scored on each 4-bit arm
- `qwen38-*-loops-2026-10-31.json` — per-prompt finish reason, tokens, repetition metrics
- `qwen38-*-launch-2026-10-31.log` — server logs for the three arms (decode throughput lines)
