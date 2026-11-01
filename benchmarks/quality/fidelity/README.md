# Quantization fidelity: qwen36 incumbent vs in-house 4-bit, 8-bit reference

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

## Files

- `qwen36-{8bit,incumbent,inhouse}-dump-2026-10-31.json` — per-position true-token logprob and top-20
- `qwen36-{incumbent,inhouse}-vs8bit-2026-10-31.json` — the reference's top-20 scored on each 4-bit arm
- `qwen36-*-loops-2026-10-31.json` — per-prompt finish reason, tokens, repetition metrics
