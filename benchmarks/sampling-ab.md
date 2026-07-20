# M4-B: real sampling (patch 016) — gate receipts

Per-request temperature/top_p/top_k/min_p via `mlx_lm.make_sampler`, all 7
token-selection sites. Receipts in `benchmarks/sampling-ab/`.

| Gate | Result | Receipt |
|---|---|---|
| Greedy token identity (3 prompts, temp=0, pre vs post) | **PASS** — token-exact | `gate1-greedy-identity.json` vs `baseline-qwen36-greedy3.json` |
| Sampling divergence (temp=1.0, 5 repeats) | **PASS** — 5/5 distinct on a flat-distribution prompt | `gate2-divergence.json` |
| Thinking probe, explicit sampling (t=0.6, top-p 0.95) | **VERIFIED** | `gate3-qwen36-t06.txt` |
| Thinking probe, `--sampling-defaults model` (temperature omitted) | **VERIFIED** | `gate3-qwen36-modeldefaults.txt` |
| Greedy perf (MR=1 gen throughput, steady-state window) | **PASS** — 65.0 vs 65.9 tok/s baseline (−1.4%, gate ±3%) | `gate4-genthroughput.txt`, `baseline-qwen36-genthroughput.txt` |
| Sampled perf cost | −4.2% vs greedy (62.3 tok/s) — recorded, kill line was 25% | same |
| Mixed MR=4 batch (qwen3-moe: 2 greedy + 2 temp=1.0, 2 runs) | **PASS** — greedy rows identical across runs, sampled rows vary, no cross-row bleed | `gate4b-mixed-batch.json` |
| Modality (qwen36 vision / codegen) | **STRONG / STRONG** — unchanged | `gate5-vision.txt`, `gate5-codegen.txt` |
| Unit sanity (synthetic logits through `_select_tokens`) | 6/6 groups | `scripts/test/test_sampling_select.py` |

Notes:

- **Loop-fix claim: untestable (null result).** The spec's hypothesis (i) —
  sampling terminates the Qwen3.x infinite-`<think>` loop — required a preset
  that loops at baseline. qwen35 (the receipt-backed v0.5.12-era looper)
  returns THINKING VERIFIED under greedy on the current stack
  (`baseline-qwen35-thinking.txt`), so no baseline looper exists; the patch
  lands as fleet-parity infrastructure per the spec's fallback.
- **Peaked-prompt determinism is expected**: 5 temp=1.0 repeats of a
  near-delta-distribution prompt ("list the first 8 primes") produce
  identical completions — per-step max-prob ≈ 1 dominates the categorical
  draw. Divergence must be probed with flat-distribution prompts.
- Greedy stays bit-stable by construction: greedy requests are never
  registered (SGLang encodes greedy as top_k==1) and the all-greedy batch
  path is the unchanged `mx.argmax` graph.
