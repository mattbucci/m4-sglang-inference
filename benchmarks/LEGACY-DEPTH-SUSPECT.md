# Depth-labeled benchmark artifacts — instrument classification

Two depth instruments produced the artifacts in this directory. They have two
distinct depth bugs; every artifact classifies under exactly one instrument.

## Instrument A — `sglang.benchmark.serving` random dataset (SUSPECT pre-pin)

**Mechanism:** with `--random-range-ratio` unpinned, the upstream default (0.0)
draws prompt AND output lengths uniform in `[1, N]`
(`benchmark/datasets/common.py:56` — `np.random.randint(max(int(N*ratio),1), N+1)`
applied to both `input_lens` and `output_lens`). Every labeled-depth row
measured ~half depth on average; at `num-prompts 1` (the context sweeps) a
labeled point is ONE uniform draw — a "262144" row may have measured any depth
in `[1, 262144]`. Fleet finding doc: 3090 repo
`benchmarks/bench-depth-bug-2026-07-14.md` (server-side ground truth there: a
2K+131K+250K sweep prefilled only 121,922 tokens total).

All bench scripts now pin `--random-range-ratio 1` and record server-verified
`actual_input_tokens` per point (parsed from the bench harness's
"Total input tokens", with a `depth_shortfall` flag below 0.95× requested).
Rows produced before the pin are unrecoverable — the real depth was never
recorded — and carry `"legacy_depth_suspect": true`:

- `results.json` with pre-pin sweep rows still flagged (6):
  `coder-next-80b-4bit`, `devstral-24b-4bit`, `devstral-24b_4-bit`,
  `gemma4-26b-4bit`, `gemma4-31b-4bit`,
  `qwen3-30b-a3b_4-bit-dwq_mr4` (throughput-only). Re-measured at genuine
  depth (marker dropped): `coder-30b-4bit`, `qwen3-30b-moe-4bit`,
  `qwen3-32b-4bit`, `qwen35-27b-4bit`
- `bench_comprehensive` text outputs (7): `Coder-30B_4bit_20260412_*.txt` (2),
  `Devstral-24B_4bit_20260412_*.txt` (2), `Qwen3.5-27B_4bit_20260412_*.txt` (3)
- `bench_256k_*.log` (13, all pre-pin — the 256K harness delegates to
  `bench_all_unified.py`)
- `baselines-prepin-legacy.json` — the retired regression baselines. Post-pin
  runs send ~2× the tokens of these rows; comparing against them emits false
  REGRESSION verdicts (especially TTFT). `bench_regression.sh` reads only
  `baselines.json`, which is re-saved post-pin.
- Charts `all_models_context.png` / `all_models_concurrency.png` render
  `results.json` sweeps and are suspect wherever the underlying rows are.

Re-measured post-pin rows drop the marker and carry `sglang_version` per file —
they move two variables vs legacy rows (the pin AND the stack pin), so they are
not a depth-only correction of any legacy row.

## Instrument B — `scripts/bench/bench_long_context.py` (/v1/completions, NOT suspect)

POSTs directly to `/v1/completions` and reads `usage.prompt_tokens` — no
`benchmark.serving` call, immune to the range-ratio bug; real prefill depth is
recorded in-file per point.

- `qwen36-35b-a3b-4bit/results.json` (`long_context` array, `input_tokens`
  recorded per row)
- `long_context_*.txt` (3): `qwen36-35b-a3b-4bit/long_context_20260419T162022Z.txt`,
  `coder-30b-4bit/long_context_post_patches_20260420T174602Z.txt`,
  `qwen3-30b-moe-4bit/long_context_post_patches_20260420T180447Z.txt`
- `longctx-bisect/` receipts (server-verified `usage.prompt_tokens` per run)

**Separate, fixed estimation shortfall:** instrument B's early char-per-token
estimate under-delivered ~1.72× (a `[:input_len*4]` slice), so its LABELS
overstate depth even though the recorded `input_tokens` are true — e.g. the
qwen36 "128K" row = 76,262 actual tokens. The current script uses 6 chars/token
+ 1.1 safety (within ~5%). Read `input_tokens`, not labels, from the old rows.

## Classification of the historical "v0.5.11 fit 128K" claim

Instrument B — server-verified, NOT range-ratio-suspect. The receipt
(`qwen36-35b-a3b-4bit/results.json`) records actual `input_tokens` per row:
the "128K" label prefilled 76,262 real tokens; the deepest row ("250K" label)
prefilled 145,456 real tokens. So v0.5.11 genuinely prefilled ~145K — the
128K-capability claim survives classification, with mislabeled per-row depths.
The regression bisect this baseline fed is closed: root cause was unbounded MLX
buffer-cache growth, fixed by the patch-008 cache cap, and the current stack
re-validates 128K with true token counts (125,830 in;
`longctx-bisect/ATTRIBUTION.md`).
