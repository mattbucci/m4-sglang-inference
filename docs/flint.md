# Flint (`flint`)

[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) is a 39.8B-total / ~2.5B-active
MoE for coding agents: 49 layers (2 dense + 47 MoE, 256 routed experts top-6 plus a shared
expert), 35 sliding-window (513) + 14 full-attention layers, learned attention sinks, NoPE on
the full layers, 262K context, Apache-2.0. It runs through mlx-lm's `flint` model plus patch
019; the oMLX `oQ*e` checkpoints are converted through mlx-vlm's model class, hence the
patch's checkpoint-layout adapter.

## Checkpoints

The preset as it ships (radix cache on, `MAX_RUNNING=4`, `CHUNKED=1024`, 2 GB MLX buffer
cache, patches 036 to 038, display on), one cold request per depth with a 256-token reply
at the checkpoint's own sampling (`benchmarks/perf/2026-11-10/flint/preset/`):

| Checkpoint | bpw | On disk | Wired peak | Decode tok/s @1K / 7.6K / 30.4K | Prefill tok/s @1K / 7.6K / 30.4K | TTFT @30.4K | Follow-up TTFT @3.8K | KL to oQ8e / same argmax |
|------------|:---:|:-------:|:----------:|:-------------------------------:|:--------------------------------:|:-----------:|:--------------------:|:------------------------:|
| **`mlx-community/flint-oQ8e`** (default) | 8.5 | 42.3 GB | 52.2 GB | 62.7 / 58.4 / 49.2 | 795 / 675 / 487 | 62 s | 0.22 s | reference |
| `mlx-community/flint-oQ6e` | 6.6 | 32.8 GB | 42.7 GB | 66.8 / 62.4 / 52.0 | 833 / 744 / 528 | 58 s | 0.22 s | 0.051 / 93.7 % |
| `RecursiveSI/flint-int4` | 4.54 | 22.6 GB | 32.6 GB | 96.2 / 88.8 / 67.0 | 883 / 795 / 540 | 56 s | 0.17 s | 0.218 / 83.0 % |

- **Gates:** every checkpoint passes validator 3/3, codegen STRONG and thinking VERIFIED;
  receipts for int4 and oQ8e on the shipping flags are in
  `benchmarks/perf/2026-11-10/flint/gates/`. Greedy output of the oQ checkpoints is
  identical to mlx-lm's. `mlx-community/flint-oQ4e` (4.7 bpw, 23.5 GB) passes the same
  gates and has no row on the shipping flags; int4 is the 4-bit checkpoint the published
  engines use.
- **Choosing one:** `MODEL=RecursiveSI/flint-int4 scripts/launch.sh flint` decodes 55 %
  faster than oQ8e at 1K and 36 % faster at 30K and wires 19.6 GB less. Teacher-forced NLL
  on held-out prose and code is equal on the three (2.133 to 2.146, inside one standard
  error). oQ6e is a close copy of oQ8e, inside the band of the 4-bit Qwen builds against
  their 8-bit references. int4 is a different quantization: it disagrees with oQ8e on 17 %
  of greedy tokens on text it predicts equally well, and the BF16 weights that would say
  which is closer do not fit in 64 GB. The default stays oQ8e because the agentic cell
  below was scored on it; an agentic cell on int4 decides a switch.
- **Concurrency:** int4 decodes at 103 / 136 / 182 tok/s aggregate for 1 / 2 / 4 streams
  (batched output is not bit-identical to single-request output, the usual
  non-batch-invariance of batched kernels).

## Against the published M4 Pro engines

Two engines publish Flint numbers for this chip, both on a MacBook Pro M4 Pro with the
20-core GPU and 48 GB, on `RecursiveSI/flint-int4`, each next to llama.cpp: Recursive's
[rec](https://github.com/recursive-oss/rec) and
[Kapitsa](https://github.com/nace-ai/kapitsa/pull/290). They are this preset's reference
engines: a number already reached on the same GPU. The same sweep here, one request slot,
radix cache off, `CTX=131136 MEM_FRAC=0.5`, one cold request per depth, a 256-token reply
at temperature 1.0, the display asleep (`benchmarks/perf/2026-11-10/flint/quiet/`):

| Prompt tokens | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|---------------|---:|---:|---:|---:|----:|----:|----:|-----:|
| Decode tok/s, this stack | 102.5 | 101.7 | 98.4 | 92.8 | 83.3 | 71.4 | 55.3 | 37.5 |
| Decode, rec | 126.5 | 123.0 | 116.5 | 107.0 | 95.2 | 79.4 | 56.5 | 35.2 |
| Decode, Kapitsa | 124.3 | 121.3 | 119.2 | 113.2 | 106.2 | 85.9 | 66.2 | 45.7 |
| Decode, llama.cpp | 77.5 | 75.1 | 73.4 | 69.7 | 63.7 | 53.8 | 40.5 | 23.8 |
| Prefill tok/s, this stack | 1022 | 965 | 953 | 879 | 738 | 594 | 448 | 296 |
| Prefill, rec | 781 | 800 | 791 | 760 | 699 | 581 | 427 | 262 |
| Prefill, Kapitsa | 858 | 976 | 960 | 907 | 808 | 651 | 464 | 300 |
| Prefill, llama.cpp | 927 | 1015 | 978 | 910 | 785 | 602 | 382 | 199 |

- **Decode** is ahead of llama.cpp at every depth (131 to 157 %), ahead of rec at 128K
  (81 to 90 % of it to 32K, 98 % at 64K) and at 78 to 84 % of Kapitsa throughout.
- **Prefill** is ahead of rec at every depth (102 to 131 %) and at 91 to 99 % of Kapitsa
  past 1K. A 128K prompt reaches its first token in 439 s.
- **Follow-up turn:** 0.16 s to first token at 3.8K and 0.30 s at 15K on the preset
  (radix cache on), against 0.13 and 0.15 s for Kapitsa, 0.23 and 0.31 s for llama.cpp
  and 0.41 and 0.44 s for rec at 4K and 16K.
- **Where the gap is:** a decode step at 1K takes 9.8 ms here against 7.9 to 8.1 ms
  published, a fixed 1.7 to 1.9 ms that sits in the graph itself (the router launches,
  attention and the routed experts above their weight reads). Per token of context the
  step grows by 0.131 µs here, 0.159 for rec and 0.107 for Kapitsa, so deep context is
  already read faster than rec and Kapitsa's lead at depth is its attention read.
- **Conditions:** the MLX server shares the GPU with the desktop. With the display on the
  same sweep decodes 3.6 to 7.5 % slower; timed arms run with the display asleep and the
  GPU time of every process recorded (`scripts/bench/gpu_time.py`).
  The lane, its probes and every disposition: `experiments/12-flint-published-parity.md`
  and `benchmarks/perf/2026-11-10/flint/VERDICT.md`.

## Decode flags

- **Patch 036** (always on) keeps the frozen view of a sliding-window cache from turning
  the next token's write into a copy of both buffers.
- **Patch 037** (`SGLANG_MLX_FUSE_FLINT_MOE`, on in the preset) runs the router and the
  expert combine of a one-token step as two Metal kernels, bit-identical to the stock
  block and checked per module at install.
- **Patch 038** (`SGLANG_MLX_DECODE_SUBMIT_PARTS`, 5 in the preset) submits a decode step
  in parts so the GPU runs the front of the step while the host builds the rest.
- Patches 037 and 038 together are worth 19 to 21 % decode through 8K, 16 % at 16K, 13 %
  at 32K and 2.4 % at 128K on int4: they remove a fixed cost per step. On oQ8e patch 038
  is worth 4.4 / 2.9 / 0.4 % at 1K / 8K / 32K. Neither runs on prefill chunks, and
  prefill reads the same with and without them.
- One Metal kernel for the router logits and the routing of a block is bit-identical and
  8 to 10 % slower per step than the two launches it replaces, so the block keeps two.
- `MLX_MAX_MB_PER_BUFFER=200` is opt-in: 102.7 against 99.9 tok/s greedy at 1K on int4,
  for 6.5 % of the cold prefill at 8K and 0.5 GB more wired.

## Behaviour

The preset runs the radix cache and keeps the pipelined decode loop under it (patch 028;
the scheduler's normal loop decodes 10 % slower, `benchmarks/perf/2026-11-10/hybrid-presets/`).
Turn 2 of a 13.4K-token conversation starts in 0.6 s instead of 19 s
(`benchmarks/flint/2026-10-31/prefix-cache/`).

- **Reasoning:** Flint decides per turn whether to emit `<think>…</think>`; there is no
  template toggle. `--reasoning-parser flint` returns it as `reasoning_content` and never
  assumes a turn starts in reasoning; keep that field on assistant messages when sending the
  conversation back. Reasoning counts against `max_tokens`: give coding requests ≥ 4,096
  (Recursive recommends ≥ 32,768 for hard problems). `probe_all.sh` runs Flint's codegen
  probe with a 4,096-token budget because the 400-token default truncates its second task.
- **Tool calls:** `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
  (GLM-4.7's format without line breaks); `--tool-call-parser flint` returns structured
  `tool_calls`. Recursive's constrained grammar backend (`--grammar-backend flint`) is not
  ported. Under opencode Flint omits the `bash` tool's required `description` argument (see
  the issue tracker); the SWE-bench proxy backfills it (`BACKFILL_BASH_DESCRIPTION=1`). Under
  omp Flint fills whichever key the schema declares first: with omp's `i` intent key first it
  sends the hashline `edit` as `i` instead of `input`, with `i` moved last (the proxy's
  `INTENT_LAST=1`, the Flint lane default in `smoke.sh`) every call carries its payload in the
  tool's own key and `i` is simply omitted on bash/read/grep, which omp accepts. The
  response-side repairs (`BACKFILL_INTENT`, `REMAP_INTENT`) stay off: their rewrites get
  copied into later calls and overrun omp's 200-character `i` limit.
- **Sampling:** the preset applies the checkpoint's `generation_config` (temperature 1.0 /
  top-p 0.95 / top-k 20) to requests that omit sampling fields; greedy requests stay greedy
  and pass every gate, although Recursive validates only the sampled setting.
- **Prefix cache:** the 14 full layers share the pool (28 KB/token, 3.7 GB at the preset's
  131,072 slots) and the 35 windowed caches are snapshotted at the radix boundaries and
  restored on an exact hit (patch 021). The pool costs ~2 GB of headroom relative to the
  radix-off preset. Set `MAX_TOTAL_TOKENS=131072` explicitly for CTX=131072 runs
  (`CTX * MAX_RUNNING` would be a 14.7 GB pool); `EXTRA_ARGS="--disable-radix-cache"`
  restores the pool-free recompute path. Per-request KV is 28 KB/token on top of the pool.
  Every chunk stores a snapshot of the 35 windowed caches; patch 039 schedules its
  evaluation, so a chunk runs without a wait per layer. Cold prefill under the radix
  cache reads 880 tok/s at 7.6K and 637 to 642 at 30.4K on int4, the display asleep
  (`benchmarks/perf/2026-11-10/flint/snapshot-eval/`), against 879 at 8.2K and 594 at
  32.8K with the radix cache off.
- **Agentic cell:** SWE-bench Lite, the 26-instance scorecard set on the omp harness,
  bash-only, sealed, 64K with the radix cache, 1800 s per instance: **5/26 non-empty
  patches, 0/26 resolved** on the local scorer (three apply and fail their tests, two are
  on astropy instances the box cannot build), every instance at the cap, 13 h wall
  (`evals/swebench/runs/flint-oQ8e-26inst-omp-bashonly-remap-sealed-2026-11-08/`); qwen36
  on the same harness engages 26/26 in 99 min. Flint never ends its turn under omp and on
  14 of the 21 empties never attempts an edit, reading the code in `sed`/`grep` windows
  to the cap; runaway turns 4 in 4443 responses. The three 3-instance smokes behind the
  configuration (`…/flint-oQ8e-omp-*-64k-sealed-2026-11-08/`): the full tool set lands
  edits only with the proxy's `REMAP_INTENT` (Flint sends omp's hashline edit as `i`) and
  still routes 9 of 10 calls through bash, so the cell runs bash-only. The request-side
  smoke (`…/flint-oQ8e-omp-ilast-64k-sealed-2026-11-09/`, `i` moved last, no rewrites)
  is the lane configuration going forward: 366 of 366 calls well-formed, 0 omp
  rejections, 10 of 22 edits applied, same 1/3 engagement with every instance at the
  cap. An agentic cell on int4 is the remaining Flint item: it decides the default
  checkpoint.
