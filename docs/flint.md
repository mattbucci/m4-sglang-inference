# Flint (`flint`)

[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) is a 39.8B-total / ~2.5B-active
MoE for coding agents: 49 layers (2 dense + 47 MoE, 256 routed experts top-6 plus a shared
expert), 35 sliding-window (513) + 14 full-attention layers, learned attention sinks, NoPE on
the full layers, 262K context, Apache-2.0. It runs through mlx-lm's `flint` model plus patch
019; the oMLX `oQ*e` checkpoints are converted through mlx-vlm's model class, hence the
patch's checkpoint-layout adapter.

| Variant | bpw | Resident | Decode tok/s @83 / @3.5K / @13.9K / @27K | Prefill tok/s @3.5K / @13.9K / @27K | Validator / codegen / thinking | Greedy vs mlx-lm |
|---------|:---:|:--------:|:----------------------------------------:|:-----------------------------------:|:------------------------------:|:----------------:|
| **`flint-oQ8e`** (default) | 8.5 | 42.3 GB | 61.9 / 59.1 / 54.4 / 49.3 | 926 / 809 / 634 | 3/3 · STRONG · VERIFIED | identical |
| `flint-oQ6e` | 6.6 | 32.9 GB | 64.8 / 61.2 / 56.0 / — | 760 / 638 / — | 3/3 · STRONG · VERIFIED | identical |
| `flint-oQ4e` | 4.7 | 23.5 GB | 70.5 / 67.5 / 61.5 / — | 708 / 595 / — | 3/3 · STRONG · VERIFIED | identical |

Measured radix-off (oQ8e at the preset's `CHUNKED=1024`, 2 GB MLX buffer cache,
`MAX_RUNNING=4`; oQ6e/oQ4e at `CHUNKED=4096` with the 4 GB cache). The shipped preset runs
the radix cache, which keeps prefill where it is and costs ~12% of decode (50.3 / 47.0 /
43.1 tok/s at 3.5K / 13.9K / 27K) because the hybrid `no_buffer` bookkeeping turns the
overlap schedule off; turn 2 of a 13.4K-token conversation then starts in 0.6 s instead of
19 s (`benchmarks/flint/2026-10-31/prefix-cache/`). A 27K prefill at chunk 1024 leaves 17 GB
kernel-available and never leaves pressure level "normal"; four concurrent requests decode
at 123 tok/s aggregate (batched output is not bit-identical to single-request output, the
usual non-batch-invariance of batched kernels).

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
  cap. The oQ6e/oQ4e sweep on the depth bench and this cell is the remaining Flint item.
