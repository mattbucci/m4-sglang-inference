# qwen36 decode micro-experiments: buffer limit, SDPA block ladder, launch folding

Run date 2026-11-10. Stack: SGLang v0.5.23 + patches 001-032, MLX 0.32.4,
mlx-lm 0.32.3.dev17, mlx-vlm 0.6.5, M4 Pro 64 GB. Model
`mlx-community/Qwen3.6-35B-A3B-4bit` (preset `qwen36`): 40 layers (30
GatedDeltaNet, 10 attention), MoE with 256 experts and top-8 routing plus a
shared expert, attention with 16 query heads over 2 KV heads of dim 256.

## Method

The R9700 team's decode lane, carried over to Metal:

1. A census first. Every millisecond of a decode step is attributed to a
   family (host graph build, launch encode and commit, GPU kernels, weight
   and KV bytes read) and each family gets an owner experiment.
2. A raw-MLX microbench of the kernel or setting before any serving work
   (`scripts/bench/mlx_*.py`, receipts in `raw/`, `slack/` and the two
   `*-raw.jsonl` files).
3. Every candidate is a same-tree serving A/B: `scripts/bench/decode_ab.py`,
   3 runs per depth at 948 / 7,583 / 30,391 prompt tokens, 256 generated,
   greedy, `ignore_eos`, each arm run twice in mirrored order and pooled.
   The design's noise floor is measured (`serve-router-null/`, eleven arms
   of one build): arm medians spread 0.4 to 0.5 % (SD) and 1.5 to 1.8 % end
   to end, and one pooled pair of arms lands up to 1.0 % from another. A
   pooled difference under 1 % is not a result.
4. Deep arms at 121,606 and 251,474 prompt tokens under the long-context
   recipe (`CTX=<label+64> MEM_FRAC=0.5 CHUNKED=1024
   EXTRA_ARGS=--disable-radix-cache`) with the OOM guard, one cold request
   per arm.
5. Capability gates, depth recall and teacher-forced fidelity before a
   default changes.
6. A retained code change is a numbered opt-in patch. A neutral one is
   reverted. Every experiment ends with a disposition.
7. A flag-gated arm counts only if its server log shows the hook's
   "enabled" line. Two chains here timed flags that never installed:
   mlx-vlm checkpoints return from the runner's load before the text-path
   tail where the fusion flags were read. Patches 030 to 032 now install
   from a helper both load paths call, and each patch's unit test checks
   the call sites.

## Dispositions

| experiment | family | measured | disposition |
|---|---|---|---|
| `MLX_MAX_MB_PER_BUFFER=200` | launch commit | +4.2 to +6.1 % decode to 30K, +1.4 % at 122K, cold prefill −2 % at 122K and −5 to −7 % at 251K | retained |
| `MLX_MAX_MB_PER_BUFFER` 400 / 1000 | launch commit | equal to 200 within noise; 400 adds 0.3 GB of wired peak at 122K | 200 is enough |
| `MLX_MAX_OPS_PER_BUFFER` 100 (stock 50) | launch commit | 0.1 to 1.7 points over the buffer limit alone | not taken: small, and 200 already loses at 30K |
| `MLX_MAX_OPS_PER_BUFFER` 200 | launch commit | −17.1 % at 30K | rejected |
| `MLX_METAL_FAST_SYNCH=1` | launch commit | raw: equal to stock | rejected |
| `MLX_BFS_MAX_WIDTH` 1 / 5 / 100 / 100000 (stock 20) | graph order | raw: 1 costs 0.75 ms, 5 equal, 100 and up 0.1 to 0.4 ms slower | rejected, stock is best |
| `SGLANG_MLX_CLEAR_CACHE_STEPS=0` (stock 256) | host | no cost to 122K (−0.1 % pooled over four traces, equal wired peak); a 93 to 186 ms stall every 256 steps at 139K and 196K, 1.7 % | open, the 256K arm decides (cache-clear section) |
| `SGLANG_MLX_FUSE_SWIGLU=1` | launches | never installed on qwen36 (read on the mlx-lm load path, and it matches mlx-lm's `SwitchGLU` class only); the −0.3 to +0.6 % was noise | not measured on qwen36, coder-30b arm queued |
| `mx.compile` around the MoE and DeltaNet blocks | graph build | raw: build 1.72 → 0.87 ms, step time equal | rejected, the build already overlaps the GPU |
| constant `lhs_indices` for gathered matmuls | launches | raw: 2,780 → 2,540 primitives, step time equal | rejected |
| SDPA block ladder (patch 029) | attention | 251K: 37.3 against 40.1 to 40.8 ms per token, recall 6/6; stock kept through 196,608 keys | retained, opt-in flag |
| int8 keys and values | attention | raw: 34 to 68 % slower than the stock kernel to 131K; KL 0.004 to 0.005 at 32K | closed, the kernel is compute-bound once it dequantizes |
| fused MoE router and combine (patch 030) | launches | raw: 2,780 → 2,340 primitives, −0.5 to −0.7 ms | opt-in flag |
| fused GatedDeltaNet front (patch 031) | launches | raw: 2,340 → 1,770 primitives, −0.3 ms more | opt-in flag |
| grouped decode linears (patch 032) | launches | raw: −0.2 ms more, bit-identical | opt-in flag |
| fused GatedDeltaNet back kernel | launches | raw: 9.94 ms against 9.86 for the linear pairs on the same base | not taken |
| block-max top-k for the sampled tail | sampling | raw: 242 µs against 387 µs for the full argsort | open candidate, about 1 % of a sampled step |

## Where a step goes

`serve-census-base.jsonl` (35 windows of 64 steps across the three A/B
depths, median per step) and `serve-graph/qwen36-step-primitives.json`.

| slice | ms per step |
|---|---|
| wall | 12.90 |
| launch chain (build + evaluate) | 12.49 |
| host graph build, overlapped with the previous step's GPU work | 2.22 |
| `async_eval` (encode, commit, GPU) | 10.31 |
| token wait | 0.007 |
| result decode, stream, finalize | 0.18 / 0.11 / 0.28 |

A step is 2,779 graph primitives: 321 reshapes, 301 quantized matmuls, 261
broadcasts, 191 RMS norms, 150 multiplies, 120 gathered quantized matmuls,
50 custom kernels and 10 attention calls among them. The launch microbench
(`raw/launch-microbench.md`) prices a small launch at 0.5 to 2.5 µs of
encode plus 1.5 to 2.7 µs of GPU, so the launches alone account for the 4
to 5 ms a step spends beyond its weight reads (next section). That splits
the work into three owners: how launches are committed (buffer limits),
how many launches there are (patches 030 to 032), and what the ten
attention kernels read at depth (patch 029).

## Weight reads set the floor

`raw/weight-read-census.jsonl` (`scripts/bench/mlx_weight_read_census.py`,
safetensors headers only, no GPU): the bytes of weights one decode step
streams, counting only the routed experts of a MoE layer, one embedding
row and no vision tower. Step time is the README decode rate at 1K.

| preset | weights read per token | step at 1K | effective read rate | step minus the read at 220 GB/s |
|---|---|---|---|---|
| `qwen36` | 1,670 MB | 12.59 ms | 133 GB/s | 5.0 ms |
| `coder-30b`, `qwen3-moe` | 1,711 MB | 12.21 ms | 140 GB/s | 4.4 ms |
| `flint` (oQ8e, at 83 tokens) | 2,673 MB | 16.16 ms | 165 GB/s | 4.0 ms |
| `nemotron-30b` | 2,027 MB | not in the README table | | |
| `qwen35-9b-8bit` | 8,434 MB | 37.31 ms | 226 GB/s | −1.0 ms |
| `devstral` | 13,919 MB | 62.11 ms | 224 GB/s | −1.2 ms |
| `qwen35`, `qwen36-27b` | 14,418 MB | 68.97 ms | 209 GB/s | 3.4 ms |
| `qwen3-32b` | 17,992 MB | 83.33 ms | 216 GB/s | 1.6 ms |

The dense presets decode at 209 to 226 GB/s of weight reads, inside the
178 to 239 GB/s the launch microbench measures for chained and independent
quantized matmuls. They sit on the memory wall, and nothing in the decode
loop can move them by more than a few percent. The MoE presets read about
a tenth as much per token and spend 4 to 5 ms of every step on something
other than weights. That margin is the launch count and the host, and it
is the whole target of this lane.

For qwen36 the read splits into GatedDeltaNet 570 MB, routed experts
566 MB, LM head 286 MB, attention 153 MB, shared expert 71 MB and router
gates 22 MB. At 220 GB/s that is 7.6 ms per token, or 132 tok/s, before
any KV read: the ceiling for launch folding on this checkpoint.

## Command-buffer limits

Raw sweep (`knob-sweep-raw.jsonl`, pipelined ms per step at depth 24 /
8,192): stock 11.15 / 12.14, 400 MB buffers 10.95 / 11.91, ops limit 10
11.85 / 12.85, ops limit 400 with unlimited MB 10.94 / 13.07,
`MLX_METAL_FAST_SYNCH=1` 11.10 / 12.18.

Why a larger buffer helps (`slack/`, step ms against a host delay injected
between steps):

| arm | 0 ms | 0.5 ms | 1 ms | 2 ms | 3 ms |
|---|---|---|---|---|---|
| stock | 11.20 | 11.52 | 12.25 | 13.42 | 14.34 |
| 400 MB buffers | 11.11 | 11.03 | 11.27 | 11.66 | 12.61 |
| ops 100, 1000 MB | 10.91 | 10.97 | 11.05 | 11.06 | 11.08 |

On this GPU (`applegpu_g16s`) stock MLX commits a command buffer after 50
operations or 50 MB of touched buffers. A qwen36 step touches enough weight
to commit early and often, so the GPU queue runs dry whenever the host is
late and every host millisecond lands on the step. Larger buffers keep
more work queued and absorb the delay.

Serving A/B, pooled over both orders, ms per token and gain in tok/s over
the same chain's base arm:

| chain | arm | 948 | 7,583 | 30,391 |
|---|---|---|---|---|
| `serve-knobs/` | base | 12.19 | 12.92 | 15.06 |
| | 200 MB | 11.49 (+6.1 %) | 12.17 (+6.1 %) | 14.46 (+4.2 %) |
| | 400 MB | +5.8 % | +6.0 % | +4.5 % |
| | 1000 MB | +6.2 % | +6.0 % | +4.9 % |
| | ops 100, 1000 MB | +6.7 % | +7.7 % | +5.0 % |
| | ops 100, 400 MB | +6.3 % | +6.8 % | +4.9 % |
| | ops 200, 1000 MB | +2.8 % | +0.4 % | −17.1 % |
| `serve-flags/` | base | 12.36 | 13.02 | 15.25 |
| | clear-cache off | +0.6 % | +0.9 % | +1.1 % |
| | fused SwiGLU (hook not installed) | +0.6 % | −0.3 % | +0.5 % |
| | 400 MB | +7.4 % | +6.1 % | +4.8 % |
| `serve-sampled/` (temperature 0.7) | base | 12.44 | 13.08 | |
| | 200 MB | +6.3 % | +5.3 % | |
| | 400 MB | +5.3 % | +5.5 % | |
| `serve-prefill/` | base | | 12.56 | 14.73 |
| | 200 MB | | +4.9 % | +4.1 % |
| | 150 MB | | +4.5 % | +2.8 % |
| | 400 MB | | +3.7 % | +3.5 % |

Cold prefill is unchanged to 30K (9.73 / 50.93 s base, 9.76 / 50.52 s at
200 MB). Text hashes are equal between every buffer arm and its base: the
setting changes when work is committed, not what is computed.

Deep arms, one cold request each:

| arm | prompt tokens | ms per token | tok/s | cold TTFT s | wired peak GB |
|---|---|---|---|---|---|
| base | 121,606 | 23.56 | 42.4 | 385.4 | 25.76 |
| 200 MB | 121,606 | 23.23 | 43.1 | 376.6 | 27.45 |
| 400 MB | 121,606 | 23.24 | 43.0 | 376.0 | 27.79 |
| base | 251,474 | 40.83 | 24.5 | 1528.7 | 28.11 |
| 200 MB | 251,474 | 40.06 | 25.0 | 1434.3 | 29.20 |

At depth the step is attention-bound and the buffer limit is worth 1 to 2
%, inside the run-to-run spread there (two 200 MB arms at 251K differ by
1.6 %). The prefill gain grows with depth instead: 2.3 % at 122K and 5 to
7 % at 251K across four arms (1426 to 1451 s against 1529 s). The cost is
1.1 to 1.7 GB of wired peak, against a 51.8 GB limit.

## SDPA block ladder (patch 029)

MLX's two-pass decode attention kernel splits the keys into blocks and
stops raising the count at 1024, so blocks grow with the context. MLX reads
`MLX_SDPA_BLOCKS` on every kernel call, so the runner can set it per decode
step from the depth of the request.

Raw kernel, 10 layers, K/V sliced from a 262,208-key buffer the way the
serving cache hands them over (`scripts/bench/mlx_sdpa_layout_probe.py`,
`raw/sdpa-layout-*.jsonl`). "Joint" evaluates the ten layers in one
`mx.eval`, "per layer" evaluates them one at a time, which is how a decode
step meets them between its other work.

| keys | joint: stock | joint: 2048 | per layer: stock | per layer: 1536 | per layer: 2048 | per layer: 2560 |
|---|---|---|---|---|---|---|
| 131,072 | 12.61 | 13.38 | 14.02 | 15.20 | 15.49 | 16.56 |
| 163,840 | | | 17.84 | 18.12 | 18.71 | 19.18 |
| 196,608 | 21.68 | 20.05 | 21.23 | 21.14 | 21.20 | 21.92 |
| 212,992 | | | 23.27 | 23.00 | 22.95 | 23.77 |
| 229,376 | | | 26.43 | 24.42 | 24.69 | 25.26 |
| 245,760 | | | 30.39 | 26.68 | 26.44 | 26.64 |
| 251,475 | 37.13 | 25.42 | 32.43 | 28.19 | 27.80 | 27.09 |
| 262,144 | 41.45 | 26.12 | 38.43 | 30.48 | 28.88 | 28.26 |

Exact-size tensors time the same as the buffer slice, so the cache layout
is not a factor. The evaluation shape is: several two-pass kernels queued
back to back lose far more on large blocks than one kernel alone. The
joint shape predicts 11.7 ms from 2048 blocks at 251,475 keys and a
crossover near 139K; the per-layer shape predicts 4.6 ms and a crossover
past 196,608. Serving shows 2.8 to 3.9 ms. The ladder is sized from the
per-layer shape: stock through 196,608 keys, then
`512 * ceil(keys / 65,536)`, which is 2048 blocks to 262,144 keys.

| arm at 251,474 prompt tokens | ms per token | tok/s | cold TTFT s | wired peak GB | text hash |
|---|---|---|---|---|---|
| base | 40.83 | 24.5 | 1528.7 | 28.11 | d60f0a38 |
| 200 MB | 40.06 | 25.0 | 1434.3 | 29.20 | d60f0a38 |
| 200 MB, ladder flag set but hook not reached | 40.71 | 24.6 | 1451.0 | 29.47 | d60f0a38 |
| 200 MB, `MLX_SDPA_BLOCKS=2048` pinned | 36.89 | 27.1 | 1434.5 | 29.68 | f0e76c51 |
| 200 MB, ladder flag | 37.28 | 26.8 | 1426.0 | 29.44 | f0e76c51 |

The ladder arm logs `SDPA block ladder: 251475 keys -> 2048 blocks` and
produces the pinned arm's text. Against the no-settings base it is −8.7 %
ms per token; against the two buffer-only arms −6.9 and −8.4 %. The block
count changes the order of a float sum, so the text differs from the
1024-block arms from some token on; both texts are greedy continuations
that differ at bf16 rounding scale.

Depth recall with the ladder engaged (`scripts/eval/probe_depth_recall.py
--labels 215040`, `recall/`): 6/6 at 245,656 prompt tokens, the same as the
buffer-only arm.

Two things in `serve-256k/` and `serve-ladder/` come from earlier builds of
the patch. The first ladder arm did nothing: the hook sat only on the
batched decode path, and qwen36 decodes through the serial path because
mlx-vlm's attention is not batchable there. That arm is kept as
`deep256-ladder-noop`, a second sample of the buffer-only configuration;
the patch now asks on both paths and its unit test fails on the old tree.
The second build switched to 1536 blocks at 139,264 keys, the joint
shape's crossover. One request decoding across that point
(`serve-ladder/cross139-ladder.trace.json`, 138,991 prompt tokens, 800
generated, `scripts/bench/decode_step_trace.py`) ran 25.50 ms per token
before and 25.56 ms after: no stall on the change, and no gain, which is
what the per-layer shape says. The 256K arm above is unaffected, both
builds pick 2048 blocks there.

The current build crosses at 196,608 keys. One request decoding across it
(`serve-ladder/cross196-ladder.trace.json`, 196,298 prompt tokens, 800
generated, log line `SDPA block ladder: 196609 keys -> 2048 blocks`) ran
31.08 ms per token over the 309 steps before and 30.70 ms over the 476
steps after, −1.2 %, with no stall on the change, 0 tracebacks and a
27.91 GB wired peak. The per-layer probe says 21.23 against 21.20 ms of
attention at that depth, so the threshold sits where the two ladders meet.

Both traces show something else: a step that takes 93 to 186 ms longer
than its neighbours every 256 decode steps (at 139,491 and 139,729; at
196,484, 196,740 and 196,973), 0.43 and 0.52 ms per token averaged in.
That is the runner's periodic `mx.clear_cache()`; see the cache-clear
section.

At 251,474 tokens the ten attention layers read 5.15 GB of keys and values
per token. The ladder arm spends about 26 ms above the short-context
floor, about 200 GB/s, the rate the kernel holds at 131K. Past the ladder
the deep step is bound by how fast this machine reads the cache; what is
left is the launch-bound floor, which every depth pays.

## Periodic cache clear

The runner calls `mx.clear_cache()` every `SGLANG_MLX_CLEAR_CACHE_STEPS`
decode steps (stock 256). `scripts/common.sh` also caps the buffer cache
(`SGLANG_MLX_CACHE_LIMIT_GB`, 4 GB), so the clear is not what bounds
memory on this stack; it only empties a cache that the cap already holds
under 4 GB. What the clear costs depends on the depth.

| depth | evidence | cost of the clear |
|---|---|---|
| 948 / 7,583 / 30,391 | `serve-flags/`, A/B with the setting at 0 | 0.6 / 0.9 / 1.1 % of a step, at the A/B noise floor |
| 122K | `serve-clear/c122-*`, four 1,100-token traces in mirrored order | one 15 ms stall at the first clear, none at the next three: 0.014 ms per token, 0.06 % |
| 139K | `serve-ladder/cross139-ladder.trace.json`, stalls of 178 and 156 ms | 0.43 ms per token, 1.7 % |
| 196K | `serve-ladder/cross196-ladder.trace.json`, stalls of 186, 127 and 93 ms | 0.52 ms per token, 1.7 % |

At 122K the two arms are the same request to the token: 23.42 and 23.40 ms
per token with the clear, 23.34 and 23.42 without, pooled −0.10 %, and a
wired peak of 27.03 to 27.15 GB in all four (`serve-clear/c122-summary.txt`).
The runner's step counter is global, so the first clear of a trace lands
where earlier requests left the count; at 122K that step takes 38 to 39 ms
against 23.5 and the later clears cost nothing. The repeating stall is in
the 139K and 196K traces only.

## Launch folding (patches 030, 031, 032), raw MLX

`scripts/bench/mlx_decode_router_probe.py`, the full model outside the
server, pipelined ms per step, arms in both orders (`raw/router-*.jsonl`).

| arm | primitives per step | depth 24 | depth 8,192 |
|---|---|---|---|
| stock | 2,780 | 10.94 to 11.11 | 11.76 to 12.05 |
| route constant (upper bound for the router) | 2,460 | 10.01 to 10.05 | 11.02 to 11.08 |
| fused route kernel | 2,620 | 10.61 | 11.58 |
| fused combine kernel | 2,500 | 10.62 | 11.61 |
| route + combine (patch 030) | 2,340 | 10.34 to 10.40 | 11.31 to 11.37 |
| + GatedDeltaNet front (patch 031) | 1,770 | 10.03 to 10.07 | 11.02 to 11.07 |
| + front + back kernel | 1,620 | 9.94 | 10.87 to 10.90 |
| + front + linear pairs (patch 032) | 1,770 | 9.86 to 9.87 | 10.83 to 10.89 |

The three patches together take 1.1 to 1.2 ms off an 11 to 12 ms step, 10
%. The four-way fold mlx-vlm ships cannot take the router gates: the
mlx-community 4-bit build keeps `mlp.gate` and `mlp.shared_expert_gate` at
8 bits, so they never match the 4-bit linears beside them. Patch 032
groups by bit width instead.

## Closed

- **int8 KV.** `raw/sdpa-kv.jsonl`, `raw/kv-int8-fidelity.jsonl`. int8
  keys with bf16 values run the 10 layers in 4.26 ms at 32,768 keys and
  17.88 ms at 131,072 against 3.19 and 12.65 ms for the stock kernel; int8
  keys and values 4.97 and 21.31 ms. The same hand-written kernel on bf16
  takes 4.27 and 18.46 ms, so the bytes saved buy nothing: the kernel is
  compute-bound, and the stock kernel is the faster code. Fidelity at
  32,768 keys: KL 0.0039 to 0.0054, top-1 agreement 0.969 to 0.977.
- **Compiled blocks.** `compile-probe-raw.jsonl`. Graph build drops from
  1.72 to 0.87 ms and the step does not move (11.20 / 12.30 against 11.17
  / 12.14 ms), because the build already runs while the GPU works on the
  previous step.
- **Sampled tail.** A sampled step sorts the 248,320-entry vocabulary (87
  µs encode + 300 µs GPU, greedy argmax is 10 + 59 µs). A block-max top-20
  kernel does it in 94 + 148 µs. Not built: about 1 % of a sampled step.

## Receipts

| directory | contents |
|---|---|
| `raw/` | kernel and launch microbenches, attention layout probe, router probe arms, int8 fidelity, traversal-width sweep, patch gates and unit-test output for 029 to 032 |
| `slack/`, `knob-sweep-raw.jsonl`, `compile-probe-raw.jsonl` | raw setting sweeps |
| `serve-census-base.jsonl`, `serve-graph/`, `serve-profile/` | step census, primitive counts, graph-build profile |
| `serve-knobs/`, `serve-flags/`, `serve-sampled/`, `serve-prefill/` | serving A/B chains to 30K |
| `serve-128k/`, `serve-256k/` | deep arms with memory lines |
| `serve-ladder/` | threshold-crossing trace |
| `serve-clear/` | cache-clear traces at 122K with memory lines |
| `serve-router-null/` | eleven arms of one build (launch-folding flags set, hooks not installed): the A/B design's noise floor |
| `recall/` | depth recall at the 215,040 label |
