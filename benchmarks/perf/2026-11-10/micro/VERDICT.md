# qwen36 decode micro-experiments: buffer limit, SDPA block ladder, launch folding, prefix-hit positions

Run date 2026-11-10. Stack: SGLang v0.5.23 + patches 001-032 (033 lands in
the prefix-hit section and later arms run with it), MLX 0.32.4,
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
| `MLX_MAX_MB_PER_BUFFER=200` | launch commit | +4.2 to +6.1 % decode to 30K, +1.4 % at 122K, cold prefill −2 % at 122K and −5 to −7 % at 251K; capability and cache gates pass, token ids and logprobs equal to stock on the resume probe | retained |
| `MLX_MAX_MB_PER_BUFFER` 400 / 1000 | launch commit | equal to 200 within noise; 400 adds 0.3 GB of wired peak at 122K | 200 is enough |
| `MLX_MAX_OPS_PER_BUFFER` 100 (stock 50) | launch commit | 0.1 to 1.7 points over the buffer limit alone | not taken: small, and 200 already loses at 30K |
| `MLX_MAX_OPS_PER_BUFFER` 200 | launch commit | −17.1 % at 30K | rejected |
| `MLX_METAL_FAST_SYNCH=1` | launch commit | raw: equal to stock | rejected |
| `MLX_BFS_MAX_WIDTH` 1 / 5 / 100 / 100000 (stock 20) | graph order | raw: 1 costs 0.75 ms, 5 equal, 100 and up 0.1 to 0.4 ms slower | rejected, stock is best |
| `SGLANG_MLX_CLEAR_CACHE_STEPS=0` (stock 256) | host | no cost to 122K (−0.1 % pooled over four traces, equal wired peak); a 93 to 186 ms stall every 256 steps at 139K and 196K, 1.7 % | open, the 256K arm decides (cache-clear section) |
| `SGLANG_MLX_FUSE_SWIGLU=1` | launches | never installed on qwen36 (read on the mlx-lm load path, and it matches mlx-lm's `SwitchGLU` class only); the −0.3 to +0.6 % was noise | not measured on qwen36, coder-30b arm queued |
| `mx.compile` around the MoE and DeltaNet blocks | graph build | raw: build 1.72 → 0.87 ms, step time equal | rejected, the build already overlaps the GPU |
| constant `lhs_indices` for gathered matmuls | launches | raw: 2,780 → 2,540 primitives, step time equal | rejected |
| SDPA block ladder (patch 029) | attention | 251K: 37.3 against 40.1 to 40.8 ms per token, recall 6/6; stock kept through 196,608 keys; with patches 030 to 032 on, 36.8 ms and recall 6/6 | retained, opt-in flag |
| int8 keys and values | attention | raw: 34 to 68 % slower than the stock kernel to 131K; KL 0.004 to 0.005 at 32K | closed, the kernel is compute-bound once it dequantizes |
| fused MoE router and combine (patch 030) | launches | raw: 2,780 → 2,340 primitives, −0.5 to −0.7 ms; served +5.0 to +5.4 % to 30K | opt-in flag, gates pass with all three on |
| fused GatedDeltaNet front (patch 031) | launches | raw: 2,340 → 1,770 primitives, −0.3 ms more; served +3.3 to +3.5 % to 30K | opt-in flag, gates pass with all three on |
| grouped decode linears (patch 032) | launches | raw: −0.2 ms more on top of 030 and 031, bit-identical; served alone −0.7 to −0.9 %, on top of 030 and 031 +3.1 / +2.3 / +1.6 % over three mirrored pairs, the three together +10.1 to +11.6 % to 30K | opt-in flag, worth it only with the other two |
| fused GatedDeltaNet back kernel | launches | raw: 9.94 ms against 9.86 for the linear pairs on the same base | not taken |
| fidelity of patches 030 to 032 | rounding | teacher-forced against a float32 yardstick: KL mean 0.0189 (030), 0.0206 (031), bit-identical (032), 0.0211 (all) against 0.0165 for the stock decode path and 0.0194 for the stock prefill path; through the server, 992 teacher-forced positions: perplexity 5.227 with all three against 5.244 stock, KL 0.0091 against 0.0071 between the stock hit and cold paths | pass |
| prefix-hit positions (patch 033) | correctness | a radix hit numbered its new tokens from zero on mlx-vlm Qwen checkpoints; fixed, a re-prompting decode is bit-identical over 4,598 steps (23.9 divergences per 1000 before) and a follow-up hit is at KL 0.0064 from cold (0.0658 before) | retained, always on |
| layer RMSNorm and residual-add fold | launches | raw knock-out of all 80 layer norms: 0.07 to 0.19 ms on the stock graph, −0.05 to 0.08 ms with patches 030 to 032 installed | not built, nothing to recover |
| further launch folding on qwen36 | launches | with 030 to 032 a raw step is 10.15 ms and 1,850 primitives against 7.9 to 8.2 ms of weight and state traffic at 220 GB/s; no family holds more than 1 ms above its reads | lane closed on this checkpoint |
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
| 200 MB, ladder flag, patches 030 to 032 | 36.77 | 27.2 | 1410.6 | 29.73 | 85e8ca8a |

The ladder arm logs `SDPA block ladder: 251475 keys -> 2048 blocks` and
produces the pinned arm's text. Against the no-settings base it is −8.7 %
ms per token; against the two buffer-only arms −6.9 and −8.4 %. The block
count changes the order of a float sum, so the text differs from the
1024-block arms from some token on; both texts are greedy continuations
that differ at bf16 rounding scale.

Depth recall with the ladder engaged (`scripts/eval/probe_depth_recall.py
--labels 215040`, `recall/`): 6/6 at 245,656 prompt tokens, the same as the
buffer-only arm.

The last row is the whole retained set at this depth (`deep256-all`): the
three fusion hooks log their "enabled" lines, the ladder logs 2048 blocks,
recall is 6/6 at 245,656 prompt tokens, 0 tracebacks. Its 0.5 ms under the
ladder arm is inside the 0.65 ms the two buffer-only samples differ by, so
at this depth the fusions are free and no more: the 1.1 ms they take off a
short-context step is launch time, and a 251K step spends 26 ms reading
keys and values. The fused router changes rounding, so the text differs
from the ladder arm's.

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

## Launch folding, served

`serve-router/`: eleven arms in mirrored order on one tree, every arm with
`MLX_MAX_MB_PER_BUFFER=200`, each flagged arm's server log carrying its
hook line (40 MoE blocks patched, 30 GatedDeltaNet layers patched, grouped
linears hooked). Median ms per token pooled over both arms of a group; the
null run puts the noise floor of this design at 1 %.

| group | 948 | 7,583 | 30,391 | gain over base | tok/s |
|---|---|---|---|---|---|
| base | 11.637 | 12.325 | 14.666 | | 85.9 / 81.1 / 68.2 |
| block ladder (inert at these depths) | 11.763 | 12.306 | 14.727 | −1.1 / +0.2 / −0.4 % | 85.0 / 81.3 / 67.9 |
| fused router (030) | 11.087 | 11.689 | 13.959 | +5.0 / +5.4 / +5.1 % | 90.2 / 85.6 / 71.6 |
| fused front (031) | 11.244 | 11.929 | 14.183 | +3.5 / +3.3 / +3.4 % | 88.9 / 83.8 / 70.5 |
| grouped linears (032) alone | 11.722 | 12.439 | 14.789 | −0.7 / −0.9 / −0.8 % | 85.3 / 80.4 / 67.6 |
| all three | 10.430 | 11.105 | 13.320 | +11.6 / +11.0 / +10.1 % | 95.9 / 90.0 / 75.1 |

The served gain matches the raw probe: 1.2 to 1.35 ms off a step at every
depth, so the saving is a fixed launch cost and its share shrinks as the
attention kernels grow. The two arms of each group agree within 0.14 ms.
Wired memory is 24.4 to 24.6 GB in every arm. The grouped linears alone
sit at the noise floor on the slow side; their raw gain was measured on
top of the other two patches, where the fused MoE block folds its two
gates through the same helper. Generated text is identical to base in the
grouped-linears and ladder arms and differs in the fused-router,
fused-front and all-three arms, each of which reproduces itself across
its two arms.

`serve-router3/` isolates the grouped linears where they are meant to run:
three mirrored pairs on the tree with patch 033, `ff` = fused router and
front, `all` = `ff` plus the grouped linears, every arm with its hook
lines.

| group | 948 | 7,583 | 30,391 | gain over `ff` | tok/s |
|---|---|---|---|---|---|
| `ff` (030 + 031) | 10.536 | 11.154 | 13.434 | | 94.9 / 89.7 / 74.4 |
| `all` (030 + 031 + 032) | 10.223 | 10.902 | 13.227 | +3.1 / +2.3 / +1.6 % | 97.8 / 91.7 / 75.6 |

Every `all` arm is faster than every `ff` arm at each depth (10.12 to
10.29 against 10.49 to 10.57 ms at 948, 13.15 to 13.28 against 13.35 to
13.46 at 30,391), a fixed 0.21 to 0.31 ms per step. The six arms produce
the same text at every depth and run, which is the served form of the
bit-identity the fidelity probe shows. Patch 032 pays only next to the
fused MoE block and front; alone it is neutral to slightly slow.

One thing these chains carry that the tables do not show: the bench
builds each depth's prompt from one seeded sentence stream, so the 7,583
prompt extends the 948 prompt and the 30,391 prompt extends the 7,583
one. The first request at the two deeper depths is therefore a prefix hit
with 896 and 7,552 cached tokens, and its time to first token covers the
uncached tail only. Before patch 033 those tails were numbered from zero,
which is why the first-run hashes at 7,583 and 30,391 differ between
`serve-router/` and `serve-router3/` while the cold 948 hash of the
all-three arm is the same in both. With the patch the repeat runs at 948 (full prefix hits)
reproduce the cold run's text in all six arms; in all eleven arms of
`serve-router/` they did not. Step timings are unaffected, a hit does the
same work either way. The deep single-request arms (122K, 251K) are true
cold prefills.

## What is left of a step

`raw/census-fused.json` is the step census of `../census-qwen36.json`
rerun with patches 030 to 032 installed: the full model outside the
server, pipelined, stock buffer limits. A knock-out replaces one module
family with a pass-through and reports how much faster the step gets.
The run drifts by 0.2 to 0.3 ms between its first and last pipelined
measurement, so read differences under that as zero.

| | stock, depth 24 / 8,192 | patches 030 to 032, depth 24 / 8,192 |
|---|---|---|
| primitives per step | 2,780 | 1,850 |
| pipelined step, ms | 11.15 / 12.15 | 10.15 / 11.25 |
| host graph build (overlapped), ms | 1.69 / 1.66 | 2.00 / 2.02 |

| family | knock-out, stock | knock-out, folded | bytes moved per token | at 220 GB/s | left above the reads |
|---|---|---|---|---|---|
| GatedDeltaNet, 30 layers | 3.93 / 4.04 ms | 3.74 / 3.88 ms | 570 MB weights + 63 to 126 MB of recurrent state read and written | 2.9 to 3.2 ms | 0.6 to 0.9 ms |
| MoE block, 40 layers | 4.68 / 4.86 ms | 4.01 / 4.31 ms | 659 MB | 3.0 ms | 1.0 ms |
| of which the routed experts | 2.65 / 2.79 ms | 3.02 / 3.29 ms | 566 MB | 2.6 ms | 0.5 ms |
| attention, 10 layers | 1.08 / 1.98 ms | 0.95 / 2.14 ms | 153 MB + keys and values | 0.7 ms + KV | 0.25 ms at depth 24 |
| layer RMSNorm, 80 | 0.07 / 0.19 ms | −0.05 / 0.08 ms | none | | nothing |
| embedding, final norm, head, loop | 1.28 / 1.30 ms | 1.29 / 1.29 ms | 286 MB | 1.3 ms | nothing |

Knock-outs are marginal costs in a pipelined loop and do not add up
exactly (the routed experts cost more to remove after folding than
before, while the block that contains them costs less). The shape is
still clear. The folded step is 10.15 ms against 7.9 to 8.2 ms of bytes
that must cross the memory bus at the rate the dense presets hold. The 2
ms that remain are spread over 80 module instances at 20 to 30 µs each,
and no family holds more than 1 ms. The head sits on its read. The 80
layer norms are free, so folding them into the residual add recovers
nothing. Another launch-folding kernel on this checkpoint would be
chasing 2 to 3 % per family. The lane is closed here: what moves qwen36
further at short context is fewer bytes per token, which is a
quantization decision with a quality cost, not a decode-loop change.

## Decode fidelity: a float32 yardstick

A served A/B of a decode-only kernel cannot grade fidelity: the arms part
ways at the first flipped token and text hashes then compare different
continuations. `scripts/eval/decode_path_precision.py` is teacher-forced
instead. It takes a served greedy dump (`serve-dfid/base.json`: 24
documents, 12 prose and 12 code, 4,598 decoded steps) and recomputes the
next-token distribution at every step on paths that are the same function
in exact arithmetic. The yardstick is one batched forward with float32
activations over the unchanged 4-bit weights. KL is taken from the
yardstick to each path; a far step is one with KL above 0.5.

`precision/stock.json`:

| path | KL mean | KL median | KL p99 | far steps per 1000 | top token differs per 1000 |
|---|---|---|---|---|---|
| one-token steps, float32 | 1.4e-5 | 8.7e-10 | 2.2e-5 | 0 | 0.65 |
| batched forward, bf16 (the prefill path) | 0.0194 | 5.0e-4 | 0.113 | 4.1 | 22.4 |
| one-token steps, bf16 (the decode path) | 0.0165 | 4.8e-4 | 0.163 | 5.0 | 21.3 |
| the served top-20 | 0.0099 | 3.9e-4 | 0.144 | 5.2 | 21.3 |

The float32 step path lands on the yardstick, so the decode path computes
the function the prefill path computes: there is no logic defect in it.
What remains is bf16 rounding of the activations, the same size on either
path: about 2 % of greedy steps pick a different token than the float32
model would, and one step in a thousand picks a token the float32 model
gives under 1 %. Every far step is in prose; the code documents have none.
A greedy prose continuation on this checkpoint is therefore not a stable
object, and a text hash that differs between two arms says nothing by
itself.

### Fusion fidelity gate

The same probe with `--fuse`, which installs the decode fusions before the
bf16 step path runs (`precision/router.json`, `front.json`, `linears.json`,
`all.json`):

| bf16 decode path with | KL mean | KL median | KL p99 | far steps per 1000 | top token differs per 1000 |
|---|---|---|---|---|---|
| stock | 0.0165 | 4.8e-4 | 0.163 | 5.0 | 21.3 |
| fused router (030) | 0.0189 | 4.8e-4 | 0.119 | 3.5 | 16.5 |
| fused front (031) | 0.0206 | 4.9e-4 | 0.212 | 6.3 | 23.5 |
| grouped linears (032) | 0.0165 | 4.8e-4 | 0.163 | 5.0 | 21.3 |
| all three | 0.0211 | 4.5e-4 | 0.133 | 3.9 | 20.7 |
| for scale: stock prefill path | 0.0194 | 5.0e-4 | 0.113 | 4.1 | 22.4 |

The grouped linears are bit-identical to stock. The fused router and front
move individual steps (their kernels accumulate in float32 and round once)
but sit in the band the two stock paths already span. Patches 030 and 031
pass.

### Prefill path through the server

`scripts/eval/quant_fidelity.py --docs 4` (`fidelity/`), on the tree with
patch 033: 992 held-out positions, half prose and half code. Each position
is a radix hit that extends the previous prefix by four tokens and scores
the next token, so this instrument grades the multi-token forward a
follow-up turn uses, not the single-token decode step. Every arm carries
the buffer limit; KL and top-token agreement are against the first arm's
dump.

| arm | NLL | perplexity | KL mean | KL median | top token agrees |
|---|---|---|---|---|---|
| stock kernels | 1.6572 | 5.244 | | | |
| fused router (030) | 1.6538 | 5.227 | 0.0091 | 0.0022 | 97.5 % |
| fused front (031) | 1.6572 | 5.244 | 0 | 0 | 100 % |
| grouped linears (032) | 1.6572 | 5.244 | 0 | 0 | 100 % |
| all three | 1.6538 | 5.227 | 0.0091 | 0.0022 | 97.5 % |
| for scale: stock kernels, radix off | 1.6608 | 5.263 | 0.0071 | 0.0020 | 96.7 % |

The standard error of the NLL is 0.076 in every arm, twenty times the
largest difference between arms. Patches 031 and 032 engage on the
single-token step only and leave this path bit-identical. Patch 030 runs
on every forward; its distance from the stock kernels is the distance
between the stock hit path and the stock cold path, which is what the
radix-off arm measures (the same prefixes prefilled cold, position by
position). No arm moves perplexity.

## Served decode-fidelity arms

`scripts/eval/decode_fidelity.py` (`serve-dfid/`): the arm decodes the 24
reference prompts greedily; at a step where its top token differs from the
reference it records the divergence and re-prompts with the reference
tokens, which the radix cache serves as a prefix hit. `--restart-every 48`
re-prompts on a schedule, `--split-prefill 37` starts the decode from a
hit. Files without a suffix ran on patches 001-032, files ending `-p033` on
the tree with patch 033.

| arm | differs from the reference arm by | divergences per 1000 | KL mean | with 033: divergences per 1000 | with 033: KL mean |
|---|---|---|---|---|---|
| `rerun` | nothing | 0 | 0 | 0 | 0 |
| `noknob` | stock buffer limit | 0 | 0 | | |
| `sdpa4` | attention summed in 4 blocks | 0 | 0 | | |
| `lin` | grouped linears (032) | 0 | 0 | | |
| `restart` | re-prompt every 48 steps, radix hit | 23.9 | 0.0268 | 0 | 0 |
| `restart-noradix` | the same schedule as cold prefills | 14.8 | 0.0077 | | |
| `floor` | decode starts from a radix hit | 40.2 | 0.0347 | 18.1 | 0.0084 |
| `chunk128` | cold prefill on a 128-token grid | 36.1 | 0.0278 | 19.4 | 0.0088 |
| `fused` | fused router (030) | 37.8 | 0.0314 | | |
| `front` | fused front (031) | 32.8 | 0.0289 | | |
| `ff` | 030 + 031 | 38.3 | 0.0298 | | |
| `all` | 030 + 031 + 032 | 38.3 | 0.0298 | 23.9 | 0.0119 |

The buffer limit, the block count of the attention sum and the grouped
linears are exact. The row that did not fit was `restart`: a re-prompt
served from the radix cache drifted more than the same re-prompt prefilled
cold, though a hit that forwards one token repeats the decode step it
replaces. Every arm that diverges also resyncs through a hit, so the left
columns of the lower rows carry that excess too.

## Prefix-hit positions (patch 033)

`scripts/bench/mlx_prefix_tail_probe.py` isolates the hit in raw MLX: run
the one-token decode chain, restore the cache as it stood K tokens
earlier, forward those K tokens at once, and take the KL of the last
position against the chain (6 documents, 54 checkpoints per K). With the
language model's position state kept, K = 1 is bit-identical and longer
tails sit at the chunk-grid level. Cleared, as the runner clears it at
every new prefill, K = 1 already moves (`precision/tail-keep.json`,
`precision/tail-reset.json`):

| tail tokens | state kept: KL median / mean | state cleared: KL median / mean / max |
|---|---|---|
| 1 | 0 / 0 | 1.4e-3 / 0.014 / 0.30 |
| 2 | 1.8e-4 / 1.9e-3 | 2.3e-3 / 0.017 / 0.47 |
| 4 | 1.4e-4 / 5.6e-3 | 2.3e-3 / 0.028 / 0.71 |
| 8 | 2.0e-4 / 2.1e-3 | 1.8e-3 / 0.021 / 0.29 |
| 16 | 3.8e-4 / 1.9e-3 | 1.8e-3 / 0.028 / 0.62 |
| 48 | 5.1e-4 / 6.9e-3 | 2.9e-3 / 0.022 / 0.39 |

The mechanism: mlx-vlm's Qwen language models (`qwen3_5`, `qwen3_5_moe`)
add the cache offset to the positions only while `_rope_deltas` is set.
`prefill_start` clears it for every mlx-vlm model so that one request's
image positions cannot reach the next. On a prefix hit the forward then
carries only the tokens after the cached prefix, the model rebuilds
positions from those ids alone, and the tail is rotated as positions
0..K−1 in the ten attention layers: wrong against the prefix it attends
to, and stored that way in the KV cache for every later token. Only the
first forward after the clear is affected. Cold prefills, chunk
continuation and decode steps were always right, which is why every
capability gate (single-turn cold requests) passed. Every follow-up turn
on qwen36, qwen36-27b, qwen35, qwen35-9b-8bit and Qwen3.8-27B took the
wrong path; mlx-lm-loaded presets (flint, coder-30b, qwen3-moe, qwen3-32b,
nemotron-30b) and devstral carry no such state.

Patch 033 seeds the text-only zero delta at the end of the prefix-hit
branch, the state a cold prefill of the same prompt leaves. It does not
change the work a hit does, so the timings in the sections above stand.

Served, the `restart` arm is bit-identical to the uninterrupted decode
over all 4,598 steps and 72 scheduled re-prompts, and the hit-heavy arms
fall to within reach of the cold re-prompt arm (table above). A one-token
hit is now exact, which a cold re-prefill is not.

In chat shape (`scripts/eval/probe_followup_fidelity.py`, `followup/`):
a rendered two-turn history of 1K or 6K tokens, a follow-up of 32 to 3,000
new tokens, 64 greedy tokens with top-20 logprobs, the follow-up served
once from the hit and once cold after a flush. 24 cases per arm; the hit
resumes from a 64-token grid state, so it forwards 116 to 3,124 tokens.
`followup/pooled.txt`:

| comparison | cases | token-identical | agreeing steps of 57 | first-step KL median | KL mean | KL max |
|---|---|---|---|---|---|---|
| hit against cold, without 033 | 24 | 1 | 12.8 | 0.0117 | 0.0658 | 1.02 |
| hit against cold, with 033 | 24 | 9 | 31.7 | 0.0011 | 0.0064 | 0.095 |
| hit against cold, with 033, chunk 2048 | 24 | 13 | 38.7 | 0.0007 | 0.0041 | 0.065 |
| cold chunk 2048 against cold chunk 4096, prompts over 2,048 tokens | 12 | 3 | 31.6 | 0.0018 | 0.0056 | 0.073 |
| hit chunk 4096 against cold chunk 2048, the same 12 | 12 | 4 | 36.1 | 0.0016 | 0.0057 | 0.052 |
| cold without 033 against cold with 033 | 24 | 24 | all | 0 | 0 | 0 |

By tail length, hit against cold at the default chunk size:

| new tokens | with 033: KL mean / agreeing steps | without 033: KL mean / agreeing steps |
|---|---|---|
| 32 | 0.0086 / 25.2 | 0.0856 / 5.3 |
| 256 | 0.0054 / 34.5 | 0.0789 / 10.2 |
| 1,024 | 0.0045 / 38.8 | 0.0386 / 20.2 |
| 3,000 | 0.0072 / 28.2 | 0.0599 / 15.3 |

With the patch a follow-up served from a hit is as close to a cold prefill
as two cold prefills on different chunk sizes are to each other, which is
the noise this stack already accepts. Without it the hit sat ten times
further out and the greedy text parted from the cold run after 5 to 20
tokens. The cold path is untouched: 24 of 24 cold runs are token-identical
across the two trees.

What this reaches beyond the probe: the archived qwen36 SWE-bench cells
ran every follow-up turn on the wrong positions. A re-roll with patch 033
is the measurement of what that cost. The position state lives on the model object, so an image
request and a text request decoding at once could still cross; that is a
multi-request hazard outside the single-user target.

## Capability and cache gates

`gates/`. Three restarts of qwen36 on the tree with patch 033: stock
settings, the buffer limit, and the buffer limit with patches 030 to 032
on (the three hook lines are in that arm's server log). Each arm runs the
capability validator, the thinking, vision and codegen probes, the
2,048-token radix repeat test, the multi-turn hit probe at 8,192 and
24,576 tokens, and the finish-state resume probe over nine generation
lengths.

| gate | stock | buffer limit | buffer limit + 030 to 032 |
|---|---|---|---|
| `validate_capabilities.py` | 3/3 | 3/3 | 3/3 |
| `probe_thinking.py` | VERIFIED | VERIFIED | VERIFIED |
| `probe_vision.py` | STRONG | STRONG | STRONG |
| `probe_codegen.py` | STRONG, 8 of 8 tests | STRONG, 8 of 8 tests | STRONG, 8 of 8 tests |
| radix repeat, 4 iterations | consistent | consistent | consistent |
| follow-up hit against its cold twin, 96 tokens | identical at both depths | identical at both depths | identical at both depths |
| two follow-up hits against each other | differ from token 28 at 8,192, identical at 24,576 | differ from token 28 at 8,192, identical at 24,576 | identical at both depths |
| finish-state resume | pass | pass, token ids and logprobs equal to the stock arm | pass |
| tracebacks in the server log | 0 | 0 | 0 |
| wired memory after the last gate | 28.4 GB | 28.9 GB | 30.4 GB |

Every capability gate passes in every arm. The buffer limit does not
touch numerics: its resume probe returns the stock arm's token ids with a
logprob difference of zero at every generation length.

The multi-turn probe exits non-zero on the stock and buffer-limit arms
because its two hits at 8,192 differ. They are not the same computation.
The first hit resumes from the 8,192-token checkpoint and prefills 80
tokens. That prefill writes a later 64-token-grid checkpoint, so the
second hit resumes from 8,256 and prefills 16 (`#cached-token` in the
server log, the same pair in all three arms). Two chunk grids give two
bf16 roundings, the same noise two cold prefills on different chunk sizes
show, and whether 96 greedy tokens survive it is a coin the fused arm
happened to win. At 24,576 both hits resume from the same checkpoint and
agree in every arm. The row that carries the gate is the hit against its
cold twin: identical over 96 tokens at both depths in all three arms,
where the tree without patch 033 diverged at token 24 and token 55
(`../qwen36-hybrid-overlap/VERDICT.md`).

The wired reading is taken once after the last gate, with whatever the
4 GB buffer cache then holds. The serving A/B arms, which end on the same
request in every arm, read 24.4 to 24.6 GB with and without the fusions
(`serve-router/pooled.txt`, `serve-router3/pooled.txt`).

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
| `raw/` | kernel and launch microbenches, attention layout probe, router probe arms, int8 fidelity, traversal-width sweep, the step census with patches 030 to 032 installed, patch gates and unit-test output for 029 to 033 |
| `slack/`, `knob-sweep-raw.jsonl`, `compile-probe-raw.jsonl` | raw setting sweeps |
| `serve-census-base.jsonl`, `serve-graph/`, `serve-profile/` | step census, primitive counts, graph-build profile |
| `serve-knobs/`, `serve-flags/`, `serve-sampled/`, `serve-prefill/` | serving A/B chains to 30K |
| `serve-128k/`, `serve-256k/` | deep arms with memory lines |
| `serve-ladder/` | threshold-crossing trace |
| `serve-clear/` | cache-clear traces at 122K with memory lines |
| `serve-router/` | launch-folding serving A/B, eleven arms with hook lines, wired memory and text hashes in `pooled.txt` |
| `serve-router3/` | grouped linears on top of the fused router and front, three mirrored pairs on the tree with patch 033 |
| `serve-router-null/` | eleven arms of one build (launch-folding flags set, hooks not installed): the A/B design's noise floor |
| `precision/` | float32-yardstick summaries for the stock and fused decode paths, raw prefix-tail probe with the position state kept and cleared |
| `serve-dfid/` | served decode-fidelity arms: summaries and per-document divergence counts, `-p033` files on the tree with patch 033 |
| `followup/` | chat-shaped follow-up hits against cold prefills, with and without patch 033, pooled in `pooled.txt` |
| `fidelity/` | teacher-forced NLL and KL through the server for each fusion and a radix-off arm, summaries and `pooled.txt` |
| `gates/` | capability, radix-repeat, multi-turn hit and finish-state resume gates for the stock, buffer-limit and all-flags arms |
| `recall/` | depth recall at the 215,040 label |
