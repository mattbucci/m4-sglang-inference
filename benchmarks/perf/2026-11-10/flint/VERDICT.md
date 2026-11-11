# Flint against the published M4 Pro engines: windowed lookahead view (patch 036), fused MoE block (patch 037)

Run dates 2026-11-10 and 2026-11-11. Stack: SGLang v0.5.23 + patches 001-037
(036 and 037 come out of this run), MLX 0.32.4, mlx-lm 0.32.3.dev17,
mlx-vlm 0.6.5, M4 Pro 64 GB with the 20-core GPU. Models:
`RecursiveSI/flint-int4` (the official 4-bit MLX checkpoint every published
number uses: 4.54 bits per weight, 22.6 GB) and `mlx-community/flint-oQ8e`
(the `flint` preset's default). Preset `flint`: 32K label, four request
slots, radix cache on, 1,024-token prefill chunks.

Two other engines publish Flint numbers for this GPU. They are the reference
of this lane in the way vllm-radiance is the R9700 team's: a number somebody
else reached on the same silicon, so a gap is a list of missing mechanisms.
Their binaries were not run here; their rows are quoted from their receipts.

## The published rows

MacBook Pro M4 Pro, 20-core GPU, 48 GB, `RecursiveSI/flint-int4`. Context
sweep: one cold request per depth, a 256-token reply at temperature 1.0,
tokens per second.

| prompt tokens | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode, rec 0.1.0 | 126.5 | 123.0 | 116.5 | 107.0 | 95.2 | 79.4 | 56.5 | 35.2 |
| decode, Kapitsa | 124.3 | 121.3 | 119.2 | 113.2 | 106.2 | 85.9 | 66.2 | 45.7 |
| decode, llama.cpp | 77.5 | 75.1 | 73.4 | 69.7 | 63.7 | 53.8 | 40.5 | 23.8 |
| prefill, rec 0.1.0 | 781 | 800 | 791 | 760 | 699 | 581 | 427 | 262 |
| prefill, Kapitsa | 858 | 976 | 960 | 907 | 808 | 651 | 464 | 300 |
| prefill, llama.cpp | 927 | 1015 | 978 | 910 | 785 | 602 | 382 | 199 |

Short runs, five runs of 512 generated tokens on a 1K prompt:

| | decode T=1.0 | decode greedy | ms per token greedy | follow-up TTFT 4K | follow-up TTFT 16K |
|---|--:|--:|--:|--:|--:|
| Kapitsa | 127.90 | 131.60 | 7.60 | 0.13 s | 0.15 s |
| rec 0.1.0 | 126.56 | 132.37 | 7.55 | 0.41 s | 0.44 s |
| llama.cpp | 79.57 | 79.93 | 12.51 | 0.23 s | 0.31 s |

Sources: `github.com/nace-ai/kapitsa` pull requests 286 and 290 (the bench
tables for all three engines), `github.com/recursive-oss/rec`, and the model
card `huggingface.co/RecursiveSI/flint`.

## Method

The method of `../micro/VERDICT.md` (census, raw probe before serving work,
same-tree serving A/B in both arm orders, gates, a numbered patch when
retained, a disposition per experiment, a kill criterion up front), plus:

1. Like-for-like first. `scripts/bench/published_shape_sweep.py` issues the
   published request shapes against one booted server: a cold request per
   depth with its own seed (no depth is a prefix of another), the same
   request again for the hit path, five runs of 512 tokens on a 1K prompt
   sampled (temperature 1.0, top-p 0.95, top-k 20) and greedy, and a chat
   follow-up. Decode is tokens per second from the first to the last
   streamed token; prefill is fresh prompt tokens over time to first token.
2. Raw before served. `scripts/bench/mlx_flint_decode_probe.py` times each
   launch fold on the real checkpoint in one process: 64 teacher-forced
   steps against the stock arm's log-probabilities, then three 96-step
   pipelined segments, the arm list forward and then reversed.
3. A fold is carried to serving only when it is bit-identical to the stock
   forward and clears noise raw. Serving arms inside 1 % are reverted.
4. A flag arm counts only if the server log shows the hook's line (`MLX
   fused Flint MoE enabled: patched 47 blocks`). Every flag arm here has it
   and zero tracebacks (`ab/driver.txt`).
5. Greedy text can be compared between arms here because every retained
   change is bit-identical; the hash of the greedy runs is the check.

## Dispositions

| experiment | family | measured | disposition |
|---|---|---|---|
| checkpoint: oQ8e against int4 | weight reads | oQ8e reads 2,672.5 MB of weights per token, int4 1,450.7; served greedy 55.0 against 74.9 tok/s at 1K on the same tree | published rows are compared on int4; the preset default stays oQ8e until the gates and a fidelity dump run on int4 |
| windowed lookahead view (patch 036) | host and copies | a chained step held a frozen view of each of the 35 sliding caches, so the next write copied the buffer: 1.20 against 0.84 ms per step of cache writes raw; served +7.8 % greedy at 1K, +6.1 / +6.1 / +7.3 % at 1K / 8K / 32K | retained, always on |
| `gatek`: router logits as one launch on the BF16 weight | launches | 10.66 to 10.86 against 11.32 to 11.53 ms raw at 1K, bit-identical | in patch 037 |
| `comb`: expert combine plus shared expert as one launch | launches | 10.79 to 10.89 ms raw at 1K, bit-identical | in patch 037 |
| `comb+gatek` (patch 037) | launches | 10.03 to 10.16 against 11.47 to 11.51 ms raw at 1K, 11.15 to 11.36 against 12.47 to 13.17 at 8K; served +12.7 % greedy over the view at 1K, +11.7 / +8.9 % at 8K / 32K; oQ8e +14.1 % | retained, flag `SGLANG_MLX_FUSE_FLINT_MOE`, on in the `flint` preset |
| `gate32`: float32 copy of the router weight, stock matmul | launches | 10.99 to 11.11 ms | superseded by `gatek` |
| `qkv`, `mlp2`, `front`, `mid` | launches | 11.31 to 11.47, 11.19 to 11.30, 11.28 to 11.44, 11.51 to 11.80 ms: inside noise or slower | not shipped |
| `tail`, `all` | launches | `tail+gatek` 10.01 to 10.12, `all` 9.93 to 10.02 with one run at 11.12: equal to `comb+gatek` while removing 96 to 509 more primitives | not shipped: launch count below 2,183 buys nothing here |
| `sglu`: the tree's fused routed-expert kernel | launches | 11.67 to 12.81 ms, KL up to 0.17, argmax 62 of 64 at 8K | rejected: slower and not bit-identical |
| routing knock-out (`route_const`, `route_replay`) | launches | 9.02 to 9.09 ms with `comb` against 9.89 to 9.96 for `comb+gatek`; replaying real routes costs what constant routes cost | not a candidate; it prices the two routing launches per block at 0.85 ms a step, independent of which experts are read. One routing kernel per block is the open follow-up |
| radix cache off | host | 82.5 against 74.9 tok/s greedy before patch 036, 92.4 against 91.0 with both patches; a repeat of a 30K prompt answers in 51.5 s against 0.21 s | radix stays on: patch 036 removed the reason to turn it off |
| command-buffer limits | host slack | raw: stock limits lose 2.0 ms of step time to a 2 ms host delay, `MLX_MAX_MB_PER_BUFFER=200` loses 0.5, 100 operations with 400 MB loses none, 200 operations with 1,000 MB is slower than stock; served, eight mirrored arms: +9.9 / +10.3 / +11.1 % greedy at 1K for 200 MB / 100 operations and 400 MB / 100 operations and 1,000 MB, +5.2 to +5.5 % decode at 32K, cold prefill 6.5 to 7.9 % slower at 8K and 3.7 to 6.0 % slower at 32K, 0.5 to 1.2 GB more wired | a trade, decode for cold prefill; the three settings tie in decode and 200 MB costs the least prefill and memory. The decode-only submission arms decide what the preset takes |
| decode submission in parts | host slack | raw: four early submissions recover 1.0 ms of a 1 ms host delay on stock limits and nothing once the limits are 100 operations and 400 MB | raw go on stock limits; served arms decide whether it ships |

## Baseline

`baseline/`. The preset on both checkpoints before this lane, patches
001-035.

| prompt tokens | int4 prefill / decode | int4 radix off prefill / decode | oQ8e prefill / decode |
|--:|--:|--:|--:|
| 959 | 869.7 / 74.20 | 918.8 / 75.84 | 805.6 / 52.36 |
| 1,912 | 896.8 / 73.34 | 907.5 / 74.27 | 822.7 / 52.17 |
| 3,811 | 856.7 / 72.20 | 844.8 / 72.29 | 694.3 / 51.26 |
| 7,601 | 800.8 / 68.62 | 784.9 / 68.39 | 632.0 / 49.91 |
| 15,215 | 683.0 / 60.00 | 709.3 / 66.86 | 591.5 / 47.75 |
| 30,422 | 540.3 / 56.96 | 579.3 / 59.64 | 513.3 / 43.46 |

Short runs on int4: 73.98 sampled and 75.00 greedy tok/s. Chat follow-up:
0.18 s to first token at 3.8K and 0.34 s at 15.2K. `mlx_lm.generate` on the
same checkpoint: 87.5 tok/s, peak 22.7 GB. The int4 checkpoint passes the
chat-template check, the quantization-scale scan (486 layers healthy) and
`validate_capabilities.py` 3/3.

Prefill is already at the published level (870 to 540 tok/s against rec's
781 to 581). The follow-up turn (0.18 s at 3.8K, 0.34 s at 15.2K) sits
between Kapitsa's (0.13 / 0.15 s) and rec's (0.41 / 0.44 s). The gap is
decode: 13.35 ms per token against 7.6.

## Raw census

`raw/decode-census-int4.txt` (`scripts/bench/mlx_decode_census.py`, module
families knocked out of the raw step) and `raw/weight-read-census-int4.json`.

| depth | pipelined step | attention | MLP blocks | routed experts | shared expert | floor |
|--:|--:|--:|--:|--:|--:|--:|
| 24 | 10.93 ms | 3.81 | 6.16 | 2.92 | 0.68 | 0.88 |
| 8,192 | 12.44 ms | 5.31 | 6.28 | 3.12 | 1.03 | 0.88 |
| 32,768 | 16.64 ms | 9.49 | 7.08 | 3.82 | 0.90 | 0.88 |

A step is 2,747 graph primitives, about 56 per layer. It reads 1,450.7 MB of
weights (attention 520.3, routed experts 499.0, output head 173.0), which is
6.1 to 8.2 ms at the 178 to 239 GB/s this GPU sustains. The published
engines sit at 7.6 ms. Attention reads a third of the weights and costs a
third of the step at shallow depth, then grows with the 14 full-attention
layers.

## Raw fold probe

`raw/fold-probe.jsonl`, `raw/fold-combination-probe.jsonl`,
`raw/route-replay-probe.jsonl`. Pipelined ms per step, two passes (arm list
forward, then reversed), 64 teacher-forced steps for fidelity.

| arm | primitives | 1K | 8K | bit-identical |
|---|--:|--:|--:|---|
| base | 2,747 | 11.32 / 11.53 | 12.59 / 12.66 | reference |
| qkv | 2,698 | 11.31 / 11.47 | 12.64 / 12.66 | yes |
| mlp2 | 2,747 | 11.19 / 11.30 | 12.48 / 12.50 | yes |
| front | 2,383 | 11.28 / 11.44 | 12.34 / 12.70 | yes |
| mid | 2,698 | 11.51 / 11.80 | 12.68 / 12.52 | yes |
| tail | 2,322 | 10.97 / 11.15 | 12.00 / 11.95 | yes |
| gate32 | 2,700 | 10.99 / 11.11 | 11.96 / 12.10 | yes |
| comb | 2,418 | 10.79 / 10.89 | 11.99 / 11.85 | yes |
| gatek | 2,512 | 10.66 / 10.86 | 11.77 / 11.84 | yes |
| sglu | 2,700 | 11.67 / 12.81 | 12.75 / 12.83 | no (KL max 0.04 / 0.17) |
| comb+gatek | 2,183 | 10.03 / 10.16 | 11.36 / 11.15 | yes |
| tail+gatek | 2,087 | 10.01 / 10.12 | 11.12 / 11.19 | yes |
| comb+gatek+front | 1,819 | 9.92 / 9.92 | 11.03 / 10.99 | yes |
| all | 1,674 | 10.01 / 9.93 | 10.94 / 10.85 | yes |
| comb+route_replay | 2,089 | 9.09 / 9.09 | | yes (routes replayed) |
| comb+route_const | 2,089 | 9.02 / 9.04 | | no (routes frozen) |

Two folds carry the whole gain. The router product was two casts and a
matmul per block, with the BF16 weight cast to float32 on every step of every
block; the kernel reads the BF16 weight directly and reproduces the stock
matmul's summation order. The combine was multiply, sum, cast and add. The
other folds remove up to 509 more primitives and no time, so launch count
alone is not what the step pays for at this size.

## Frozen view probe

`raw/frozen-view-probe.txt`. 35 sliding caches, window 513, 400 steps of
cache writes, median ms per step: 0.84 with nothing held, 1.20 with a frozen
view of the previous state held (each write copies the buffer), 0.85 when
only the offset is kept. Patch 036 keeps the offset and reads the state one
token back.

## Served A/B

`ab/`. `RecursiveSI/flint-int4` unless noted, five runs of 512 tokens on a
963-token prompt, arms in mirrored order and pooled.

| arm | arms | sampled tok/s | greedy tok/s | greedy ms/token | greedy against base |
|---|--:|--:|--:|--:|--:|
| base (patches 001-035) | 2 | 73.44 | 74.91 | 13.35 | |
| windowed lookahead view (036) | 2 | 79.64 | 80.75 | 12.38 | +7.8 % |
| view + fused MoE (036 + 037) | 2 | 89.17 | 91.00 | 10.99 | +21.5 % |
| radix off, flag off | 2 | 81.22 | 82.49 | 12.12 | |
| radix off, fused MoE | 2 | 90.19 | 92.44 | 10.82 | +12.1 % over its base |
| oQ8e base | 1 | 54.38 | 55.04 | 18.17 | |
| oQ8e view + fused MoE | 1 | 61.82 | 62.80 | 15.92 | +14.1 % over its base |

Greedy run ranges: base 73.96 to 75.82, view 79.94 to 81.34, both patches
90.57 to 91.41. No two arms overlap.

Per depth, cold request, 256-token reply at temperature 1.0:

| arm | decode 1K / 8K / 32K | prefill 1K / 8K / 32K | repeat TTFT 1K / 8K / 32K |
|---|--:|--:|--:|
| base | 74.62 / 69.23 / 56.70 | 889.6 / 791.2 / 540.6 | 0.170 / 0.173 / 0.212 s |
| view (036) | 79.14 / 73.44 / 60.83 | 867.2 / 797.6 / 555.5 | 0.172 / 0.174 / 0.211 s |
| view + fused MoE | 88.80 / 82.02 / 66.23 | 877.6 / 805.1 / 563.5 | 0.157 / 0.175 / 0.211 s |
| radix off, flag off | 80.53 / 74.41 / 61.30 | 977.0 / 841.0 / 570.5 | 0.984 / 9.466 / 51.479 s |
| radix off, fused MoE | 90.81 / 83.82 / 67.25 | 969.8 / 818.5 / 561.5 | 0.978 / 9.866 / 51.541 s |

Both patches against base: +19.0 / +18.5 / +16.8 % decode at 1K / 8K / 32K,
prefill and repeat TTFT inside noise. Several streams (`ab/*.streams.json`,
200 tokens, aggregate tok/s at 1 / 2 / 4 streams): base 76.75 / 111.59 /
147.91, both patches 93.45 / 126.56 / 176.17. The chat follow-up at 3.8K
answers in 0.18 to 0.23 s in every arm.

Command-buffer limits on top of both patches (`buffers/`, eight arms in
mirrored order in one window, pooled per setting; the stock limits on this
chip are 50 operations and 50 MB):

| limits | arms | sampled tok/s | greedy tok/s | greedy ms/token | greedy against stock | wired GB |
|---|--:|--:|--:|--:|--:|--:|
| stock | 2 | 91.08 | 93.41 | 10.70 | | 31.1 |
| 200 MB | 2 | 100.49 | 102.68 | 9.74 | +9.9 % | 31.6 |
| 100 operations, 400 MB | 2 | 101.12 | 103.07 | 9.70 | +10.3 % | 31.9 |
| 100 operations, 1,000 MB | 2 | 101.41 | 103.80 | 9.63 | +11.1 % | 32.3 |

| limits | decode 1K / 8K / 32K | prefill 1K / 8K / 32K | repeat TTFT 1K / 8K / 32K | streams 1 / 2 / 4 |
|---|--:|--:|--:|--:|
| stock | 90.73 / 83.13 / 67.41 | 893.7 / 808.7 / 560.4 | 0.153 / 0.184 / 0.208 s | 95.42 / 129.60 / 180.29 |
| 200 MB | 99.30 / 91.50 / 70.89 | 906.8 / 756.5 / 539.6 | 0.175 / 0.185 / 0.213 s | 107.60 / 137.78 / 185.14 |
| 100 operations, 400 MB | 101.25 / 91.96 / 71.12 | 894.2 / 752.4 / 535.3 | 0.175 / 0.189 / 0.217 s | 106.51 / 139.30 / 185.34 |
| 100 operations, 1,000 MB | 101.00 / 92.34 / 71.12 | 876.5 / 745.2 / 527.0 | 0.186 / 0.189 / 0.223 s | 107.48 / 138.61 / 185.27 |

Greedy run ranges: stock 92.12 to 94.38, 200 MB 101.79 to 103.34, the two
100-operation settings 102.56 to 104.91. The three settings sit within
1.1 % of one another in decode, which is noise. All of them buy decode
with cold prefill: +9 to +12 % decode at 1K and 8K and +5 % at 32K, against
prefill 6.5 to 7.9 % slower at 8K and 3.7 to 6.0 % slower at 32K. The gain
shrinks with company: +12 % for one stream, +6 to +7 % for two, +3 % for
four. The greedy text is the same in every arm. The stock arm of this
window reads 2.6 % above the two-patch arms of the first window, so
settings are compared inside a window only.

## Fidelity

- Greedy text: one hash (`155684ab780a41c8`) for every int4 arm, base
  included, and one (`5165ea1975c35024`) for both oQ8e arms.
- Follow-up fidelity (`scripts/eval/probe_followup_fidelity.py`,
  `ab/followup-compare.txt`): the view arm and the two-patch arm against
  base, 8 of 8 hit cases and 8 of 8 cold cases with equal tokens, 512 of
  512 steps with equal top-k log-probabilities, largest difference 0,
  cached-token counts equal in every case. The three dumps hash the same;
  the base dump is kept as `ab/base-1.followup.json`.
- Install-time checks: patch 037 compares its router kernel and its fused
  forward with the stock block bit for bit on the block's own weights and
  keeps the stock call wherever that fails. All 47 blocks pass on int4 and
  on oQ8e.
- `validate_capabilities.py` with both patches on int4: 3/3
  (`ab/both-validate.txt`).
- Unit tests: `scripts/eval/test_swa_lookahead_view_unit.py`,
  `scripts/eval/test_flint_moe_unit.py`. Patch gates
  (`scripts/test_patch_gates.sh`): 27 of 27 apply clean on pristine
  v0.5.23 and reproduce the 27-patch tree byte for byte.

## Host slack

`raw/pipeline-slack-probe.jsonl` and `buffers/pipeline-slack-by-limit.jsonl`
(`scripts/bench/mlx_pipeline_slack_probe.py`). A pipelined loop builds the
next step while the GPU runs the current one. The probe adds a busy-wait
before each graph build, standing in for the host work a serving step adds,
and optionally submits the step early after four evenly spaced layers. A
flat row means the host work is hidden; a slope of one means the GPU waits
for it. Raw ms per step at 1K, two passes, tokens equal in every arm.

| operations per buffer | MB per buffer | early submissions | delay 0 ms | delay 1 ms | delay 2 ms |
|--:|--:|--:|--:|--:|--:|
| 50 (stock) | 50 (stock) | 0 | 11.31 / 11.35 | 12.21 / 12.16 | 13.34 / 13.39 |
| 50 | 50 | 4 | 11.18 / 11.24 | 11.17 / 11.24 | 11.71 / 11.75 |
| 50 | 200 | 0 | 11.10 / 11.07 | 11.10 / 11.08 | 11.61 / 11.59 |
| 50 | 200 | 4 | 11.11 / 11.10 | 11.09 / 11.08 | 11.04 / 11.07 |
| 50 | 400 | 0 | 11.04 / 11.03 | 11.04 / 11.08 | 11.09 / 11.13 |
| 50 | 400 | 4 | 11.01 / 11.04 | 11.02 / 11.02 | 10.99 / 11.02 |
| 50 | 1000 | 0 | 11.01 / 11.04 | 11.02 / 11.04 | 11.09 / 11.09 |
| 100 | 400 | 0 | 10.95 / 10.98 | 10.92 / 10.94 | 10.90 / 10.91 |
| 100 | 400 | 4 | 10.92 / 10.92 | 10.90 / 10.92 | 10.93 / 10.90 |
| 100 | 1000 | 0 | 10.78 / 10.98 | 10.79 / 10.99 | 10.94 / 10.94 |
| 200 | 1000 | 0 | 11.46 / 11.42 | 11.35 / 11.34 | 11.21 / 11.23 |
| 200 | 1000 | 4 | 11.70 / 11.68 | 11.53 / 11.54 | 11.42 / 11.38 |

The served host phase is 2.1 to 2.4 ms a step (`serve-census/`), so the
stock limits leave about 2 ms of every served step with the GPU idle. Either
lever hides it raw: larger command buffers keep work queued when
`async_eval` returns, and early submission starts the GPU on the front of
the next step while the host builds the rest. Once the limits are 100
operations and 400 MB, early submission has nothing left to recover. 200
operations per buffer is past the optimum: the step is slower than stock at
zero delay.

The two-kernel fold under each limit (`buffers/fold-probe-*.jsonl`,
`comb+gatek`, bit-identical in every setting): 9.94 ms on stock limits,
9.49 to 9.52 at 200 MB, 9.42 to 9.50 at 400 MB, 9.39 to 9.41 at 100
operations and 400 MB, 9.97 to 9.98 at 200 operations and 1,000 MB.

## The gap at 1K

| | ms per token | tok/s |
|---|--:|--:|
| published, rec and Kapitsa, greedy | 7.55 to 7.60 | 131.6 to 132.4 |
| weight reads alone at 178 to 239 GB/s | 6.1 to 8.2 | |
| raw two-kernel step, 100 operations and 400 MB buffers | 9.4 | 106 |
| raw two-kernel step, stock limits | 9.9 to 10.0 | 100 to 101 |
| raw mlx-lm step | 11.3 to 11.5 | 87 to 88 |
| served, both patches, 200 MB buffers | 9.74 | 102.7 |
| served, both patches | 10.99 | 91.0 |
| served, view only | 12.38 | 80.8 |
| served, base | 13.35 | 74.9 |

Served decode moved from 57 % of the published rate to 69 % with the two
patches. The buffer limit brings the served step to the raw step and the
rate to 78 %, and pays for it in cold prefill. What is left is about
1.8 ms inside the graph itself: the routing launches (0.85 ms), and
attention and the routed experts above their weight reads.

## Receipts

| folder | contents |
|---|---|
| `baseline/` | published-shape sweeps of the preset before this lane (int4 radix on and off, oQ8e), the validator, the chat-template check, the scale scan, `mlx_lm.generate` |
| `raw/` | decode census, weight-read census per checkpoint, fold probes, route replay probe, frozen view probe, host slack probe |
| `serve-census/` | served step split into host and GPU phases, radix on and off and with both changes |
| `ab/` | the thirteen served arms, streams, follow-up fidelity, the validator with both patches, the driver log |
| `buffers/` | host slack and the two-kernel fold under each command-buffer limit; the eight served arms, their streams and the driver log |
