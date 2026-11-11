# Flint against the published M4 Pro engines: windowed lookahead view (patch 036), fused MoE block (patch 037), decode submission in parts (patch 038)

Run dates 2026-11-10 and 2026-11-11. Stack: SGLang v0.5.23 + patches 001-038
(036, 037 and 038 come out of this run), MLX 0.32.4, mlx-lm 0.32.3.dev17,
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
| checkpoint: oQ8e, oQ6e, int4 | weight reads | per token oQ8e reads 2,672.5 MB of weights, oQ6e 2,465.0, int4 1,450.7; the shipped preset decodes 62.5 / 66.7 / 96.7 tok/s sampled at 1K and peaks at 52.2 / 42.7 / 32.6 GB wired; int4 and oQ8e pass the validator, the thinking probe and the codegen probe with the preset's flags; teacher-forced against oQ8e, oQ6e is at KL 0.051 with 93.7 % of argmax tokens equal and int4 at KL 0.218 with 83.0 %, NLL equal within one standard error on all three | the default stays oQ8e: the agentic cell was scored on it, and int4 is a different distribution from the oQ builds, not a smaller copy of them. int4 is the gate-validated fast option (`MODEL=RecursiveSI/flint-int4 scripts/launch.sh flint`); a default switch waits for an agentic cell on int4 |
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
| command-buffer limits | host slack | raw: stock limits lose 2.0 ms of step time to a 2 ms host delay, `MLX_MAX_MB_PER_BUFFER=200` loses 0.5, 100 operations with 400 MB loses none, 200 operations with 1,000 MB is slower than stock; served, eight mirrored arms: +9.9 / +10.3 / +11.1 % greedy at 1K for 200 MB / 100 operations and 400 MB / 100 operations and 1,000 MB, +5.2 to +5.5 % decode at 32K, cold prefill 6.5 to 7.9 % slower at 8K and 3.7 to 6.0 % slower at 32K, 0.5 to 1.2 GB more wired | not a preset default: a trade, decode for cold prefill. The three settings tie in decode and 200 MB costs the least prefill and memory. Patch 038 takes three quarters of the same gain for free, so the limit stays an opt-in for decode-heavy work |
| decode submission in parts (patch 038) | host slack | raw: four early submissions recover 1.0 ms of a 1 ms host delay on stock limits and nothing once the limits are 100 operations and 400 MB; served, nine arms: five parts +7.5 % greedy and +7.8 % sampled at 1K, +8.2 / +8.9 / +2.7 % cold decode at 1K / 8K / 32K, +8.8 / +6.0 / +2.1 % at 1 / 2 / 4 streams, prefill inside the arm spread, wired memory equal, same greedy text; nine parts +8.1 % on one arm; five parts with 200 MB buffers +10.3 %, equal to the buffers alone | retained, flag `SGLANG_MLX_DECODE_SUBMIT_PARTS`, 5 in the `flint` preset |
| attention parts (knock-out census) | launches | at 1K attention costs 4.29 ms of the 11.47 ms raw mlx-lm step and its reads cost 3.12 ms at 200 GB/s; the query and output projections run at 218 to 243 GB/s, their read cost; head norms, the rotary embedding and a non-additive rest hold 1.17 ms | no fold carried: the folds over these operations (`qkv`, `front`, `mid`) measured inside noise. Recorded as the attention floor of this graph |
| deep ladder through 128K | like for like | decode 98.9 tok/s at 1K to 35.5 at 128K: 78 to 84 % of rec through 32K, 94 % at 64K, 101 % at 128K, 75 to 80 % of Kapitsa at every depth; per token of context a step costs 0.140 µs here, 0.159 for rec, 0.107 for Kapitsa | measured. The fixed 2.1 ms per step is this graph's launches; the per-token term is the read rate of the shared keys and values, and a raw probe of the attention kernel at 32K to 128K is the next experiment |

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

Decode submission in parts on top of both patches (`submit/`, nine arms in
mirrored order, pooled per setting, a tree with patches 001-038):

| setting | arms | sampled tok/s | greedy tok/s | greedy ms/token | greedy against one submission | wired GB |
|---|--:|--:|--:|--:|--:|--:|
| one submission, stock limits | 2 | 90.72 | 92.95 | 10.76 | | 31.1 |
| 5 parts | 2 | 97.84 | 99.92 | 10.01 | +7.5 % | 31.1 |
| 9 parts | 1 | 98.21 | 100.50 | 9.95 | +8.1 % | 31.1 |
| 5 parts, 200 MB buffers | 2 | 100.60 | 102.56 | 9.75 | +10.3 % | 31.4 |
| 5 parts, 100 operations and 400 MB | 2 | 101.28 | 103.56 | 9.66 | +11.4 % | 32.0 |

| setting | decode 1K / 8K / 32K | prefill 1K / 8K / 32K | repeat TTFT 1K / 8K / 32K | streams 1 / 2 / 4 |
|---|--:|--:|--:|--:|
| one submission, stock limits | 89.97 / 81.95 / 66.78 | 882.8 / 804.5 / 536.5 | 0.157 / 0.171 / 0.210 s | 94.61 / 128.09 / 178.11 |
| 5 parts | 97.38 / 89.28 / 68.56 | 880.0 / 792.5 / 556.0 | 0.166 / 0.171 / 0.210 s | 102.98 / 135.72 / 181.88 |
| 9 parts | 97.35 / 89.86 / 69.01 | 880.8 / 779.5 / 541.6 | 0.177 / 0.174 / 0.215 s | |
| 5 parts, 200 MB buffers | 100.31 / 91.50 / 71.52 | 901.6 / 767.4 / 539.0 | 0.175 / 0.184 / 0.222 s | 106.48 / 137.52 / 184.22 |
| 5 parts, 100 operations and 400 MB | 100.61 / 92.59 / 71.51 | 896.9 / 760.3 / 529.0 | 0.184 / 0.188 / 0.225 s | 107.40 / 139.88 / 185.77 |

Greedy run ranges: one submission 92.23 to 94.48, five parts 99.61 to
100.13, nine parts 100.31 to 100.72. Five parts on stock limits take 0.75 ms
off the step and leave prefill alone: a prefill chunk keeps its single
submission, and the 8K prefill of the two five-part arms (777 and 808
tok/s) straddles the two one-submission arms (796 and 814). Nine parts are
not separable from five. With a larger buffer the parts add nothing: 102.56
against 102.68 for the 200 MB buffer alone. So the two levers hide the same
host slack, the patch hides three quarters of it at no cost, and the
buffer hides the rest for 5 % of the cold prefill at 8K. The preset takes
the patch with five parts and keeps the stock limits.

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

Middle limits (`buffers/pipeline-slack-middle-limits.jsonl`, same probe,
delays 0 and 2 ms, two passes, tokens equal in every arm):

| operations per buffer | MB per buffer | early submissions | delay 0 ms | delay 2 ms |
|--:|--:|--:|--:|--:|
| 50 | 75 | 0 | 11.44 / 11.41 | 13.36 / 13.31 |
| 50 | 75 | 4 | 11.16 / 11.20 | 11.68 / 11.72 |
| 50 | 100 | 0 | 11.12 / 11.08 | 12.20 / 12.15 |
| 50 | 100 | 4 | 11.10 / 11.08 | 11.13 / 11.09 |
| 50 | 150 | 0 | 11.07 / 11.10 | 11.63 / 11.66 |
| 50 | 150 | 4 | 11.10 / 11.16 | 11.04 / 11.05 |
| 100 | 50 (stock) | 0 | 11.34 / 11.41 | 13.27 / 13.40 |
| 100 | 50 (stock) | 4 | 11.13 / 11.26 | 11.62 / 11.81 |

With four early submissions, 100 MB per buffer is the smallest limit that
hides 2 ms of host work in full. 75 MB behaves like stock, and so does the
operations limit on its own: the megabyte limit is the one that splits a
Flint step.

The two-kernel fold under each limit (`buffers/fold-probe-*.jsonl`,
`comb+gatek`, bit-identical in every setting): 9.94 ms on stock limits,
9.49 to 9.52 at 200 MB, 9.42 to 9.50 at 400 MB, 9.39 to 9.41 at 100
operations and 400 MB, 9.97 to 9.98 at 200 operations and 1,000 MB.

## Raw prefill

`buffers/raw-prefill-by-limit.jsonl`, `buffers/raw-prefill-by-cache.jsonl`
and `buffers/raw-prefill-cache-in-process-*.jsonl`
(`scripts/bench/mlx_prefill_limit_probe.py`): the int4 checkpoint prefilled
by a bare MLX loop in 1,024-token chunks, trunk only, one `mx.eval` on the
cache arrays per chunk. Depths are exact token counts. Limits are read once
per process, so every limit setting is its own process; the buffer-cache
limit and the per-chunk `mx.clear_cache()` change inside one.

Prefill tok/s by command-buffer limit, cache cleared after every chunk, 2 GB
buffer cache. Two passes at 8,192, one at 32,768, processes in run order:

| operations per buffer | MB per buffer | 8,192 | 32,768 | last chunk at 32,768 | peak over weights at 32,768, GB |
|--:|--:|--:|--:|--:|--:|
| 50 (stock) | 50 (stock) | 775.7 / 795.0 | 576.6 | 442.7 | 7.8 |
| 50 | 75 | 853.6 / 847.3 | 611.9 | 449.7 | 8.0 |
| 50 | 100 | 864.8 / 869.6 | 628.1 | 461.6 | 8.2 |
| 50 | 150 | 866.8 / 849.8 | 618.8 | 462.0 | 8.7 |
| 50 | 200 | 862.1 / 861.8 | 626.3 | 460.8 | 8.9 |
| 100 | 400 | 862.6 / 866.9 | 625.2 | 459.0 | 10.5 |
| 100 | 50 | 862.0 / 859.5 | 625.1 | 458.6 | 7.8 |
| 50 (stock, again) | 50 (stock) | 865.6 / 862.8 | 630.6 | 462.7 | 7.8 |

The first stock process is the slow one and the repeated stock process is
as fast as any other arm, so the limits are not what moved it. The cache
probe repeats the pattern (stock process first, cache kept between chunks,
three passes at 8,192):

| MB per buffer | buffer cache, GB | 8,192 | 32,768 |
|--:|--:|--:|--:|
| 50 (stock) | 2 | 795.0 / 748.0 / 743.4 | 550.8 |
| 50 (stock) | 8 | 803.6 / 744.6 / 775.4 | 587.8 |
| 100 | 2 | 866.5 / 854.4 / 839.4 | 610.1 |
| 100 | 8 | 866.6 / 866.7 / 867.3 | 629.5 |
| 200 | 2 | 863.3 / 863.4 / 866.7 | 627.4 |
| 200 | 8 | 866.9 / 864.9 / 864.7 | 626.5 |

The in-process probe settles it: one process per row, four arms inside it
(clear every chunk, keep the cache, keep an 8 GB cache, clear again), stock
limits before and after the 100 MB process.

| process | 8,192, range over four arms and two passes | 32,768, range over four arms | first chunk | last chunk at 8,192 | last chunk at 32,768 |
|---|--:|--:|--:|--:|--:|
| stock | 862.9 to 881.3 | 626.1 to 634.5 | 979 to 1,016 | 778 to 799 | 460 to 467 |
| 100 MB | 871.5 to 876.8 | 631.6 to 635.1 | 989 to 1,013 | 788 to 791 | 463 to 468 |
| stock, again | 868.0 to 877.3 | 631.4 to 635.6 | 988 to 1,015 | 782 to 792 | 461 to 467 |

Raw chunked prefill is about 870 tok/s at 8,192 tokens and 633 at 32,768,
and it does not depend on the command-buffer limits, on the buffer-cache
limit or on clearing the cache per chunk. Two of five stock processes ran 8
to 12 % slow at every depth, and none of the nine with a raised limit. Both
slow ones started right after another GPU process had exited (a server stop,
the slack probes), the arm order puts stock there, and `pmset -g therm`
records no thermal or performance warning: a process-level event, unexplained,
and the reason every sequence here repeats its reference arm last.

Against the published prefill rows at the same depths (rec 760 and 581,
Kapitsa 907 and 651, llama.cpp 910 and 602), the raw MLX loop is past rec
at both depths, 4 % under Kapitsa and llama.cpp at 8K, and at 32K 3 % under
Kapitsa and past llama.cpp. No served lever is left in this family: a
per-chunk cache clear or a raised limit would change nothing the raw loop
does not already have. The served arms with 200 MB buffers read 6.5 to 7.9 %
slower at 8K (Served A/B) and the raw loop does not reproduce it, so that
loss belongs to the served chunk path and not to MLX's prefill.

## Attention census

`raw/attention-census-int4.txt` (`scripts/bench/mlx_flint_attention_census.py`):
one part of every layer's attention knocked out of the raw mlx-lm step, both
arm orders, the saved step time next to the bytes the part reads. The read
cost is priced at 200 GB/s. The step with all attention removed is 7.13 to
7.21 ms at every depth.

| part | saved at 1,024 | read cost | saved at 8,192 | read cost |
|---|--:|--:|--:|--:|
| whole attention | 4.29 ms | 3.12 ms (623.0 MB) | 5.49 ms | 4.33 ms (865.2 MB) |
| query projection | 0.95 | 1.16 (231.2 MB) | 1.09 | 1.16 |
| output projection | 1.06 | 1.16 (231.2 MB) | 1.21 | 1.16 |
| key and value projections | 0.39 | 0.29 (57.8 MB) | 0.57 | 0.29 |
| attention kernel | 0.66 | 0.51 (102.8 MB) | 1.84 | 1.73 (345.0 MB) |
| head norms | 0.23 | | 0.02 | |
| rotary embedding | 0.20 | | 0.15 | |
| cache write alone | 0.01 | | -0.03 | |
| rest (the parts are not additive) | 0.79 | | 0.65 | |
| one more attention launch per layer | 0.84 | 0.51 | 2.26 | 1.73 |

The attention kernel's bytes are the size of the cache buffers, an upper
bound of what a step reads. The two base arms at depth 24 disagree by 1.3 ms
(the first arm of the process), so that depth is not read.

- The projections are at their read cost: 231.2 MB in 0.95 to 1.21 ms is 191
  to 243 GB/s. Nothing is left to fold there.
- The attention kernel costs what its reads cost, and a second launch of it
  on warm buffers costs 0.3 to 0.5 ms more than its reads.
- 1.17 ms of attention sits above its reads at both depths: the head norms
  and the rotary embedding (0.43 ms at 1K) and a non-additive rest of 0.65 to
  0.79 ms. The fold probe already ran one launch over these operations
  (`qkv`, `front`, `mid`) and measured inside noise, so this is the floor of
  attention in this graph, not a queue of folds.

## Deep ladder against the published rows

`deep/`: the published context sweep through 128K on int4. One request slot,
radix cache off, `CTX=131136 MEM_FRAC=0.5`, 1,024-token chunks, the OOM guard
running, one cold request per depth and a 256-token reply at temperature 1.0.
The shipping flags are the preset's (patches 037 and 038 on); the base flags
have both off. Patch 036 is in both.

Decode, tok/s:

| prompt tokens | shipping flags | base flags | gain | rec | Kapitsa | llama.cpp | of rec | of Kapitsa | of llama.cpp |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,025 | 98.86 | 82.75 | +19.5 % | 126.5 | 124.3 | 77.5 | 78 % | 80 % | 128 % |
| 2,056 | 96.94 | 81.26 | +19.3 % | 123.0 | 121.3 | 75.1 | 79 % | 80 % | 129 % |
| 4,086 | 93.96 | 79.41 | +18.3 % | 116.5 | 119.2 | 73.4 | 81 % | 79 % | 128 % |
| 8,202 | 88.07 | 75.14 | +17.2 % | 107.0 | 113.2 | 69.7 | 82 % | 78 % | 126 % |
| 16,406 | 79.56 | 69.53 | +14.4 % | 95.2 | 106.2 | 63.7 | 84 % | 75 % | 125 % |
| 32,799 | 66.39 | 60.60 | +9.6 % | 79.4 | 85.9 | 53.8 | 84 % | 77 % | 123 % |
| 64,958 | 52.92 | 49.53 | +6.8 % | 56.5 | 66.2 | 40.5 | 94 % | 80 % | 131 % |
| 129,953 | 35.46 | 34.62 | +2.4 % | 35.2 | 45.7 | 23.8 | 101 % | 78 % | 149 % |

Prefill, tok/s (prompt tokens over time to first token):

| prompt tokens | base flags | shipping flags | TTFT base | rec | Kapitsa | llama.cpp | base of rec | base of Kapitsa | base of llama.cpp |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,025 | 948.1 | 956.2 | 1.1 s | 781 | 858 | 927 | 121 % | 111 % | 102 % |
| 2,056 | 947.1 | 931.2 | 2.2 s | 800 | 976 | 1015 | 118 % | 97 % | 93 % |
| 4,086 | 928.2 | 904.7 | 4.4 s | 791 | 960 | 978 | 117 % | 97 % | 95 % |
| 8,202 | 859.9 | 791.6 | 9.5 s | 760 | 907 | 910 | 113 % | 95 % | 94 % |
| 16,406 | 761.2 | 677.0 | 21.6 s | 699 | 808 | 785 | 109 % | 94 % | 97 % |
| 32,799 | 605.8 | 561.1 | 54.1 s | 581 | 651 | 602 | 104 % | 93 % | 101 % |
| 64,958 | 442.3 | 424.0 | 146.9 s | 427 | 464 | 382 | 104 % | 95 % | 116 % |
| 129,953 | 281.1 | 281.0 | 462.4 s | 262 | 300 | 199 | 107 % | 94 % | 141 % |

- Decode is ahead of llama.cpp at every depth, level with rec at 128K and
  behind Kapitsa by a fifth everywhere.
- Patches 037 and 038 are worth 19.5 % at 1K and 2.4 % at 128K: they remove
  a fixed cost per step, and the step grows from 10.1 to 28.2 ms.
- The gap is two terms. Against both engines the step is 2.1 to 2.2 ms
  longer at 1K (10.12 against 7.91 and 8.05 ms). Per token of context the
  step grows by 0.140 µs here, 0.159 for rec and 0.107 for Kapitsa. The 14
  full-attention layers hold 28.7 KB of BF16 keys and values per token, so
  those slopes are read rates of about 204, 180 and 267 GB/s if the other
  engines read the same bytes. This stack already reads deep context faster
  than rec; Kapitsa's lead at depth is the attention read.
- Prefill with the base flags is the reading of this stack: past rec at
  every depth, 93 to 97 % of Kapitsa past 1K, past llama.cpp at 1K and from
  32K and 93 to 97 % of it between. It matches the raw loop (859.9 at 8.2K
  against about 870 raw, 605.8 at 32.8K against 633).
- The shipping arm's lower prefill between 4K and 64K is a slow stretch of
  the host, not the flags. `deep/chunk-rates.txt`
  (`scripts/bench/prefill_chunk_rates.py` on the two server logs) has the
  rate of each chunk: the first chunk of the 4K, 8K, 16K, 32K and 64K
  requests runs at 919, 879, 828, 850 and 895 tok/s against 934 to 936 in
  the base arm, and the 128K request, the last one, matches the base arm on
  every chunk (932 / 441 / 279 / 162 against 930 / 441 / 279 / 162 at chunks
  0, 31, 63 and 124). A cost of the fused block would be on every request.
  The stretch is the first five minutes after the boot that followed the
  oQ8e and oQ6e arms (52 and 43 GB wired).

## Checkpoints

`gates/`, `preset/`, `fidelity/`. The preset as it ships (radix on, four
slots, 32K label, patches 037 and 038 on), one server per checkpoint.

| checkpoint | bits per weight | on disk | cold decode 959 / 7,601 / 30,422 | same request again | cold prefill | sampled 1K | greedy 1K | follow-up TTFT 3.8K | wired peak |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| int4 | 4.54 | 22.6 GB | 96.23 / 88.75 / 66.99 | 96.10 / 88.88 / 67.66 | 883.1 / 795.2 / 540.4 | 96.70 | 99.51 | 0.172 s | 32.6 GB |
| oQ6e | 6.6 | 32.8 GB | 66.76 / 62.39 / 52.00 | 66.65 / 62.83 / 52.25 | 832.5 / 743.5 / 527.8 | 66.68 | 67.64 | 0.219 s | 42.7 GB |
| oQ8e (default) | 8.5 | 42.3 GB | 62.73 / 58.44 / 49.20 | 62.01 / 58.44 / 49.76 | 795.2 / 674.9 / 487.2 | 62.53 | 63.38 | 0.222 s | 52.2 GB |
| oQ8e, one decode submission | 8.5 | 42.3 GB | 60.07 / 56.77 / 49.02 | 60.45 / 56.95 / 49.36 | 826.3 / 702.5 / 503.7 | 60.96 | 61.87 | 0.217 s | 52.2 GB |

- Gates with the preset's flags (`gates/`): int4 and oQ8e each pass
  `validate_capabilities.py` 3/3, `probe_thinking.py` (THINKING VERIFIED) and
  `probe_codegen.py` (STRONG, 8 of 8 tests), with zero tracebacks in either
  server log.
- Patch 038 on oQ8e, one pair: +4.4 / +2.9 / +0.4 % cold decode at 1K / 8K /
  32K, +2.6 % sampled and +2.4 % greedy at 1K. Smaller than on int4 because
  the oQ8e step is longer, never negative, so the preset's value of 5 covers
  every Flint checkpoint. The prefill columns of the pair differ by 4 % in a
  single cold reading per depth, inside what one cold request scatters.
- The follow-up turn of a 3.8K-token conversation starts in 0.17 s on int4
  (Kapitsa 0.13 s, llama.cpp 0.23 s, rec 0.41 s).

Teacher-forced fidelity (`scripts/eval/quant_fidelity.py` through the served
preset, eight documents, 1,984 scored positions, half prose and half code;
oQ8e is the reference dump):

| checkpoint | domain | NLL | PPL | top-1 | KL to oQ8e, mean | median | argmax equal to oQ8e |
|---|---|--:|--:|--:|--:|--:|--:|
| oQ8e | all | 2.1405 | 8.504 | 58.6 % | | | |
| oQ8e | prose | 2.6604 | 14.301 | 49.9 % | | | |
| oQ8e | code | 1.6207 | 5.057 | 67.2 % | | | |
| oQ6e | all | 2.1455 | 8.546 | 58.3 % | 0.0514 | 0.0050 | 93.7 % |
| oQ6e | prose | 2.6736 | 14.491 | 49.9 % | 0.0616 | 0.0083 | 91.6 % |
| oQ6e | code | 1.6174 | 5.040 | 66.7 % | 0.0412 | 0.0023 | 95.8 % |
| int4 | all | 2.1331 | 8.441 | 59.2 % | 0.2177 | 0.0491 | 83.0 % |
| int4 | prose | 2.6528 | 14.194 | 50.9 % | 0.2942 | 0.0814 | 77.7 % |
| int4 | code | 1.6134 | 5.020 | 67.4 % | 0.1412 | 0.0178 | 88.2 % |

- NLL is equal on the three checkpoints inside one standard error (0.073 to
  0.075): none of them predicts the held-out text worse.
- oQ6e is a close copy of oQ8e: KL 0.051 and 93.7 % of argmax tokens equal,
  inside the band of the 4-bit Qwen builds against their 8-bit references
  (KL 0.034 to 0.069, 89.8 to 91.3 % equal, `benchmarks/quality/fidelity/`).
- int4 is not a copy of the oQ builds: KL 0.218 and 83.0 % equal, three to
  six times the Qwen band, on text it predicts as well. The dumps cannot say
  which of the two is closer to the BF16 weights, which do not fit in 64 GB.
- So the choice is not made on these numbers. int4 is 55 % faster than oQ8e
  at 1K and 36 % faster at 30K and wires 19.6 GB less; oQ6e is 6.6 % faster
  and wires 9.5 GB less. The default stays oQ8e because the agentic cell
  was scored on it. An agentic cell on int4 is what decides a switch.

## The gap at 1K

| | ms per token | tok/s |
|---|--:|--:|
| published, rec and Kapitsa, greedy | 7.55 to 7.60 | 131.6 to 132.4 |
| weight reads alone at 178 to 239 GB/s | 6.1 to 8.2 | |
| raw two-kernel step, 100 operations and 400 MB buffers | 9.4 | 106 |
| raw two-kernel step, stock limits | 9.9 to 10.0 | 100 to 101 |
| raw mlx-lm step | 11.3 to 11.5 | 87 to 88 |
| served, three patches, 200 MB buffers | 9.75 | 102.6 |
| served, three patches (the preset) | 10.01 | 99.9 |
| served, patches 036 and 037 | 10.99 | 91.0 |
| served, view only | 12.38 | 80.8 |
| served, base | 13.35 | 74.9 |

Served decode moved from 57 % of the published rate to 69 % with patches
036 and 037 and to 76 % with patch 038, which puts the served step on the
raw step. A larger buffer adds two points and pays for them in cold
prefill. What is left is about 1.8 ms inside the graph itself: the routing
launches (0.85 ms), and attention and the routed experts above their
weight reads.

## Receipts

| folder | contents |
|---|---|
| `baseline/` | published-shape sweeps of the preset before this lane (int4 radix on and off, oQ8e), the validator, the chat-template check, the scale scan, `mlx_lm.generate` |
| `raw/` | decode census, attention census, weight-read census per checkpoint, fold probes, route replay probe, frozen view probe, host slack probe |
| `serve-census/` | served step split into host and GPU phases, radix on and off and with both changes |
| `ab/` | the thirteen served arms, streams, follow-up fidelity, the validator with both patches, the driver log |
| `buffers/` | host slack and the two-kernel fold under each command-buffer limit, with the middle limits; raw prefill by limit, by buffer-cache limit and inside one process, with driver logs; the eight served arms, their streams and the driver log |
| `submit/` | the nine served arms for decode submission in parts (alone, with 200 MB buffers, with 100 operations and 400 MB), their streams and the driver log |
| `gates/` | validator, thinking probe and codegen probe on int4 and oQ8e with the preset's flags, and the driver log |
| `preset/` | the shipped preset on int4, oQ6e and oQ8e, oQ8e with one decode submission, and one-second wired memory samples of each arm |
| `deep/` | the published context sweep through 128K on int4 with radix off and one slot, shipping flags against base flags, and the per-chunk prefill rates of both |
| `fidelity/` | teacher-forced dumps of oQ8e (the reference), oQ6e and int4 |
