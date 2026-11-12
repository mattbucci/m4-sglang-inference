# Flint against the published M4 Pro engines: windowed lookahead view (patch 036), fused MoE block (patch 037), decode submission in parts (patch 038), scheduled snapshot evaluation (patch 039)

Run dates 2026-11-10 and 2026-11-11. Stack: SGLang v0.5.23 + patches 001-039
(036 to 039 come out of this run), MLX 0.32.4, mlx-lm 0.32.3.dev17,
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
6. Timed arms run on a quiet GPU from the display A/B on. The server shares
   the GPU with the desktop: the compositor of a lit display, a media app's
   helper and the terminal this session draws in. Those arms run with the
   display asleep and the session idle (one background driver per stage, no
   tool call until it ends), and `scripts/bench/gpu_time.py` records the GPU
   time of every process per arm. Rows measured before that carry up to 5 %
   of decode and up to 10 % of deep prefill of desktop share, so they are
   compared only inside their own window.

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
| routing knock-out (`route_const`, `route_replay`) | launches | 9.02 to 9.09 ms with `comb` against 9.89 to 9.96 for `comb+gatek`; replaying real routes costs what constant routes cost | not a candidate; it prices the two routing launches per block at 0.85 ms a step, independent of which experts are read. One routing kernel per block was built next and is slower (`route1`, next row) |
| one-launch routing (`route1`) | launches | router logits and routing of a block in one Metal kernel, bit-identical: 10.76 to 10.81 against 9.91 to 9.96 ms raw at 1K and 11.91 to 11.98 against 11.04 to 11.09 at 8K, at 2,136 against 2,183 primitives; with a 256-thread group 10.88 to 10.91 and 12.07 to 12.10; the replay bound reads 9.14 to 9.16 and 10.16 to 10.24 | rejected: 8 to 10 % slower than the two launches it replaces. The 0.78 ms a step the routing launches cost at 1K (0.87 at 8K) stays unclaimed |
| radix cache off | host | 82.5 against 74.9 tok/s greedy before patch 036, 92.4 against 91.0 with both patches; a repeat of a 30K prompt answers in 51.5 s against 0.21 s | radix stays on: patch 036 removed the reason to turn it off |
| command-buffer limits | host slack | raw: stock limits lose 2.0 ms of step time to a 2 ms host delay, `MLX_MAX_MB_PER_BUFFER=200` loses 0.5, 100 operations with 400 MB loses none, 200 operations with 1,000 MB is slower than stock; served, eight mirrored arms: +9.9 / +10.3 / +11.1 % greedy at 1K for 200 MB / 100 operations and 400 MB / 100 operations and 1,000 MB, +5.2 to +5.5 % decode at 32K, cold prefill 6.5 to 7.9 % slower at 8K and 3.7 to 6.0 % slower at 32K, 0.5 to 1.2 GB more wired; on top of patch 038, six mirrored arms: 100 / 150 MB +2.2 / +1.7 % greedy at 1K for 3.8 / 4.9 % of the cold prefill at 8K | not a preset default: a trade, decode for cold prefill. The three settings tie in decode and 200 MB costs the least prefill and memory. Patch 038 takes three quarters of the same gain for free, so the limit stays an opt-in for decode-heavy work |
| decode submission in parts (patch 038) | host slack | raw: four early submissions recover 1.0 ms of a 1 ms host delay on stock limits and nothing once the limits are 100 operations and 400 MB; served, nine arms: five parts +7.5 % greedy and +7.8 % sampled at 1K, +8.2 / +8.9 / +2.7 % cold decode at 1K / 8K / 32K, +8.8 / +6.0 / +2.1 % at 1 / 2 / 4 streams, prefill inside the arm spread, wired memory equal, same greedy text; nine parts +8.1 % on one arm; five parts with 200 MB buffers +10.3 %, equal to the buffers alone; qwen36, four mirrored arms: five parts +4.4 % greedy, +3.4 % sampled, +5.0 / +5.0 / +1.5 % cold decode at 1K / 8K / 32K, prefill and memory equal, same greedy text | retained, flag `SGLANG_MLX_DECODE_SUBMIT_PARTS`, 5 in the `flint` preset. Clears the criterion on qwen36, and with patches 030 to 032 and five parts qwen36 passes the validator 3/3, thinking VERIFIED, vision, codegen and video STRONG (`gates/q36-fuse-sub5/`); 5 in the `qwen36` preset as well, see the next row |
| attention parts (knock-out census) | launches | at 1K attention costs 4.29 ms of the 11.47 ms raw mlx-lm step and its reads cost 3.12 ms at 200 GB/s; the query and output projections run at 218 to 243 GB/s, their read cost; head norms, the rotary embedding and a non-additive rest hold 1.17 ms | no fold carried: the folds over these operations (`qkv`, `front`, `mid`) measured inside noise. Recorded as the attention floor of this graph |
| deep ladder through 128K | like for like | on a quiet GPU decode is 102.5 tok/s at 1K to 37.5 at 128K: 81 to 90 % of rec through 32K, 98 % at 64K, 106 % at 128K, 78 to 84 % of Kapitsa at every depth; prefill 1,022 to 296 tok/s, 102 to 131 % of rec and 91 to 99 % of Kapitsa past 1K; with the display on the same arm read 98.9 to 35.5 tok/s decode; per token of context a step costs 0.131 µs here, 0.159 for rec, 0.107 for Kapitsa | measured. The fixed 1.7 to 1.9 ms per step is this graph's launches; the per-token term is the read rate of the shared keys and values, and a raw probe of the attention kernel at 32K to 128K is the next experiment |
| served prefill chunk (census) | host | host time is 3 to 4 ms of a 1.1 to 1.8 s chunk; with the radix cache on the tracked snapshot evaluates its 35 layers one at a time and the chunk costs 75.5 ms more at 8K (796 against 850 tok/s) and 51 to 87 ms more at 32K (573 to 585 against 602); the decode flags do not run on prefill chunks | measured. Served prefill is not host-bound; the per-layer evaluations are the radix-on cost, and patch 039 removes their waits |
| scheduled snapshot evaluation (patch 039, `snapshot-eval/`) | prefill under the radix cache | the snapshot of the tracked state schedules its arrays instead of waiting for each layer: Flint cold TTFT -3.5 % at 7.6K and -4.0 % at 30.4K, follow-up TTFT -3.4 to -5.0 %, extend -2.3 to -4.0 %, repeat -2.8 %, all with the arms of the two settings apart; served prefill 879 to 881 tok/s at 7.6K and 637 to 642 at 30.4K against 849 and 611 to 618; qwen36 cold -0.9 % at 7.6K, repeat -2.9 to -5.7 %, the rest inside the arm spread; decode unchanged; greedy text equal run for run | retained, always on, no flag. Cold follow-ups equal the previous tree's at KL 0 on both models and every capability gate passes |
| decode flags against prefill, by arm position (`slot/`) | prefill | six arms, base and shipping flags in mirrored order, six depths to 32K: prefill 983 / 933 / 917 / 832 / 705 / 565 tok/s on the shipping flags against 970 / 936 / 913 / 834 / 709 / 578 on base, with up to 9 % between two arms of one setting; decode +19 to +21 % to 8K, +16 % at 16K, +13 % at 32K | measured: the flags are prefill-neutral, and the deep prefill scatter follows neither the flags nor the position after a boot |
| display and session GPU share (`display/`, `gpu-time/`) | instrument | the compositor of the lit display holds 160 to 200 ms of GPU time per second and this session's terminal up to 43; display asleep with the session idle in both arms reads +3.5 to +4.6 % decode at every depth and +2.1 to +3.4 % cold prefill; a foreground wait in the session costs prefill at 8K to 32K a further 6 to 10 % | adopted as the protocol: timed arms run with the display asleep and the session idle, and each arm's GPU share is recorded |
| qwen36 preset settings (`quiet-settings/`) | launches and host slack | quiet GPU, five settings in mirrored order to 30K and three to 130K: patches 030 to 032 with five parts read 97.55 / 91.57 / 76.44 tok/s cold at 959 / 7,601 / 30,408 prompt tokens against 85.20 / 80.49 / 68.97 (+14.5 / +13.8 / +10.8 %), +14.0 % sampled and +15.0 % greedy at 1K, 61.84 against 56.86 at 65K (+8.8 %) and 45.43 against 42.75 at 130K (+6.3 %); the fusions alone +7.3 / +9.9 / +8.8 %, five parts alone +1.9 / +2.7 / +0.9 % and +1.2 / -0.1 % at depth; prefill, TTFT, follow-up TTFT and wired memory equal | the `qwen36` preset sets the three fusion flags and five parts; gates, teacher-forced fidelity and depth recall were already on file for the setting |
| 200 MB command buffers on the qwen36 preset (`quiet-settings/`) | host slack | on top of the fusions and five parts: +1.8 / +2.6 / +3.6 % cold decode at 1K / 7.6K / 30K, +2.1 % sampled, +1.6 % greedy, prefill not lower, 32.5 against 30.9 GB wired, follow-up TTFT 0.186 against 0.171 s | candidate: no arm at 65K and 130K yet |
| Flint checkpoints, quiet (`quiet-settings/`) | weight reads | the shipping preset with patches 036 to 039, two servers per checkpoint: int4 101.89 / 93.20 / 72.25 tok/s cold at 1K / 7.6K / 30.4K, oQ6e 69.70 / 64.88 / 53.60, oQ8e 65.50 / 62.00 / 51.65; wired peaks 31.84 / 42.03 / 51.51 GB | these are the rows of `docs/flint.md`; the default checkpoint still waits for an agentic cell on int4 |

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

Middle buffer limits on top of the preset (`buffers/`, six arms in mirrored
order, patches 036 to 038 with five parts in every arm, pooled per limit,
wired memory as the peak of one-second samples):

| limit | arms | sampled tok/s | greedy tok/s | greedy ms/token | greedy against stock | wired GB |
|---|--:|--:|--:|--:|--:|--:|
| stock, 50 MB | 2 | 96.21 | 98.64 | 10.14 | | 32.6 |
| 100 MB | 2 | 98.33 | 100.86 | 9.91 | +2.2 % | 32.8 |
| 150 MB | 2 | 98.48 | 100.33 | 9.97 | +1.7 % | 32.9 |

| limit | decode 1K / 8K / 32K | prefill 1K / 8K / 32K | repeat TTFT 1K / 8K / 32K |
|---|--:|--:|--:|
| stock, 50 MB | 96.03 / 88.28 / 68.06 | 880.7 / 796.6 / 548.4 | 0.157 / 0.171 / 0.212 s |
| 100 MB | 98.19 / 89.22 / 69.20 | 897.2 / 766.1 / 545.6 | 0.167 / 0.180 / 0.210 s |
| 150 MB | 98.91 / 90.22 / 69.78 | 898.7 / 758.0 / 531.9 | 0.174 / 0.187 / 0.215 s |

Greedy run ranges: stock 97.80 to 99.22, 100 MB 100.23 to 101.21, 150 MB
99.75 to 101.22. The middle limits keep the shape of the large ones: about
2 % of decode for 3.8 to 4.9 % of the cold prefill at 8K, in both arms of each
pair (760 and 772 tok/s at 100 MB, 760 and 756 at 150 MB, against 805 and 788
on stock limits). The raw probe, where 100 MB with four early submissions hid
2 ms of host work and raw prefill did not move, does not carry to serving.
The greedy text is the same in every arm. The stock limits stay.

Decode submission in parts on qwen36 (`submit/`, four arms in mirrored
order, the `qwen36` preset as shipped; the five-part arms log "4 of 40
layers submit early"):

| setting | arms | sampled tok/s | greedy tok/s | greedy ms/token | greedy against one submission | wired GB |
|---|--:|--:|--:|--:|--:|--:|
| one submission | 2 | 78.89 | 80.22 | 12.47 | | 31.5 |
| 5 parts | 2 | 81.58 | 83.73 | 11.94 | +4.4 % | 31.5 |

| setting | decode 1K / 8K / 32K | prefill 1K / 8K / 32K | repeat TTFT 1K / 8K / 32K |
|---|--:|--:|--:|
| one submission | 78.26 / 73.96 / 64.36 | 827.0 / 775.2 / 543.1 | 0.176 / 0.171 / 0.162 s |
| 5 parts | 82.13 / 77.62 / 65.34 | 844.5 / 774.2 / 543.5 | 0.186 / 0.184 / 0.149 s |

Greedy per arm: 80.09 and 80.35 for one submission, 83.72 and 83.75 for five
parts (runs 79.82 to 80.40 and 83.60 to 83.97). Five parts take 0.53 ms off
the qwen36 step: +4.4 % greedy, +3.4 % sampled, +5.0 / +5.0 / +1.5 % cold
decode at 1K / 8K / 32K. Prefill per arm is 842 and 812 tok/s at 1K for one
submission against 830 and 859 for five parts, so it does not move, and
neither does wired memory. The greedy text is the same in all four arms. The
host slack the patch hides is a property of the serving loop, not of Flint.

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

## Served prefill census

`serve-census/prefill-<arm>.jsonl`, `.summary.json`, `.sweep.json` and
`serve-census/prefill-census.txt` (`scripts/bench/serve_census/sitecustomize.py`
with `SERVE_CENSUS_PREFILL`, `scripts/bench/prefill_census_report.py`): one
row per served prefill chunk, int4 with the preset's flags, cold prompts of
7,601 and 30,422 tokens in 1,024-token chunks. A chunk splits into the
scheduler gap before its launch, the launch (graph build, the tracked
snapshot, the submission) and the wait for its token. The tracked snapshot is
the copy of the 35 sliding-window caches that the radix cache restores on a
hit, taken at every chunk boundary. Five arms, the preset first and last.
Each cell is the 8K reading, then the 32K reading.

| arm | prefill tok/s | ms per chunk | host | snapshot | submission | wait | wired peak, GB |
|---|--:|--:|--:|--:|--:|--:|--:|
| preset, first | 796.1 / 584.6 | 1,192.1 / 1,733.4 | 4.1 / 4.0 | 1,109.5 / 1,623.4 | 63.8 / 86.5 | 14.6 / 19.6 | 31.3 |
| radix cache off | 850.4 / 602.4 | 1,115.7 / 1,682.2 | 3.1 / 3.0 | none | 1,061.1 / 1,592.6 | 51.5 / 86.6 | 27.4 |
| stock MoE forward (`SGLANG_MLX_FUSE_FLINT_MOE=0`) | 799.3 / 576.5 | 1,187.2 / 1,757.7 | 3.9 / 3.8 | 1,103.5 / 1,650.4 | 65.9 / 88.6 | 13.9 / 14.9 | 31.2 |
| one slot | 819.2 / 582.2 | 1,158.3 / 1,740.6 | 3.8 / 3.7 | 1,085.4 / 1,642.4 | 49.5 / 75.7 | 19.5 / 18.8 | 28.2 |
| preset, last | 797.1 / 572.9 | 1,190.3 / 1,768.9 | 3.8 / 3.8 | 1,109.1 / 1,660.4 | 64.2 / 88.0 | 13.3 / 16.7 | 31.1 |

Host is the gap plus the graph build: 3 to 4 ms of a chunk that takes 1.1 to
1.8 s, under 0.4 %. A served prefill chunk is evaluation time and nothing
else; the scheduler adds 1 ms between chunks. The driver log printed its
summaries before the report counted the snapshot as evaluation time, so its
"host share" lines read 93 %; `prefill-census.txt` is the same rows under the
split used here.

With the radix cache on, the chunk's forward runs inside the snapshot. The
store (`kv_cache/auxiliary_state.py`) evaluates one layer's copy at a time,
and each `mx.eval` runs the forward up to that layer as its own submission
and waits for it, so one chunk becomes 35 submit-and-wait cycles and a short
tail. Against the radix-off arm the chunk costs 75.5 ms more at 8K (6.8 %)
and 51 to 87 ms more at 32K (3.0 to 5.2 %): a fixed cost per chunk of about
2 ms per cycle, not one that grows with depth. That is the whole difference
between the radix-on and radix-off prefill rows of this file.

The decode flags do not run on a prefill chunk. Patch 037 takes its fused
call only for a one-token step and keeps the stock forward otherwise, and
patch 038 submits early only on one-token steps; the stock-forward arm reads
799.3 / 576.5 against 796.1 to 797.1 and 572.9 to 584.6 for the preset. The
lower prefill of the shipping arm in the deep ladder is therefore not the
flags.

One slot reads 2.8 % more at 8K and the same at 32K, a single arm against two
preset arms that agree within 0.2 % at 8K and 2 % at 32K. Radix off is 2 to
5 % under the raw loop of the section above at slightly shallower prompts;
that remainder is not attributed here.

The candidate this prices is one evaluation for all 35 layers of a snapshot:
the same state, stored at the same boundary, as one submission. The section
"Scheduled snapshot evaluation" measures it against a second form, and the
waits turn out to be worth 3 to 4 % of a cold prefill.

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
have both off. Patch 036 is in both. The display was on for these arms; the
quiet rerun of the shipping arm is the section "Quiet ladder" below and is
the row to compare with the published ones.

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
- The shipping arm's lower prefill between 4K and 64K is not the flags.
  `deep/chunk-rates.txt`
  (`scripts/bench/prefill_chunk_rates.py` on the two server logs) has the
  rate of each chunk: the first chunk of the 4K, 8K, 16K, 32K and 64K
  requests runs at 919, 879, 828, 850 and 895 tok/s against 934 to 936 in
  the base arm, and the 128K request, the last one, matches the base arm on
  every chunk (932 / 441 / 279 / 162 against 930 / 441 / 279 / 162 at chunks
  0, 31, 63 and 124). A cost of the fused block would be on every request.
  The slot arms below rule out the position after a boot, and the display
  A/B finds the cause: other clients of the GPU, the lit display and this
  session's terminal among them.

## Decode flags against prefill, by arm position

`slot/`: six arms in one window, int4, one request slot, radix cache off,
`CTX=131136 MEM_FRAC=0.5`, six depths to 32K, base flags and shipping flags
in mirrored order (base, shipping, base, shipping, shipping, base) after a
discarded warm-up arm (`swap.json`). The question: is the shipping arm's
lower prefill in the deep ladder the flags, or the arm's place after a boot.
Display on, wired peak 27.2 GB, no traceback.

| prompt tokens | 1,025 | 2,056 | 4,086 | 8,202 | 16,406 | 32,799 |
|---|--:|--:|--:|--:|--:|--:|
| prefill tok/s, base flags, mean of three | 970.4 | 935.8 | 912.6 | 834.3 | 709.0 | 577.6 |
| prefill tok/s, shipping flags, mean of three | 982.8 | 933.1 | 917.3 | 831.8 | 704.8 | 564.8 |
| shipping against base | +1.3 % | -0.3 % | +0.5 % | -0.3 % | -0.6 % | -2.2 % |
| spread inside the base arms | 4.0 % | 0.7 % | 0.7 % | 5.9 % | 6.5 % | 4.5 % |
| spread inside the shipping arms | 0.1 % | 1.7 % | 1.0 % | 4.1 % | 9.1 % | 6.7 % |
| decode tok/s, base flags | 80.66 | 79.75 | 78.73 | 73.72 | 68.44 | 59.53 |
| decode tok/s, shipping flags | 97.14 | 96.86 | 93.84 | 87.96 | 79.50 | 67.11 |
| shipping against base | +20.4 % | +21.5 % | +19.2 % | +19.3 % | +16.2 % | +12.7 % |

The flags do not move prefill: the two means sit inside the spread of either
setting at every depth. Decode separates by 13 to 21 % with at most 2.4 % of
spread inside a setting. The deep prefill scatter, up to 9 % between two arms
on the same flags, follows neither the flags nor the position: at 16K the
slow arms are the second, fifth and sixth, at 32K all but the third and
fourth. `slot/chunk-rates.txt` has the rate of each chunk: a slow request is
slow from its first chunk (844 to 874 against 914 to 927 tok/s on chunk 0 of
the 16K request), so the cause is outside the request.

## Desktop and session GPU share

`display/`, `gpu-time/`. The server is not the only client of the GPU.
`ioreg` reports accumulated GPU time per process (`IOGPUDeviceUserClient`,
`AppUsage`), and `scripts/bench/gpu_time.py` samples it every five seconds
and reports ms of GPU time per wall second over a set of windows. While the
slot arms ran, three other clients held GPU time: the compositor of the lit
display (4,608 x 2,592 at 120 Hz), a media app's helper, and the terminal
this session draws in, whose status line animates during a foreground wait.

Four arms on the slot arms' server settings and shipping flags: display on
and asleep (`pmset displaysleepnow`), each once with this session blocked in
a foreground wait (a) and once with the session idle behind a background
driver (b). Wired peak 26.2 GB, no traceback.

| prompt tokens | 1,025 | 2,056 | 4,086 | 8,202 | 16,406 | 32,799 |
|---|--:|--:|--:|--:|--:|--:|
| prefill tok/s, display on, session waiting (on-a) | 981.1 | 934.8 | 906.5 | 799.1 | 675.8 | 549.8 |
| prefill tok/s, display asleep, session waiting (asleep-a) | 1,019.6 | 972.8 | 941.6 | 833.3 | 716.1 | 604.4 |
| prefill tok/s, display on, session idle (on-b) | 988.8 | 941.8 | 920.2 | 859.3 | 746.3 | 592.2 |
| prefill tok/s, display asleep, session idle (asleep-b) | 1,019.8 | 972.9 | 948.8 | 879.6 | 761.8 | 612.2 |
| asleep-b against on-b | +3.1 % | +3.3 % | +3.1 % | +2.4 % | +2.1 % | +3.4 % |
| on-b against on-a | +0.8 % | +0.7 % | +1.5 % | +7.5 % | +10.4 % | +7.7 % |
| decode tok/s, on-a | 97.23 | 96.10 | 94.49 | 87.39 | 79.15 | 66.63 |
| decode tok/s, asleep-a | 102.36 | 101.30 | 97.92 | 91.64 | 82.99 | 70.25 |
| decode tok/s, on-b | 97.91 | 97.15 | 95.93 | 89.43 | 80.89 | 68.40 |
| decode tok/s, asleep-b | 102.42 | 101.19 | 99.44 | 93.15 | 83.76 | 71.17 |
| asleep-b against on-b | +4.6 % | +4.2 % | +3.7 % | +4.2 % | +3.5 % | +4.0 % |
| on-b against on-a | +0.7 % | +1.1 % | +1.5 % | +2.3 % | +2.2 % | +2.7 % |

GPU time per wall second in each window, ms (`gpu-time/share.txt`):

| window | server | compositor | other clients | of which |
|---|--:|--:|--:|---|
| on-a | 693 | 196 | 88 | media app helper 63, terminal 25 |
| asleep-a | 849 | 0 | 110 | media app helper 66, terminal 43 |
| on-b | 768 | 158 | 44 | media app helper 40, terminal 4 |
| asleep-b | 930 | 0 | 42 | media app helper 42 |

- With the session idle in both arms, the display asleep is worth 3.5 to
  4.6 % of decode at every depth and 2.1 to 3.4 % of cold prefill.
- A foreground wait in the session costs a prefill of 8K to 32K a further
  6 to 10 % with the display on (5.6 / 6.4 / 1.3 % with it asleep) and
  decode up to 2.7 %. That is the deep prefill scatter of the slot arms and
  the slow stretch of the deep ladder.
- A share is not a cost. The compositor's 16 to 20 % of GPU time costs 2 to
  5 % of throughput: a desktop client's accounted time grows while a long
  compute kernel runs beside it. The share says who was on the GPU; the A/B
  prices it.
- Protocol from here on: timed arms run with the display asleep and the
  session idle, the share of every arm is recorded, and settings are
  compared inside one window. The media app's helper stays; it is the
  user's app and its cost is not priced.

## Quiet ladder

`quiet/ladder.json`: the deep ladder's shipping arm again under the
protocol. Int4, the preset's flags, one request slot, radix cache off,
`CTX=131136 MEM_FRAC=0.5`, 1,024-token chunks, the OOM guard running, one
cold request per depth and a 256-token reply at temperature 1.0. Wired peak
27.7 GB, no traceback. This is the row to hold against the published ones.

Decode, tok/s:

| prompt tokens | quiet | display on | rec | Kapitsa | llama.cpp | of rec | of Kapitsa | of llama.cpp |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,025 | 102.45 | 98.86 | 126.5 | 124.3 | 77.5 | 81 % | 82 % | 132 % |
| 2,056 | 101.68 | 96.94 | 123.0 | 121.3 | 75.1 | 83 % | 84 % | 135 % |
| 4,086 | 98.44 | 93.96 | 116.5 | 119.2 | 73.4 | 84 % | 83 % | 134 % |
| 8,202 | 92.81 | 88.07 | 107.0 | 113.2 | 69.7 | 87 % | 82 % | 133 % |
| 16,406 | 83.32 | 79.56 | 95.2 | 106.2 | 63.7 | 88 % | 78 % | 131 % |
| 32,799 | 71.40 | 66.39 | 79.4 | 85.9 | 53.8 | 90 % | 83 % | 133 % |
| 64,958 | 55.31 | 52.92 | 56.5 | 66.2 | 40.5 | 98 % | 84 % | 137 % |
| 129,953 | 37.48 | 35.46 | 35.2 | 45.7 | 23.8 | 106 % | 82 % | 157 % |

Prefill, tok/s, and time to first token:

| prompt tokens | quiet | TTFT | rec | Kapitsa | llama.cpp | of rec | of Kapitsa | of llama.cpp |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,025 | 1,022.5 | 1.0 s | 781 | 858 | 927 | 131 % | 119 % | 110 % |
| 2,056 | 964.6 | 2.1 s | 800 | 976 | 1015 | 121 % | 99 % | 95 % |
| 4,086 | 952.8 | 4.3 s | 791 | 960 | 978 | 120 % | 99 % | 97 % |
| 8,202 | 878.7 | 9.3 s | 760 | 907 | 910 | 116 % | 97 % | 97 % |
| 16,406 | 738.1 | 22.2 s | 699 | 808 | 785 | 106 % | 91 % | 94 % |
| 32,799 | 593.5 | 55.3 s | 581 | 651 | 602 | 102 % | 91 % | 99 % |
| 64,958 | 447.7 | 145.1 s | 427 | 464 | 382 | 105 % | 96 % | 117 % |
| 129,953 | 296.1 | 438.9 s | 262 | 300 | 199 | 113 % | 99 % | 149 % |

- Decode is 3.6 to 7.5 % above the display-on ladder at every depth: 81 to
  90 % of rec through 32K, 98 % at 64K and past it at 128K, 78 to 84 % of
  Kapitsa throughout, 131 to 157 % of llama.cpp.
- The step is 9.76 ms at 1K against 7.91 for rec and 8.05 for Kapitsa, a
  fixed 1.7 to 1.9 ms. Per token of context it grows by 0.131 µs, against
  0.159 for rec and 0.107 for Kapitsa.
- Prefill is past rec at every depth and within 1 to 4 % of Kapitsa except
  at 16K and 32K (91 %). A 128K prompt reaches its first token in 439 s.
- The warm sweep before the ladder (`quiet/warm.json`) reads 1,020.4 tok/s
  prefill and 101.71 decode at 1K, 880.7 and 91.55 at 8K.
- One client was left on the GPU. The media app's helper holds 15 to 24 ms
  per second through the 8K request, 38 at 16K, 64 at 32K, 101 at 64K and
  206 during the 128K prefill (`gpu-time/share-quiet-ladder-by-depth.txt`).
  The 16K and 32K prefill readings sit 3 % under the display A/B's quiet
  arm (738 against 762, 594 against 612), single arms each.

## One-launch routing

`raw/one-launch-routing-probe.jsonl`, `raw/one-launch-routing-probe-tg256.jsonl`
and the kernel self tests next to them. The knock-out prices the two routing
launches of a block (the router logits, then mlx-lm's routing kernel) at
0.8 ms a step. `route1` is the candidate that follows: one Metal kernel
whose tiles compute the logits the way the stock matrix-vector kernel does,
leave them in threadgroup memory, and run mlx-lm's routing on them in the
first SIMD group. Indices and weights are bit-identical to the stock chain
(40 trials per dtype, 64 teacher-forced steps). Pipelined ms per step, the
arm list forward and reversed:

| arm | primitives | 1K | 8K |
|---|--:|--:|--:|
| comb+gatek (patch 037) | 2,183 | 9.91 / 9.96 | 11.04 / 11.09 |
| comb+route1, 1,024-thread group | 2,136 | 10.76 / 10.81 | 11.91 / 11.98 |
| comb+route_replay (the bound) | 2,089 | 9.14 / 9.15 | 10.16 / 10.24 |
| comb+gatek, second run | 2,183 | 9.93 / 9.93 | 11.10 / 11.14 |
| comb+route1, 256-thread group | 2,136 | 10.91 / 10.88 | 12.10 / 12.07 |
| comb+route_replay, second run | 2,089 | 9.14 / 9.16 | 10.20 / 10.24 |

One launch is 0.85 to 0.98 ms a step slower than two at 1K and 0.87 to
1.0 ms at 8K, with either threadgroup. The bound does not move: 0.78 ms at
1K and 0.87 at 8K separate the shipping step from replayed routes. Rejected
and not carried to serving. A kernel whose reduction crosses rows has to run
in one threadgroup, which takes the logits off the many groups the stock
kernel spreads them over; that reading is not verified. The routing cost
stays unclaimed.

## Scheduled snapshot evaluation (patch 039)

`snapshot-eval/`. With the radix cache on, every prefill chunk stores the
tracked state of the windowed and recurrent layers so that a later request
can resume from it. The store ran one synchronous `mx.eval` per layer, and
the first of them pulled the chunk's forward with it. The served prefill
census counts 35 of those evaluations per chunk on Flint (30 on qwen36): the
GPU finishes a layer, idles while the host is woken, and starts the next.

Two forms of the fix, each a tree built from the same pin and patches
001-038:

- `one`: collect the arrays of all layers and evaluate them in one call.
- `async`: keep the loop and schedule each layer's arrays with
  `mx.async_eval`. Nothing in the chunk waits; the chunk's token is the one
  read. The snapshot holds the same arrays, bound at the same moment.

Both store the same values, and the greedy text of every run is equal across
all arms, hash for hash, on both models.

First pass, display on (`snapshot-eval/display-on/`): `one` against stock in
mirrored order. Flint cold TTFT -3.2 % at 7.6K, the follow-up's first request
at 15K -4.0 %, extend at 7.6K -5.3 %, each with the arms apart; cold at 30.4K
-1.6 % with the arms overlapping. qwen36 moves inside its spread except the
repeat (-3.2 and -2.8 %). That pass predates the quiet protocol.

Second pass on the quiet GPU: three trees, int4 and qwen36, the presets'
flags, radix cache on, arms in the order of the columns. "apart" means no arm
of one setting reaches into the range of the other. TTFT in seconds, decode
in ms per step; the percentage is the mean of a form against the mean of
stock.

Flint int4:

| | stock | async | one | async | stock | async against stock | one against stock |
|---|--:|--:|--:|--:|--:|--:|--:|
| cold, 7,601 tokens | 8.962 | 8.640 | 8.677 | 8.657 | 8.967 | -3.5 %, apart | -3.2 %, apart |
| cold, 30,422 | 49.26 | 47.77 | 47.65 | 47.40 | 49.84 | -4.0 %, apart | -3.8 %, apart |
| follow-up at 3,813, first request | 4.203 | 4.272 | 4.223 | 4.203 | 4.347 | -0.9 %, overlap | -1.2 %, overlap |
| follow-up at 3,813, second | 0.1628 | 0.1571 | 0.1572 | 0.1570 | 0.1624 | -3.4 %, apart | -3.3 %, apart |
| follow-up at 15,228, first request | 20.58 | 19.97 | 19.88 | 19.80 | 20.84 | -4.0 %, apart | -4.0 %, apart |
| follow-up at 15,228, second | 0.3176 | 0.3017 | 0.3064 | 0.3002 | 0.3163 | -5.0 %, apart | -3.3 %, apart |
| extend to 3,799 | 4.203 | 4.102 | 4.083 | 4.075 | 4.167 | -2.3 %, apart | -2.4 %, apart |
| repeat at 3,799 | 0.1139 | 0.1116 | 0.1100 | 0.1113 | 0.1154 | -2.8 %, apart | -4.1 %, apart |
| decode at 3,799, ms | 9.77 | 9.73 | 9.66 | 9.76 | 9.83 | -0.6 % | -1.4 % |
| extend to 7,599 | 4.869 | 4.674 | 4.696 | 4.655 | 4.852 | -4.0 %, apart | -3.4 %, apart |
| repeat at 7,599 | 0.1816 | 0.1764 | 0.1764 | 0.1769 | 0.1820 | -2.8 %, apart | -3.0 %, apart |
| decode at 7,599, ms | 10.29 | 10.21 | 10.17 | 10.18 | 10.28 | -0.9 % | -1.1 % |

qwen36:

| | stock | async | one | one | async | stock | async against stock | one against stock |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| cold, 7,601 tokens | 8.818 | 8.734 | 8.792 | 8.828 | 8.719 | 8.786 | -0.9 %, apart | +0.1 %, overlap |
| cold, 30,408 | 49.49 | 49.50 | 48.50 | 48.61 | 47.81 | 48.39 | -0.6 %, overlap | -0.8 %, overlap |
| extend to 3,799 | 4.057 | 4.041 | 4.027 | 3.979 | 3.963 | 4.038 | -1.1 %, overlap | -1.1 %, apart |
| repeat at 3,799 | 0.1222 | 0.1146 | 0.1159 | 0.1181 | 0.1154 | 0.1217 | -5.7 %, apart | -4.1 %, apart |
| decode at 3,799, ms | 11.89 | 11.63 | 11.59 | 11.53 | 11.54 | 11.74 | -1.9 %, apart | -2.2 %, apart |
| extend to 7,583 | 4.499 | 4.493 | 4.496 | 4.458 | 4.412 | 4.447 | -0.5 %, overlap | +0.1 %, overlap |
| repeat at 7,583 | 0.1416 | 0.1376 | 0.1366 | 0.1381 | 0.1373 | 0.1416 | -2.9 %, apart | -3.0 %, apart |
| decode at 7,583, ms | 12.12 | 12.12 | 12.25 | 12.07 | 12.17 | 12.08 | +0.4 %, overlap | +0.5 %, overlap |

The prefill census of the same arms (`*.summary.json`), Flint:

| | stock | async | one | async | stock |
|---|--:|--:|--:|--:|--:|
| synchronous evaluations per chunk | 35 | 0 | 1 | 0 | 35 |
| ms per chunk, 7,601 tokens | 1,118.7 | 1,078.5 | 1,083.1 | 1,080.6 | 1,119.4 |
| ms per chunk, 30,422 tokens | 1,641.0 | 1,591.1 | 1,587.0 | 1,578.9 | 1,660.1 |
| served prefill tok/s, 7,601 | 849.3 | 881.0 | 877.2 | 879.3 | 848.8 |
| served prefill tok/s, 30,422 | 618.0 | 637.3 | 639.0 | 642.3 | 610.9 |

- A chunk loses 36 to 41 ms at 7.6K: 35 waits at about 1.1 ms each. That is
  3.5 % of a 1,024-token chunk on Flint. qwen36 takes 30 waits per chunk of
  4,096 tokens, 4.4 s long, so the same waits are under 1 % of its cold
  prefill and only its cache hits show them.
- The two forms tie: under 0.5 % apart on the Flint cold rows, and trading
  places by 1 to 2 % on the cache-hit rows, with a single `one` arm on
  Flint. `async` is one line, leaves no synchronous evaluation in the chunk,
  and is the only form whose cold gain on qwen36 stands apart from stock.
- Decode does not move on Flint. The qwen36 decode row at 3.8K (-1.9 %) is
  not claimed: the 7.6K row overlaps.
- GPU time per wall second (`gpu-time/share.txt`): the server holds 697 and
  718 ms in the stock arms and 745 to 752 in the others, the media app's
  helper 36 to 42, the display 0. The terminal shows 22 ms in the first
  stock arm, which the last stock arm (none) agrees with to 0.1 % at 7.6K
  and 1.2 % at 30.4K. qwen36 arms: server 674 to 699, helper 46 to 48.

Decision: `async` is patch 039, always on, no flag.

Fidelity and gates on the patched tree (`snapshot-eval/gates/`), radix cache
on, both models:

- `probe_followup_fidelity.py`, depths 1,024 and 6,144, tails 32 and 1,024,
  two seeds, teacher-forced. The patched tree's cold follow-ups against the
  previous tree's cold follow-ups: every step of all eight cases agrees and
  the KL is 0 at every step, on Flint and on qwen36. The hit-against-cold
  statistics are the same numbers in both trees (Flint: KL mean 6.2e-03 at
  tail 32 and 4.4e-03 at tail 1,024; qwen36: 9.8e-03 and 9.0e-03).
- Flint (oQ8e, the preset): validator 3/3, thinking VERIFIED, codegen STRONG.
  qwen36: validator 3/3, thinking VERIFIED, vision, codegen and video STRONG.
  No traceback in any server log.
- `scripts/eval/test_snapshot_async_eval_unit.py` (CPU): 13 checks pass on
  the patched tree; the four scheduling checks fail on the previous tree.

## Checkpoints

`gates/`, `preset/`, `fidelity/`. The preset as it ships (radix on, four
slots, 32K label, patches 037 and 038 on), one server per checkpoint, display
on. The same table on a quiet GPU with patch 039 is in the next section.

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

## Quiet settings: Flint checkpoints and the qwen36 preset

`quiet-settings/`. Display asleep and session idle for the whole queue, one
server per arm, 26 arms in 1 h 50 min, zero tracebacks. Every arm booted from
the tree with patches 001 to 039 (`tree.txt`). The compositor holds 0 ms of
GPU time per second in every arm and the server 904 to 935 (`driver.txt`,
the row after each arm). `table.txt` is the summary these tables are cut
from. Each Flint checkpoint and the qwen36 preset open with one arm that pages
the checkpoint in and is not in the tables; the deep part has none.

Flint checkpoints on the shipping preset, mean of two servers per checkpoint
(prompts of 959 / 7,601 / 30,422 tokens, 256-token replies at the
checkpoint's own sampling):

| checkpoint | cold decode tok/s | cold prefill tok/s | TTFT s | sampled 1K | greedy 1K | follow-up TTFT 3.8K / 15.2K | same request again, TTFT | wired peak |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| int4 | 101.89 / 93.20 / 72.25 | 945.2 / 878.0 / 606.4 | 1.01 / 8.66 / 50.20 | 101.99 | 104.47 | 0.160 / 0.297 s | 0.16 / 0.16 / 0.20 s | 31.84 GB |
| oQ6e | 69.70 / 64.88 / 53.60 | 867.8 / 837.0 / 568.5 | 1.11 / 9.08 / 53.56 | 69.41 | 70.24 | 0.208 / 0.363 s | 0.20 / 0.20 / 0.22 s | 42.03 GB |
| oQ8e (default) | 65.50 / 62.00 / 51.65 | 877.3 / 843.9 / 566.7 | 1.09 / 9.01 / 53.71 | 65.75 | 66.53 | 0.208 / 0.366 s | 0.22 / 0.21 / 0.22 s | 51.51 GB |

- Decode per arm: int4 102.77 / 93.55 / 71.79 and 101.01 / 92.85 / 72.72,
  oQ6e 69.63 / 64.93 / 53.16 and 69.78 / 64.82 / 54.04, oQ8e 65.38 / 62.15 /
  51.46 and 65.63 / 61.84 / 51.84. Two servers of one checkpoint agree inside
  1.7 % on decode. Cold prefill at 30K scatters 5 to 6 % between two
  servers (622.2 and 590.6 on int4), so the TTFT column is good to that.
- int4 against oQ8e: 56 % faster at 1K, 50 % at 7.6K, 40 % at 30K, in 19.7 GB
  less wired memory. oQ6e against oQ8e: 6.4 % faster at 1K in 9.5 GB less.
- Against the display-on table of the Checkpoints section every checkpoint
  reads higher: the quiet GPU and patch 039 together are worth 3 to 8 % of
  decode and 4 to 25 % of cold prefill, most on oQ8e's prefill (10 / 25 /
  16 %).

The `qwen36` preset as it ships (radix on, one slot, 4,096-token chunks),
five settings in the order base, parts, fusions, both, both with 200 MB
buffers, then the same five reversed. Prompts of 959 / 7,601 / 30,408
tokens; means of the two arms:

| setting | cold decode tok/s | against base | sampled 1K | greedy 1K | cold prefill tok/s | TTFT 30K | follow-up TTFT 3.8K | wired peak, both arms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| base: one submission, stock ops | 85.20 / 80.49 / 68.97 | | 85.76 | 87.25 | 902.5 / 834.6 / 585.8 | 51.91 s | 0.174 s | 30.93, 30.71 GB |
| five parts (patch 038) | 86.81 / 82.66 / 69.56 | +1.9 / +2.7 / +0.9 % | 87.23 | 89.09 | 915.7 / 837.5 / 587.1 | 51.81 s | 0.173 s | 30.87, 30.72 GB |
| fusions (patches 030 to 032) | 91.40 / 88.45 / 75.04 | +7.3 / +9.9 / +8.8 % | 93.78 | 95.46 | 924.7 / 827.4 / 582.2 | 52.24 s | 0.171 s | 30.89, 30.89 GB |
| fusions and five parts | 97.55 / 91.57 / 76.44 | +14.5 / +13.8 / +10.8 % | 97.76 | 100.38 | 923.3 / 848.6 / 590.1 | 51.54 s | 0.171 s | 30.88, 30.82 GB |
| fusions, five parts, `MLX_MAX_MB_PER_BUFFER=200` | 99.34 / 93.96 / 79.22 | +16.6 / +16.7 / +14.9 % | 99.84 | 101.98 | 935.8 / 854.1 / 590.1 | 51.53 s | 0.186 s | 32.47, 32.48 GB |

| setting | arm a, cold decode | arm b, cold decode | sampled a, b | greedy a, b |
|---|--:|--:|--:|--:|
| base | 85.87 / 80.37 / 68.39 | 84.53 / 80.61 / 69.55 | 85.69, 85.83 | 87.12, 87.37 |
| five parts | 86.93 / 82.62 / 69.58 | 86.70 / 82.71 / 69.53 | 87.14, 87.32 | 89.19, 88.99 |
| fusions | 90.54 / 88.13 / 74.71 | 92.26 / 88.77 / 75.38 | 93.90, 93.65 | 95.73, 95.18 |
| fusions and five parts | 97.47 / 91.58 / 76.90 | 97.62 / 91.57 / 75.98 | 97.82, 97.71 | 100.36, 100.40 |
| with 200 MB buffers | 99.25 / 93.66 / 79.70 | 99.44 / 94.26 / 78.75 | 99.90, 99.79 | 102.02, 101.95 |

qwen36 at depth (`CTX=131136 MEM_FRAC=0.5 CHUNKED=1024`, radix off, prompts
of 64,919 and 129,887 tokens, 256-token replies), order base, parts, both,
both, parts, base:

| setting | decode 65K / 130K | against base | arm a | arm b | cold prefill tok/s | TTFT s | wired peak, both arms |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 56.86 / 42.75 | | 57.21 / 42.86 | 56.51 / 42.65 | 472.9 / 325.7 | 137.4 / 398.8 | 29.31, 28.95 GB |
| five parts | 57.56 / 42.69 | +1.2 / -0.1 % | 57.23 / 42.97 | 57.89 / 42.41 | 487.5 / 326.6 | 133.2 / 397.7 | 28.56, 29.25 GB |
| fusions and five parts | 61.84 / 45.43 | +8.8 / +6.3 % | 62.18 / 45.27 | 61.50 / 45.59 | 488.1 / 327.3 | 133.0 / 396.9 | 29.26, 29.33 GB |

- The fusions and five parts together win at every depth and in both arm
  orders, and their lower arm is above the higher base arm everywhere:
  +14.5 / +13.8 /
  +10.8 % cold decode to 30K, +14.0 % sampled and +15.0 % greedy at 1K,
  +8.8 % at 65K, +6.3 % at 130K. A sampled step at 1K goes from 11.66 to
  10.23 ms.
- The two levers compound. Five parts alone are worth +1.9 / +2.7 / +0.9 %
  and nothing at depth; on top of the fusions they are worth +6.7 / +3.5 /
  +1.9 %, because the host work between two submissions is a larger share of
  a shorter step. The display-on pair of the Served A/B section read +5.0 /
  +5.0 / +1.5 % for five parts alone, so that slack is smaller on a quiet GPU.
- Prefill does not move. Per arm at 1K the five settings read 889.6 to 936.8
  tok/s with 25.7 between the two base arms, at 30K 577.1 to 594.1; the deep
  base arm that opens the window reads 458.5 tok/s at 65K against 486.9 to
  488.9 for the five arms after it, a first-in-window effect and not a
  setting. Mean TTFT at 30K is 51.5 to 52.2 s in every setting, and the
  follow-up starts in 0.17 s in the first four.
- Wired memory is equal across the first four settings (30.7 to 30.9 GB to
  30K, 28.6 to 29.3 GB at depth).
- 200 MB command buffers on top add +1.8 / +2.6 / +3.6 % cold decode, +2.1 %
  sampled and +1.6 % greedy, with prefill not lower (unlike Flint, where the
  same limit costs 6.5 % of the cold prefill at 8K). They cost 1.6 GB of
  wired memory, 15 ms on a follow-up's TTFT and 13 to 14 ms on a repeated
  request at 1K and 8K. No arm at 65K and 130K yet, so they stay a
  candidate.

Decision: the `qwen36` preset sets `SGLANG_MLX_FUSE_MOE_ROUTER`,
`SGLANG_MLX_FUSE_GDN_FRONT`, `SGLANG_MLX_FUSE_DECODE_LINEARS` and five decode
parts. What a default needs was on file for this setting: the five
capability gates (`gates/q36-fuse-sub5/`), teacher-forced fidelity of the
fusions (KL 0.0211 to a float32 yardstick against 0.0165 for the stock
decode path and 0.0194 for the stock prefill path, perplexity 5.227 against
5.244 through the server) and depth recall 6/6 at 251K
(`../micro/VERDICT.md`); five parts are bit-identical. `=0` on a fusion flag
and `SGLANG_MLX_DECODE_SUBMIT_PARTS=1` restore the stock step.

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
launches (0.8 ms, which one launch per block does not recover), and
attention and the routed experts above their weight reads. On a quiet GPU
the preset's sampled step at 1K is 9.76 ms, 102.5 tok/s, 81 % of rec.

## Receipts

| folder | contents |
|---|---|
| `baseline/` | published-shape sweeps of the preset before this lane (int4 radix on and off, oQ8e), the validator, the chat-template check, the scale scan, `mlx_lm.generate` |
| `raw/` | decode census, attention census, weight-read census per checkpoint, fold probes, route replay probe, one-launch routing probe and its kernel self tests, frozen view probe, host slack probe |
| `serve-census/` | served step split into host and GPU phases, radix on and off and with both changes; the per-chunk prefill census of five arms with its report and driver log |
| `ab/` | the thirteen served arms, streams, follow-up fidelity, the validator with both patches, the driver log |
| `buffers/` | host slack and the two-kernel fold under each command-buffer limit, with the middle limits; raw prefill by limit, by buffer-cache limit and inside one process, with driver logs; the eight served arms, their streams and the driver log; the six middle-limit arms on top of the preset with wired samples and their driver log |
| `submit/` | the nine served arms for decode submission in parts (alone, with 200 MB buffers, with 100 operations and 400 MB), their streams and the driver log; the four qwen36 arms |
| `gates/` | validator, thinking probe and codegen probe on int4 and oQ8e with the preset's flags, and the driver log; `q36-fuse-sub5/`: the five gates on qwen36 with patches 030 to 032 and five decode parts |
| `quiet-settings/` | quiet GPU: two arms per Flint checkpoint on the shipping preset, ten arms of five settings on the `qwen36` preset, six deep arms of three settings at 65K and 130K, one-second wired memory samples per arm, `table.txt`, `tree.txt`, the driver log |
| `preset/` | the shipped preset on int4, oQ6e and oQ8e, oQ8e with one decode submission, and one-second wired memory samples of each arm |
| `deep/` | the published context sweep through 128K on int4 with radix off and one slot, shipping flags against base flags, and the per-chunk prefill rates of both |
| `fidelity/` | teacher-forced dumps of oQ8e (the reference), oQ6e and int4 |
| `slot/` | six arms of base and shipping flags in mirrored order to 32K, the per-chunk prefill rates, the driver log |
| `display/` | display on and asleep, session waiting and idle, to 32K, the driver log |
| `gpu-time/` | five-second samples of GPU time per process over the display, quiet and snapshot windows; the share per window and per depth of the quiet ladder |
| `quiet/` | the published context sweep through 128K with the display asleep and the session idle, its warm sweep, the driver log |
| `snapshot-eval/` | patch 039: per arm the sweep, the prefill census and its summary, the greedy hashes; `report.txt` and the driver log; `display-on/` the first pass; `gates/` the follow-up fidelity dumps of both trees and the capability gates on the patched tree |
