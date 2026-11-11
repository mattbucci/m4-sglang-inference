# Concurrent decode on mRoPE hybrids: rows per forward, per-request position delta (patch 034), batched decode (patch 035)

Run dates 2026-11-10 and 2026-11-11. Stack: SGLang v0.5.23 + patches 001-035 (034 and 035
come out of this run), MLX 0.32.4, mlx-lm 0.32.3.dev17, mlx-vlm 0.6.5, M4 Pro
64 GB. Models: `mlx-community/Qwen3.6-35B-A3B-4bit` (preset `qwen36`, MoE,
40 layers), `mlx-community/Qwen3.6-27B-4bit` (preset `qwen36-27b`, dense,
64 layers), `mlx-community/Qwen3.5-27B-4bit` (preset `qwen35`, dense,
the one mRoPE preset that ships four request slots) and
`mlx-community/Qwen3.5-9B-MLX-8bit` (preset `qwen35-9b-8bit`, for the
greedy-text attribution).

The single-stream lane is in `../micro/VERDICT.md`. This folder is the
second optimization target: several streams at once, without moving the
single-stream number.

## Method

The method of `../micro/VERDICT.md` (census, raw kernel bench before
serving work, same-tree serving A/B in both arm orders, gates, a numbered
patch when retained, a disposition per experiment), plus:

1. A kill criterion is written before a kernel is built. The multi-row
   kernel had to run 8 rows in under 0.75 of the stock time for the same
   sweep. It did not, and it is not in the tree.
2. Concurrent throughput is measured in the window in which every stream
   decodes (`scripts/bench/concurrent_decode_probe.py`): tokens of all
   streams between the last stream's first chunk and the first stream's
   last chunk, divided by that window. A stream's own first-to-last rate
   mixes in the stretch where it ran with fewer neighbours. Streams have
   distinct prompts of about 220 tokens (no shared radix prefix), 200
   generated tokens, greedy, `ignore_eos`.
3. Text is not compared between arms. Two correct decode paths round
   differently in bf16 and greedy prose flips. Fidelity is teacher-forced
   on the real weights, against the spread between two stock paths, with
   a wrong-position control that shows what a real defect costs.
4. A flag arm counts only if its server log shows the hook's line (`MLX
   batched mRoPE decode enabled`). Every arm here has it and zero
   tracebacks (`batched/*.server-lines.txt`).

## Dispositions

| experiment | family | measured | disposition |
|---|---|---|---|
| rows per forward on the stock 4-bit matmul | weight reads | 2 rows cost what 1 row costs (9.3 ms for 2.05 GB of weights), then about 3.5 ms per extra row to 12 rows, then a 32-row tile kernel at 40 ms flat from about 15 rows | the second row is free: batching two requests is the cheapest throughput on this machine |
| multi-row 4-bit kernel (unpack each weight once for all rows) | weight reads | best layout 40.07 against 34.40 ms at 8 rows (×1.16), ×1.19 at 4 rows, ×0.93 at 3 rows; in a model step 49.35 against 40.67 ms at 8 rows (qwen36), 346.80 against 189.08 ms (dense) | rejected: fails the 0.75 kill criterion at every layout tried |
| pad a batch to 16 rows for the tile kernel | weight reads | 12 rows cost 44 to 54 ms, 16 rows 40 ms | not taken: no preset runs more than 8 slots, and the tile kernel's error against float32 is 1.5 times the per-row kernel's |
| per-request position delta (patch 034) | correctness | two live requests on an mlx-vlm Qwen checkpoint decode under whichever delta the last prefill left: text stream differs from its solo run at character 264 of 1,503, image stream at 133 of 1,304; with the patch every interleaved response equals its solo run | retained, always on |
| batched mRoPE decode (patch 035) | weight reads | qwen36: 80.6 / 97.6 / 106.4 / 75.6 tok/s aggregate at 1 / 2 / 4 / 8 streams serial, 80.4 / 117.4 / 152.4 / 189.5 batched; qwen36-27b 14.0 / 14.4 / 14.9 / 14.5 against 14.1 / 25.6 / 36.0 / 37.4; qwen35 14.0 / 14.4 / 14.8 against 14.1 / 25.6 / 35.9 at 1 / 2 / 4; fidelity equal to the spread between two stock paths on all three checkpoints; capability gates under four probes at once pass on qwen36 and qwen36-27b and repeat qwen35's single-stream results | retained: flag `SGLANG_MLX_BATCH_MROPE_DECODE=1`, on by default for `qwen35`, opt-in elsewhere |
| greedy text under the batched path | determinism | a row gets the same bits at any index, batch size and neighbours, and a batch of one is bit-equal to the model's own forward; one row against several differs by KL 0.0006 (dense) to 0.0032 (qwen36) because MLX's matmul takes another kernel for one row. qwen35-9b-8bit's greedy thinking check stops at 1,196 tokens alone and at 1,682 when it joins a running batch; of 13 runs that were joined part-way, 6 repeat the solo text, 1 stops at 1,651 and 6 run to the 2,048-token cap | a property of the flag, no change to the patch: a greedy request's text depends on which of its steps had company. The serial path (flag off) repeats its solo run bit for bit |
| decode a lone request as two rows, so that greedy text never depends on traffic | determinism | a second row costs 4.6 ms a token on qwen36 (12.4 to 17.0) and 7.2 ms on the dense 27B (70.8 to 78.0) | not taken: single-stream latency comes first, and the flag off already gives the property |
| draft tokens verified in a k-row forward | weight reads | dense 27B at 8,192 tokens: a 2-row forward 77.0 ms against a 70.3 ms step, 4 rows 108.0 ms; qwen36: 18.5 ms against 12.1 ms | open candidate on the dense checkpoints only (a one-token draft pays above 10 % acceptance there, above 54 % on the MoE); not built |

## Rows per forward

`scripts/bench/mlx_qmm_rows_bench.py`: 44 matrices of 17,408 × 5,120 at
4 bits (the dense 27B MLP up-projection, 2.05 GB of packed weights per
sweep, larger than any cache), bf16 activations, ms per sweep of all 44.

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 15 | 16 | 24 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| up, 17,408 × 5,120 | 9.33 | 9.29 | 10.66 | 14.10 | 16.62 | 22.08 | 28.47 | 43.98 | 38.61 | 39.82 | 40.39 | 39.68 | 78.67 |
| down, 5,120 × 17,408 | 9.28 | 9.17 | 11.96 | 15.47 | 21.62 | 24.61 | 29.76 | 43.96 | 40.13 | 39.79 | 41.36 | 40.81 | 78.51 |

One row reads the weights at 220 GB/s, which is the memory bus. The second
row rides on the same read. From the third row the per-row kernel is
launched again per row and each launch unpacks the weights again, so cost
grows by about 3.5 ms per row. From about 15 rows MLX switches to a kernel
that works on 32-row tiles: 40 ms flat to 32 rows, 79 ms at 64. Its
outputs equal the per-row kernel's on about 60 % of entries, and its mean
error against a float32 product is 2.14e-3 against 1.41e-3.

With float32 scales under bf16 activations (`qmm-rows-up-f32scales`) the
free second row is gone: 10.81 / 15.49 / 23.21 / 32.27 ms at 1 / 2 / 3 /
4 rows, 101.83 at 12. The matching dtype is what the two-row path needs.

## A k-row forward on a cached prefix

`scripts/bench/mlx_verify_rows_probe.py`: one request, a cached prefix,
then forwards of k tokens at once (what a draft-and-verify decoder would
run), ms per forward. The last column is the ordinary pipelined one-token
step at that depth.

| model | prefix | k=1 | k=2 | k=4 | k=8 | k=16 | one-token step |
|---|---|---|---|---|---|---|---|
| qwen36 | 24 | 13.99 | 16.74 | 23.27 | 36.15 | 58.39 | 11.09 |
| qwen36 | 8,192 | 14.95 | 18.54 | 26.46 | 50.33 | 74.23 | 12.06 |
| qwen36, patches 030 to 032 | 8,192 | 13.90 | 17.87 | 25.52 | 51.09 | 75.01 | 11.12 |
| qwen36-27b | 8,192 | 73.95 | 77.04 | 108.03 | 221.15 | 325.24 | 70.25 |

On the dense checkpoint a 2-token forward costs 10 % more than a step, so
a free one-token draft (a prompt-lookup match, for example) pays whenever
it is accepted more than one time in ten, and a fully accepted 4-token
forward is 2.6 steps' worth of tokens for 1.54 steps of time. On the MoE
the step is short and launch-bound, a 2-token forward costs 54 % more, and
a draft needs better than even odds. Open candidate for `qwen36-27b` and
`qwen35`; nothing is built.

## Multi-row kernel (rejected)

The R9700 team's W8A16 GEMM for 2 to 32 rows, carried to Metal
(`scripts/bench/mlx_qmm_rows_kernel_probe.py`): a custom kernel that
unpacks each 4-bit weight once and accumulates every row from it, the
stock one-row layout with a row loop inside. Kill criterion: 8 rows under
0.75 of stock. ms per sweep of the up shape, best layout (4 output columns
per simdgroup, 8 input values per thread):

| rows | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| stock | 9.37 | 10.59 | 13.88 | 26.25 | 34.40 | 53.45 | 39.86 | 41.44 | 40.90 |
| kernel | 9.41 | 9.85 | 16.45 | 31.96 | 40.07 | 56.04 | 68.91 | 98.27 | 126.61 |
| ratio | 1.00 | 0.93 | 1.19 | 1.22 | 1.16 | 1.05 | 1.73 | 2.37 | 3.10 |

Other layouts at 8 rows: 4 columns × 16 values ×1.92, 2 × 16 ×1.50,
8 × 8 ×2.67, 8 × 16 ×4.42, 4 × 32 ×4.41; the down shape ×1.74. The kernel
is exact (equal to stock on 100 % of outputs to 12 rows) and it reads the
weights once, but the row loop inside a thread costs more than the second
and third weight reads it saves: the stock path's launches run at bus
speed, and this one's throughput falls to 51 GB/s at 8 rows.

In a model step, with the kernel on every quantized linear for 3 to 14
rows: qwen36 with patches 030 to 032 30.49 / 49.35 ms at 4 / 8 rows
against 24.45 / 40.67 stock; dense 27B 199.83 / 346.80 against 99.85 /
189.08. Rejected and removed. The only win, 7 % at exactly 3 rows, does
not carry a kernel.

## Per-request position delta (patch 034)

mlx-vlm's Qwen language models number a token that arrives on a cache with
history as cache offset plus `_rope_deltas`. That field lives on the
model, and every prefill overwrites it: zero after a text prompt, negative
after an image (an image's tokens share positions, so a 784-token image
block leaves −756). With one request at a time the field is always the
live request's. With two, the request that prefilled first decodes under
the other's delta, and each such token is rotated at the wrong position in
all ten attention layers and written to its KV cache that way.

Patch 034 keeps the delta per request after each prompt forward and puts
it back before a continuation chunk and before each request's serial
decode forward. One live request never reassigns the field.

Served check (`scripts/eval/probe_concurrent_positions.py`, qwen36,
`CTX=8192 MAX_RUNNING=4`, radix cache off, 300 tokens, an 896 px image of
a red circle; each request is run alone twice, then interleaved with the
other in both arrival orders):

| tree | run | request | equal prefix with the solo run | verdict |
|---|---|---|---|---|
| without 034 | text first | text | 264 of 1,503 characters | FAIL |
| without 034 | image first | image | 133 of 1,304 | FAIL |
| with 034, serial | both orders | both | whole response, 1,544 and 1,362 | PASS |
| live tree with 034 and 035, flag off | both orders | both | whole response | PASS |

Teacher-forced, a swapped delta costs KL 0.010 to 0.034 per request
against 0.003 between two stock paths (the control rows in the fidelity
table below). The defect needs two live requests on an mlx-vlm Qwen
checkpoint. Presets with one slot (`qwen36`, `qwen36-27b`,
`qwen35-9b-8bit`) never reach it; `qwen35` ships four.

The same probe on `qwen35` at its shipped four slots, tree at patch 033
(`landed/pos35-pre034.*`): a text response that arrives first leaves its
solo run at character 159 of 1,562, an image response that arrives first
at 137 of 1,304. The request that prefilled second equals its solo run
both times, which is the signature of the shared field. With patch 034
both orders equal the solo runs (1,553 and 1,305 characters).

## Batched mRoPE decode (patch 035)

The runner batches a decode step across requests only when the attention
module has a `rope` it can call with an offset per row. mlx-vlm's Qwen
attention rotates from explicit position ids instead, so those models fell
back to one forward per request per step, and every forward read all the
weights again. Patch 035 gives the attention wrapper the per-request
position (cache offset plus the delta kept by patch 034), rotates all rows
at once, and runs one SDPA per request on the cache buffer that request
already owns. The DeltaNet layers, the MoE and the MLPs run once for all
rows. A step with a single request keeps the model's own forward.

### Raw step

`rows/batch-*`: the runner's batched path driven directly, ms per step.

| rows | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| qwen36, depth 1,024 | 14.20 | 17.77 | 26.21 | 41.63 | 69.65 |
| qwen36, patches 030 to 032 | 12.98 | 16.51 | 24.45 | 40.67 | 68.06 |
| qwen36-27b, depth 128 | 70.81 | 74.03 | 99.85 | 189.08 | 304.44 |

### Served A/B

`CTX=8192 MAX_RUNNING=8 MEM_FRAC=0.45`, one tree, flag off against flag
on. qwen36: 200 tokens, 2 repeats per level, two passes in mirrored order
pooled. Fused arms (patches 030 to 032 on) and the dense 27B (64 tokens,
1 repeat): one pass on this window. Aggregate tok/s, with the time per
token a stream sees in ms:

| arm | 1 stream | 2 | 4 | 8 | wired GB |
|---|---|---|---|---|---|
| qwen36 serial | 80.6 (12.4) | 97.6 (20.5) | 106.4 (37.6) | 75.6 (105.8) | 30.1 |
| qwen36 batched | 80.4 (12.4) | 117.4 (17.0) | 152.4 (26.3) | 189.5 (42.2) | 30.6 |
| gain | −0.2 % | +20.3 % | +43.2 % | +150.6 % | |
| qwen36 fused serial | 96.6 (10.4) | 116.7 (17.1) | 119.8 (33.4) | 87.4 (91.5) | 30.0 |
| qwen36 fused batched | 96.6 (10.4) | 137.5 (14.6) | 174.9 (22.9) | 194.6 (41.1) | 30.4 |
| gain | +0.1 % | +17.8 % | +45.9 % | +122.6 % | |
| qwen36-27b serial | 14.0 (71.7) | 14.4 (139.4) | 14.9 (269.5) | 14.5 (552.1) | 31.8 |
| qwen36-27b batched | 14.1 (70.8) | 25.6 (78.0) | 36.0 (111.0) | 37.4 (213.9) | 33.3 |
| gain | +1.3 % | +78.7 % | +142.6 % | +158.0 % | |

The single-stream column is inside the 1 % noise floor in every pair. The
served batched time per token equals the raw batched step (17.0 / 26.3 /
42.2 against 17.77 / 26.21 / 41.63 ms), so the server adds nothing on top
of the forward. Time to first token is unchanged (0.36 / 0.55 / 1.2 /
2.6 s at 1 / 2 / 4 / 8 streams in both arms: prefills still run one
request at a time). On the dense checkpoint a second stream is nearly
free (78.0 against 70.8 ms per token) because its step is one long weight
read; the MoE's step is launch-bound and each extra row adds router and
expert work. The serial path's 8-stream column falls below its 4-stream
column on qwen36; the batched path keeps rising.

Longer prompts, on the live tree at `MAX_RUNNING=4` (`landed/q36-*-6k*`):
5,343-token prompts, 200 tokens, 2 repeats per level, arm order serial,
batched, batched, serial, pooled. Time to first token is for 1 / 2 / 4
streams:

| arm | 1 stream | 2 | 4 | time to first token, s | wired GB |
|---|---|---|---|---|---|
| qwen36 serial | 75.7 (13.2) | 91.2 (21.9) | 101.0 (39.6) | 6.9 / 12.3 / 17.8 | 29.0 |
| qwen36 batched | 75.7 (13.2) | 109.2 (18.3) | 142.1 (28.2) | 6.7 / 11.6 / 17.4 | 29.4 |
| gain | −0.1 % | +19.7 % | +40.7 % | | |

The gain is the one the 220-token prompts show (+20.3 / +43.2 %): it comes
from the weight reads of a step, not from the attention over the prompt.

Passes are in `batched/pooled.txt`. The first pass (order batched then
serial) used each stream's own first-to-last rate and reads 3 to 6 % lower
at 2 and 4 streams in both arms; its gains are +18.5 / +40.7 / +148 % at
2 / 4 / 8 streams on qwen36.

### Fidelity

`scripts/eval/batched_decode_fidelity.py`, real qwen36 weights, no server:
four requests of 311 / 1,527 / 1,530 / 2,617 prompt tokens, the second
with a 784-token image block (delta −756), 96 teacher-forced steps each.
Three paths over the same tokens: one prefill forward, serial one-token
steps, and batched steps of four rows. The control decodes each request
serially under another request's delta.

| pair | KL mean | KL max | top-1 same | with patches 030 to 032: KL mean | top-1 same |
|---|---|---|---|---|---|
| serial steps against one prefill forward (two stock paths) | 0.00307 | 0.108 | 376 / 384 | 0.00364 | 379 / 384 |
| batched steps against one prefill forward | 0.00280 | 0.058 | 373 / 384 | 0.00378 | 379 / 384 |
| batched steps against serial steps (the gate) | 0.00304 | 0.092 | 372 / 384 | 0.00324 | 380 / 384 |
| control: serial steps under the wrong delta, per request | 0.0269 / 0.0132 / 0.0342 / 0.0101 | 0.538 | | 0.0278 / 0.0135 / 0.0344 / 0.0135 | |

The batched path sits as far from the serial path as the serial path sits
from a prefill forward: stepping noise. A wrong position is 3 to 11 times
that, so the gate has the resolution to see the defect class it guards.

### Served positions and gates under load

With the flag on, an interleaved response no longer equals its solo run to
the character (a batched step and a single step round differently, and
which steps are batched depends on arrival times), so the served check
grades content: every image response describes the red circle, every text
response stays on its prompt (`positions/pos-035`, and the same arm on the
live tree). PASS. The size and the source of that difference are in
"Greedy text under the batched path" below.

Capability gates on qwen36 with the flag on, `MAX_RUNNING=8`, and the four
probes started at the same moment so they share decode steps
(`gates-load/`): validator 3/3, thinking VERIFIED, vision STRONG, codegen
STRONG, every exit code 0, zero tracebacks. The scheduler's log lines show
27 decode batches of two requests, 2 of three and 4 of four.

## The dense checkpoints: qwen35 and qwen36-27b

`landed/`: the live tree with both patches applied, `CTX=8192 MAX_RUNNING=4
MEM_FRAC=0.45`. `qwen35` is the one mRoPE preset that ships four slots, so
it is the preset where patch 034's defect was reachable as shipped and
where patch 035 pays without a launch override.

Served A/B on qwen35 (64 tokens, 1 repeat per level, arm order serial,
batched, batched, serial, pooled), aggregate tok/s with the time per token
a stream sees in ms:

| arm | 1 stream | 2 | 4 | wired GB |
|---|---|---|---|---|
| serial | 14.03 (71.3) | 14.41 (138.8) | 14.80 (270.4) | 30.9 |
| batched | 14.11 (70.9) | 25.64 (78.0) | 35.91 (111.4) | 31.6 |
| gain | +0.6 % | +77.9 % | +142.7 % | |

Teacher-forced fidelity on the real weights of both dense checkpoints
(same four requests and 96 steps as above), KL mean with top-1 agreement:

| checkpoint | serial steps against one prefill forward | batched against prefill | batched against serial (the gate) | control, per request |
|---|---|---|---|---|
| qwen35 | 0.00089 (381 / 384) | 0.00161 (380 / 384) | 0.00114 (381 / 384) | 0.0197 / 0.1526 / 0.0314 / 0.0101 |
| qwen36-27b | 0.00085 (380 / 384) | 0.00084 (380 / 384) | 0.00062 (382 / 384) | 0.0162 / 0.0401 / 0.0291 / 0.2948 |

On qwen35 the image request carries most of the gate's mean: 0.0031
batched against serial with one step at 0.195, where the serial path
itself sits 0.0017 from the prefill forward with one step at 0.097. Top-1
agrees on all 96 of its steps in every pair, and its wrong-position
control is 0.153. The three text requests read 0.0003 to 0.0006.

Served positions on qwen35: with the flag off both interleaved responses
equal their solo runs to the character (1,553 and 1,305 characters); with
the flag on the content check passes.

Gates on qwen35 with the flag on and the four probes started together:
thinking VERIFIED, vision STRONG, codegen STRONG, validator 2/3. The
validator's miss is its greedy thinking check running to the 2,048-token
cap, which this checkpoint also does single-stream on the serial path
(`../hybrid-presets/VERDICT.md`, `benchmarks/qwen35-27b-4bit/remeasure-validator.txt`);
the batched path does not change the preset's gate results. 20 decode
batches of two requests, 4 of three, 3 of four in the scheduler's log
lines, zero tracebacks.

Gates on qwen36-27b with the flag on, `MAX_RUNNING=4` and the four probes
started together (`landed/gates27-load.*`): validator 3/3 (its thinking
check stops at 1,400 tokens, 1,282 alone), thinking VERIFIED, vision
STRONG, codegen STRONG, every exit code 0, zero tracebacks; 16 decode
batches of two requests, 4 of three, 3 of four.

Disposition: `scripts/launch.sh qwen35` sets
`SGLANG_MLX_BATCH_MROPE_DECODE=1` unless the environment says otherwise.
The one-slot presets leave the flag off; it engages there only when
`MAX_RUNNING` is raised. Checked on the launch script with no override
(`landed/q35-default.*`, `landed/q35-off.*`): the hook's line is in the
server log and 1 / 2 streams read 14.13 / 25.51 tok/s; with
`SGLANG_MLX_BATCH_MROPE_DECODE=0` the line is absent and they read
13.97 / 14.26.

## Greedy text under the batched path

With the flag on and four gate probes at once, qwen35-9b-8bit's validator
misses its greedy thinking check: 2,048 tokens, `finish=length`, where the
same check alone stops at 1,196 tokens (`landed/gates9b-load.*`; the other
probes read thinking VERIFIED, vision STRONG, codegen STRONG, zero
tracebacks). This section attributes that miss.

### A row does not depend on its index or its neighbours

`scripts/eval/batched_decode_row_symmetry.py`, real weights, no server: one
request A is teacher-forced through the same 64 tokens in eight
arrangements and its logits are compared bit for bit (`landed/rowsym-*`).
Steps with identical bits out of 64:

| pair | qwen35-9b-8bit | qwen35 | qwen36 |
|---|---|---|---|
| the model's own forward, run twice | 64 | 64 | 64 |
| patch 035's layer walk with a batch of one, against the model's own forward | 64 | 64 | 64 |
| [A,B], [B,A], [C,A], [A,B,C,D], [B,C,D,A], [B,A,C,D], every pair | 64 | 64 | 64 |
| any of those six against the model's own forward | 0 (KL mean 0.00062, top-1 61 / 64) | 0 (0.00061, 64 / 64) | 0 (0.00319, 60 / 64) |

The batched walk is the model's own function to the bit when it has one
row. A row among several gets the same bits whatever its position, the
batch size and the other requests. The whole difference is one row against
several, and it is MLX's: `mx.quantized_matmul` at 4 and 8 bits and the
dense matmul each return other bits for one row than for the same row
among two, and identical bits among two, four and eight, on bf16 and
float32 inputs. At the logits it is the stepping noise of the fidelity
tables above.

### The thinking check by arrival order

`scripts/eval/probe_thinking_under_load.py` on qwen35-9b-8bit,
`MAX_RUNNING=4`: the validator's reasoning request (greedy, 2,048-token
cap) alone, joining streams that are already decoding, and alone for some
seconds before a stream joins it. Each run returns the log-probability of
every token it chose; the last two columns compare those with the solo run
over the tokens both runs share.

| path | run | tokens | finish | tokens shared with solo | log-prob gap, mean | max |
|---|---|---|---|---|---|---|
| serial (flag off) | alone twice, joining 1, joining 3, joined by 3 | 1,196 each | stop | 1,196 | 0 | 0 |
| serial, patches 031 and 032 on | alone, against the plain serial run | 1,196 | stop | 1,196 | 0.0045 | 0.112 |
| batched | alone, against the serial run | 1,196 | stop | 1,196 | 0 | 0 |
| batched | joining 1 or 3 running streams (6 runs, 3 servers) | 1,682 each | stop | 648 | 0.0036 | 0.118 |
| batched | joined by one stream after 2, 5, 6, 8 s | 1,196 each | stop | 1,196 | 0.0041 to 0.0042 | 0.100 to 0.117 |
| batched | joined by three after 4 s (3 runs) | 1,196, 1,196, 1,651 | stop | 1,196, 1,196, 842 | 0.0045 (the 1,651 run) | 0.102 |
| batched | joined by one after 0.5, 3, 12 s | 2,048 each | length | 384 | 0.0009 to 0.0020 | 0.052 to 0.063 |
| batched | joined by one after 1, 10 s | 2,048 each | length | 519 | 0.0019 to 0.0024 | 0.068 to 0.102 |
| batched | joined by one after 15 s | 2,048 | length | 861 | 0.0039 | 0.079 |

The serial path repeats its solo run bit for bit under load. On the
batched path a request alone is bit-equal to the serial path. A request
that joins a running batch takes every step on the several-row kernels and
always produces the same 1,682-token run, with the right answer. A request
that is joined part-way gets a mix that depends on the join time. Every
row sits the same distance from the solo run, and that distance is the one
patches 031 and 032 put between two serial runs.

The steps where runs part ways are near-ties in the solo run. At token 384
it picks `.` at p 0.478 where the other runs pick ` (` at 0.507; at 519
` including` at 0.2492 against ` just` at 0.2482; at 648, 842 and 861 the
two candidates sit between 0.37 and 0.45 each. The solo run has 89 steps
with the chosen token below 0.52. After a flip the deliberation takes
another course: of the five courses seen, two reach the answer (1,651 and
1,682 tokens) and three are still deliberating at the cap. The validator's
miss under four probes is one of these.

On qwen35 the check runs to the cap alone, which is that preset's standing
validator miss, and under load (`landed/lp35-on.*`): joined runs share 427
and 514 tokens with the solo run, at a log-prob gap of 0.0015 mean and
0.067 to 0.117 max.

Sampled at the checkpoint family's recommended settings (temperature 0.6,
top-p 0.95, top-k 20; 12 draws per row on the batched server,
`landed/samp9b-on.*`):

| run | stop with the answer | at the 2,048-token cap |
|---|---|---|
| alone | 2 of 12 (1,053 and 1,544 tokens) | 10 of 12 |
| joining a running stream | 2 of 12 (1,264 and 1,683 tokens) | 10 of 12 |

The greedy solo pass is one favourable path through a response that does
not finish inside 2,048 tokens in five draws out of six on this checkpoint.
The batched path does not move that rate at this sample size; twelve draws
a row would show a large shift and cannot resolve a small one.

Disposition: no change to patch 035. The difference is the rounding of
MLX's one-row and several-row matmul, the size of the difference between
two stock paths, and a row is computed the same wherever it sits. What the
flag changes is a guarantee: on the serial path a greedy request repeats
its solo text whatever else is running, on the batched path its text
depends on which steps had company. The flag stays the `qwen35` default.
That preset's greedy thinking check misses alone as well, and a hybrid's
greedy text already moves by this much between a radix hit and a cold
run. On a preset whose greedy check passes alone, raising `MAX_RUNNING`
with the flag on can turn the check into a miss, as it does here on
qwen35-9b-8bit; `SGLANG_MLX_BATCH_MROPE_DECODE=0` keeps solo text under
load at the serial path's throughput.

## Receipts

| directory | contents |
|---|---|
| `rows/` | raw MLX: rows-per-forward sweeps of the stock matmul (`qmm-*`), k-row forwards on a cached prefix (`verify-*`), the multi-row kernel at seven layouts (`kernel-*`), the runner's batched step with and without that kernel (`batch-*`) |
| `batched/` | served A/B arms of patch 035, three passes, with each arm's hook lines and traceback count; `pooled.txt` |
| `fidelity/` | teacher-forced batched fidelity on qwen36, stock and with patches 030 to 032 |
| `positions/` | served interleaving check: the tree without patch 034, with it, and with the batched flag |
| `gates-load/` | capability, thinking, vision and codegen probes run at once against the batched path |
| `landed/` | the same checks on the live tree after both patches landed; the qwen35 arms (`q35-*`, `fid-*`, `pos35-*`, `gates35-load`); the 5,343-token-prompt A/B (`q36-*-6k*`); gates under load on qwen36-27b and qwen35-9b-8bit; the row-symmetry probe (`rowsym-*`); the thinking check by arrival order (`think9b-*`, `lp9b-*`, `lp35-on`, `samp9b-on`) |
