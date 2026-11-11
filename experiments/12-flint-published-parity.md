# M4-F: Flint against the published M4 Pro engines

| | |
|---|---|
| **Type** | experiment lane |
| **Status** | measured; patches 036 to 039 retained. Patches 036 and 039 are always on, patches 037 and 038 are flags the `flint` preset sets. `RecursiveSI/flint-int4` is the gate-validated fast checkpoint and the preset default stays oQ8e until an agentic cell is scored on int4. Open: a raw probe of the attention kernel at 32K to 128K, and the routing launches of a block, 0.8 ms a step that one kernel per block does not recover (`benchmarks/perf/2026-11-10/flint/VERDICT.md`) |
| **Execution host** | m4-box |
| **Wall clock** | about one day of exclusive serving windows (the like-for-like baseline is 45 minutes, a raw probe 5 to 25, a mirrored serving A/B 70 minutes for thirteen arms) |
| **GPU time** | none separate from the box (Apple unified memory); raw probes and serving arms never overlap |
| **Depends on** | The `flint` preset (patches 019 and 021, the pipelined decode loop under patch 028); experiment 11's method and its launch prices (`benchmarks/perf/2026-11-10/micro/VERDICT.md`); the published checkpoint `RecursiveSI/flint-int4`. |
| **Provides to** | The `flint` preset's default checkpoint and decode flags; patches 036 to 039; the Flint rows in `README.md` and `docs/flint.md`; a cross-team note with the reference numbers. |

## Objective

Close the gap between the `flint` preset and the two engines that publish Flint numbers for this chip. Both were measured on a MacBook Pro M4 Pro with the 20-core GPU, the same GPU as this box, on `RecursiveSI/flint-int4` (4.54 bits per weight, 22.6 GB). They are the reference engine of this lane in the way vllm-radiance is the R9700 team's: a number somebody else already reached on the same silicon, so a gap is a list of missing mechanisms and not a guess about what the hardware allows.

## Targets

Published context sweep at temperature 1.0, tokens per second:

| prompt tokens | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode, rec 0.1.0 | 126.5 | 123.0 | 116.5 | 107.0 | 95.2 | 79.4 | 56.5 | 35.2 |
| decode, Kapitsa | 124.3 | 121.3 | 119.2 | 113.2 | 106.2 | 85.9 | 66.2 | 45.7 |
| decode, llama.cpp | 77.5 | 75.1 | 73.4 | 69.7 | 63.7 | 53.8 | 40.5 | 23.8 |
| prefill, rec 0.1.0 | 781 | 800 | 791 | 760 | 699 | 581 | 427 | 262 |
| prefill, Kapitsa | 858 | 976 | 960 | 907 | 808 | 651 | 464 | 300 |
| prefill, llama.cpp | 927 | 1015 | 978 | 910 | 785 | 602 | 382 | 199 |

Published short runs (five runs of 512 generated tokens on a 1K prompt):

| | decode T=1.0 | decode greedy | prefill 4K | prefill 16K | follow-up TTFT 4K | follow-up TTFT 16K |
|---|--:|--:|--:|--:|--:|--:|
| Kapitsa | 127.90 | 131.60 | 968 | 817 | 0.13 s | 0.15 s |
| rec 0.1.0 | 126.56 | 132.37 | 796 | 699 | 0.41 s | 0.44 s |
| llama.cpp | 79.57 | 79.93 | 977 | 786 | 0.23 s | 0.31 s |

Sources: the Kapitsa repository's benchmark pull requests 286 and 290 (`github.com/nace-ai/kapitsa`), the rec repository (`github.com/recursive-oss/rec`), and Recursive's model card (`huggingface.co/RecursiveSI/flint`, 131.22 tok/s at 1K with 1,024 generated tokens). The reference machine has 48 GB; this box has 64 GB and the same GPU.

What the two engines say they do: rec runs a decode step as a few fused Metal kernels per layer; Kapitsa runs decode attention as one dispatch (head norms, rotary embedding, cache write and attention), the router with the shared expert's input projections as one dispatch, the routed experts as an input kernel and an output kernel, keeps the sliding layers in 528-row ring caches, checkpoints prefixes, and prefills in 512-token chunks. Both report that prompt-lookup drafts do not pay on Flint.

## Hypothesis

The published engines sit on the memory-read floor and this stack does not. A Flint decode step on the official 4-bit checkpoint reads 1,450.7 MB of weights (attention 520.3, routed experts 499.0, output head 173.0, shared expert 166.3, router 49.3, dense MLP 42.5), which is 6.1 to 8.2 ms at the 178 to 239 GB/s this GPU sustains; Kapitsa publishes 7.95 ms per token and rec 7.6. Everything this stack spends above that floor belongs to one of four mechanisms, each measurable alone:

- work per launch that the engines do not do: the router casts its BF16 weight to float32 on every step of every sparse layer, and the expert combine is a chain of five primitives;
- radix-cache bookkeeping in the decode loop: a chained step keeps a frozen view of every sliding-layer cache, and a held buffer turns the next token's write into a copy;
- host time the GPU waits for: a serving step builds a longer graph and runs the scheduler between submissions;
- launch count: mlx-lm runs the step as 2,747 graph primitives, about 56 per layer, where the engines describe four to six dispatches per layer.

## Background & receipts

- Weight reads: `benchmarks/perf/2026-11-10/flint/raw/weight-read-census-int4.json` (`scripts/bench/mlx_weight_read_census.py` on the snapshot headers). The community oQ8e checkpoint the preset defaults to reads 2,672.5 MB per token and oQ6e 2,465.0 MB, so the default checkpoint alone costs 84 % more traffic than the one the published numbers use.
- Model shape (`mlx_lm/models/flint.py`): hidden 2048, 49 layers (2 dense, 47 sparse), 32 heads over 4 KV heads, head dimension 128, 256 experts with 6 routed per token, expert width 512, shared-expert width 1024, a 513-token sliding window on 35 layers, vocabulary 150,144. The router weight is BF16 256 by 2048 and the stock forward casts it to float32 on every step of every sparse layer.
- The fused decode flags of patches 030 to 032 match Qwen module classes and mlx-vlm namespaces. None of them installs on Flint, and the pinned mlx-lm Flint file is identical to mlx-lm's main branch, so every mechanism in this lane is a local hook.
- A served Flint request decodes through the batched attention path (`_decode_with_batched_attention`): the model's own layer stack runs, and each layer's attention is the runner's wrapper, which projects q, k and v, applies head norms and the rotary embedding, writes the token into the request's layer cache, and calls MLX attention on the gathered cache.
- MLX's stock kernels fix a summation order that a replacement has to reproduce to stay bit-identical: `mx.fast.rms_norm` sums four consecutive values per thread and reduces through SIMD groups; the weighted sum over routed experts accumulates sequentially in float32; the rotary embedding derives its frequencies with `exp2` and the fast `cos` and `sin`; the 256 by 2048 router product is a 32-lane dot product per group of four rows. A kernel that sums in another order lands 1e-4 to 1e-3 KL away and flips near-ties.
- `mx.fast.metal_kernel` launches cost the same as stock ones, so a fold pays for the launches it removes and for whatever work the folded chain repeated, and nothing else.

## Method

1. Audit the checkpoint before it boots: `validate_chat_template.py --model <snapshot>`, `check_mlx_quant_scales.py RecursiveSI/flint-int4`, then `validate_capabilities.py` against the served preset.
2. Like-for-like baseline. Serve `MODEL=RecursiveSI/flint-int4 scripts/launch.sh flint` and run `scripts/bench/published_shape_sweep.py`: one cold request per depth from 1K to 32K with its own seed (no depth is a prefix of another), the same request again for the hit path, five runs of 512 and of 1,024 generated tokens on a 1K prompt at temperature 1.0 and greedy, and a chat-shaped follow-up at 4K and 16K. Repeat with the radix cache off and on the default oQ8e checkpoint.
3. Raw reference in the same window: `mlx_lm.generate` on the same checkpoint, so the serving overhead over plain mlx-lm is a measured number.
4. Step census: `scripts/bench/mlx_decode_census.py` knocks module families out of the raw step, `scripts/bench/mlx_flint_attention_census.py` does the same inside attention, and the serving census (`scripts/bench/serve_census/` on the server's `PYTHONPATH`, `SERVE_CENSUS_OUT=<jsonl>`) splits a served step into host and GPU phases.
5. Raw probes, never while a server is up: `scripts/bench/mlx_flint_decode_probe.py 1024,8192 out.jsonl` times each folding arm on the real model, in both arm orders, 64 teacher-forced steps for fidelity and three 96-step pipelined segments for time. `--selftest` checks every kernel against the stock chain at float32 and bf16 first. `scripts/bench/mlx_pipeline_slack_probe.py` adds a busy-wait before each graph build and submits the step in parts, which measures how much host time the pipelined loop hides.
6. A raw arm is carried to serving only when its gain clears noise. Land it as a numbered patch: a fused decode forward installed by class swap, a bitwise check per module at install time with the stock call kept wherever the check fails, a log line with the count of patched modules, and a CPU unit test.
7. Serving A/B: same tree, one environment variable per arm, the sweep of step 2, each arm twice in mirrored order. Check the server log for the hook's line.
8. Deep arm at 64K and 128K under the long-context recipe (`MEM_FRAC=0.5 CHUNKED=1024`, radix off, OOM guard), since the published curve runs to 128K.
9. Before the preset's default checkpoint changes: the capability gates, `probe_codegen.py`, and `quant_fidelity.py` against an oQ8e reference dump.

## Baseline & instrument

Baseline is the `flint` preset on `RecursiveSI/flint-int4` with no experiment variable set. Decode speed is tokens per second between the first and the last streamed token; prefill is prompt tokens over time to first token on a cold request; follow-up cost is the time to first token and the cached-token count of the second chat request. The raw probe reports milliseconds per pipelined step. Memory is wired pages from `vm_stat`. The published rows were measured by other people on a 48 GB machine with the same GPU; a row is compared only with the published row of the same shape.

## Success criteria

- Every published shape has a measured row on this stack and a gap in milliseconds per token.
- A retained fold is bit-identical to the stock forward (log-probabilities equal on 64 teacher-forced steps at two depths), or passes the fidelity gates and says so in its patch note.
- A retained fold wins at least 2 % at 1K in both arm orders and loses nowhere to 32K.
- The follow-up turn at 16K answers in under half a second.
- The capability gates pass with every retained change enabled.

## Kill criteria

- A raw arm inside noise: no serving work, record the null.
- A serving arm inside 1 % at every depth: revert the tree.
- A fold that cannot be made bit-identical and fails the KL band: reject.
- A mechanism that needs an edit to MLX or mlx-lm sources that a local hook cannot carry: do not take it, and never send it upstream.
- The gap that remains is the weight-read floor of this stack's kernels: stop and record the floor.

## Deliverables

- `benchmarks/perf/2026-11-10/flint/VERDICT.md` with the gap table, a disposition per arm and a receipts map.
- `scripts/bench/published_shape_sweep.py`, `scripts/bench/mlx_flint_decode_probe.py`, `scripts/bench/mlx_flint_attention_census.py`, `scripts/bench/mlx_pipeline_slack_probe.py`, `scripts/bench/prefill_census_report.py` and `scripts/bench/gpu_time.py`.
- Patches 036 to 039, each with a unit test and a row in `patches/README.md`.
- The Flint rows in `README.md` and `docs/flint.md`.
- A cross-team entry with the reference numbers and the method.

## Constraints

- One serving process at a time, and no raw MLX timing while a server is up.
- Timed arms run with the display asleep and the agent session idle behind one background driver per stage; `scripts/bench/gpu_time.py` records the GPU time of every process per arm.
- OOM guard for every arm at 32K and beyond; `MEM_FRAC` 0.5 for deep arms, never above 0.7.
- The full checkpoint in every arm; a tiny random Flint is for kernel exactness only.
- Third-party engine binaries are not run on this box; their numbers are quoted from their own receipts.
- No upstream pull requests.

## Risks

- The published runs come from a different machine (48 GB, a different macOS and thermal state). A few percent of the gap can be the machine.
- A kernel that is bit-identical on a random tiny model can still differ on the real one, because MLX picks its matmul kernel by shape. The install-time check runs on the real weights for that reason.
- The stock expert fusion in this tree rounds once where the stock chain rounds three times. It is not bit-identical and has to be graded by KL, not assumed.
- A folded weight is a second resident copy. The q, k and v fold costs about 125 MB and the shared-expert fold about 100 MB.
- A decode gain at 1K can vanish at depth, where the 14 full-attention layers read the whole cache and no fold helps.
- The sampled rows include a sampler tail that greedy rows do not; a comparison across modes mixes the two.

## Results

Receipts and the full disposition table: `benchmarks/perf/2026-11-10/flint/VERDICT.md`.

Served decode on `RecursiveSI/flint-int4`, 1K prompt, greedy, five runs of 512 tokens, the display on, against the published 131.6 (Kapitsa) and 132.4 (rec) tok/s:

| tree | ms per token | tok/s | of the published rate |
|---|--:|--:|--:|
| patches 001 to 035 | 13.35 | 74.9 | 57 % |
| patch 036 (windowed lookahead view) | 12.38 | 80.8 | 61 % |
| patches 036 and 037 (fused MoE block) | 10.99 | 91.0 | 69 % |
| patches 036 to 038 (decode submission in parts), the preset | 10.01 | 99.9 | 76 % |
| the preset with `MLX_MAX_MB_PER_BUFFER=200` | 9.75 | 102.6 | 78 % |

The published context sweep (one slot, radix off, `CTX=131136 MEM_FRAC=0.5`, 256-token reply at temperature 1.0, the display asleep):

| prompt tokens | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode tok/s | 102.5 | 101.7 | 98.4 | 92.8 | 83.3 | 71.4 | 55.3 | 37.5 |
| of rec | 81 % | 83 % | 84 % | 87 % | 88 % | 90 % | 98 % | 106 % |
| of Kapitsa | 82 % | 84 % | 83 % | 82 % | 78 % | 83 % | 84 % | 82 % |
| of llama.cpp | 132 % | 135 % | 134 % | 133 % | 131 % | 133 % | 137 % | 157 % |
| prefill tok/s | 1022 | 965 | 953 | 879 | 738 | 594 | 448 | 296 |
| of rec | 131 % | 121 % | 120 % | 116 % | 106 % | 102 % | 105 % | 113 % |
| of Kapitsa | 119 % | 99 % | 99 % | 97 % | 91 % | 91 % | 96 % | 99 % |
| of llama.cpp | 110 % | 95 % | 97 % | 97 % | 94 % | 99 % | 117 % | 149 % |

- Patch 036: a chained step held a frozen view of each of the 35 sliding caches, so the next token's write copied both buffers. The view now reads the window from the live buffer. +7.8 % greedy at 1K, +6.1 to +7.3 % to 32K.
- Patch 037: the router logits on the BF16 weight and the expert combine with the shared expert, each as one Metal kernel, bit-identical to the stock block. Raw 10.03 to 10.16 against 11.47 to 11.51 ms at 1K; served +12.7 % at 1K, +11.7 / +8.9 % at 8K / 32K over the view, +14.1 % on oQ8e.
- Patch 038: the pipelined loop left the GPU idle through the host work between submissions (a raw loop loses 2.0 ms of step time to a 2 ms host delay). Five submissions per step recover three quarters of it: +7.5 % greedy at 1K, +8.2 / +8.9 / +2.7 % cold decode at 1K / 8K / 32K, prefill and memory unchanged, the same greedy text. A larger command buffer hides the same slack and pays 5 to 8 % of cold prefill for it, so it stays an opt-in.
- The two flags are worth 19 to 21 % of decode through 8K, 13 % at 32K and 2.4 % at 128K: they remove a fixed cost per step, and the step grows from 9.8 to 26.7 ms. Prefill reads the same with and without them (three mirrored arms per setting).
- Checkpoints on the shipping preset: int4 96.2 / 88.8 / 67.0 tok/s at 1K / 7.6K / 30.4K, oQ6e 66.8 / 62.4 / 52.0, oQ8e 62.7 / 58.4 / 49.2. int4 and oQ8e pass the validator, the thinking probe and the codegen probe with the flags on. Teacher-forced NLL is equal on the three; oQ6e sits at KL 0.051 from oQ8e (93.7 % of argmax tokens equal) and int4 at 0.218 (83.0 %). The default is not decided on these numbers: it waits for an agentic cell on int4.
- The follow-up turn answers in 0.16 s at 3.8K and 0.30 s at 15K on int4 (Kapitsa 0.13 and 0.15 s, llama.cpp 0.23 and 0.31 s, rec 0.41 and 0.44 s at 4K and 16K).
- Served prefill is evaluation time: host time is 3 to 4 ms of a 1.1 to 1.8 s chunk. Patch 039: with the radix cache on a chunk stores a snapshot of the 35 tracked layers, and an evaluation that waits per layer idles the GPU about 1.1 ms each. Scheduling the evaluations takes 3.5 % off a cold prefill at 7.6K and 4.0 % at 30.4K (879 to 881 and 637 to 642 tok/s) and 3.4 to 5.0 % off a follow-up's cached request; one evaluation for all layers ties. Cold follow-ups of the two trees are equal at KL 0 and the gates pass. qwen36, with 30 waits on a 4,096-token chunk, gains under 1 % cold and 3 to 6 % on a repeat.
- Not carried: folds of the attention projections, head norms and rotary embedding (`qkv`, `front`, `mid`, inside noise), deeper launch folding past 2,183 primitives (equal step time), the tree's fused routed-expert kernel (slower, not bit-identical), one Metal kernel for the router logits and the routing of a block (bit-identical, 8 to 10 % slower per step than the two launches it replaces), nine submissions per step (not separable from five), larger operation limits per command buffer.
- What is left at 1K is 1.7 to 1.9 ms per step against both engines: the two routing launches per block (0.8 ms a step) and attention above its reads. Per token of context the step grows by 0.131 µs here, 0.159 for rec and 0.107 for Kapitsa, so the deep gap to Kapitsa is the read rate of the shared keys and values.
- The desktop is part of the instrument. A lit display costs 3.5 to 4.6 % of decode and 2 to 3 % of cold prefill, and an agent session blocked in a foreground wait up to 10 % of a deep prefill. Timed arms run with the display asleep and the session idle, and each arm carries its GPU share.
