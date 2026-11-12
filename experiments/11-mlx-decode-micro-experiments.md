# M4-P: MLX decode micro-experiments (qwen36 first)


| | |
|---|---|
| **Type** | experiment lane |
| **Status** | measured; the three launch-folding patches are on in the `qwen36` preset, together with five decode parts (patch 038, experiment 12). The buffer limit and the block ladder are retained behind environment variables, patches 033 and 034 are always on, patch 035 (one forward per decode step of several requests on the mRoPE hybrids) is a flag the `qwen35` preset sets by default, further launch folding on qwen36 is closed. Defaults for the other presets wait on the cross-preset arms (`benchmarks/perf/2026-11-10/micro/VERDICT.md`, `benchmarks/perf/2026-11-10/concurrent/VERDICT.md`) |
| **Execution host** | m4-box |
| **Wall clock** | about two days of exclusive serving windows (a cold 251K arm is 24 to 26 minutes of prefill; the mirrored short-context A/B is 6 minutes per arm) |
| **GPU time** | none separate from the box (Apple unified memory); raw probes and serving arms never overlap |
| **Depends on** | Patch 028 (the overlap decode loop on hybrid presets); the 256K long-context recipe in `docs/long-context.md`; the R9700 decode lane as the method donor (`~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference`, `CROSS-TEAM.md` and its `benchmarks/perf/` verdict files). |
| **Provides to** | `scripts/common.sh` defaults; patches 029 to 035; the `qwen35` preset's multi-stream default; the decode curve in `docs/long-context.md`; a cross-team note on which parts of the method survive a change of GPU vendor. |

## Objective

Raise single-user decode speed on the M4 at every depth from 1K to 256K by the R9700 team's method: attribute each millisecond of a decode step to a family, give each family an owner experiment, time the kernel or setting in raw MLX first, and keep only what wins a same-tree serving A/B and passes the quality gates. Their kernels are HIP and do not port; the method does.


## Hypothesis

A qwen36 decode step is not bound by one thing. Its weights set a floor (1,670 MB read per token, about 7.6 ms at the 220 GB/s the dense presets sustain), 2,779 small launches and the host sit on top of that at short context, and ten attention kernels reading the KV cache take over at depth. Each of the three has an independent lever: how launches are committed, how many launches there are, and how the attention kernel partitions a deep cache.


## Background & receipts

- Method donor: the R9700 decode lane keeps a census per decode step, a raw kernel microbench before serving work, serving A/Bs at short, medium and deep context in both arm orders, a numbered opt-in patch for every retained change, a reverted tree for every neutral one, and a disposition per experiment.
- Step census: `benchmarks/perf/2026-11-10/micro/serve-census-base.jsonl` and `serve-graph/qwen36-step-primitives.json`. Wall 12.90 ms per step, of which 10.31 ms is `async_eval` and 2.22 ms is host graph build overlapped with the previous step; 2,779 primitives per step.
- Weight reads: `raw/weight-read-census.jsonl` (`scripts/bench/mlx_weight_read_census.py`). Dense presets decode at 209 to 226 GB/s of weight reads and have no margin; MoE presets spend 4 to 5 ms per step on something other than weights.
- Launch prices: `raw/launch-microbench.md`. A small launch costs 0.5 to 2.5 µs of encode plus 1.5 to 2.7 µs of GPU; a sampled token tail costs 387 µs against 69 µs greedy.
- MLX reads `MLX_MAX_MB_PER_BUFFER`, `MLX_MAX_OPS_PER_BUFFER` and `MLX_METAL_FAST_SYNCH` once per process, and `MLX_SDPA_BLOCKS` on every attention kernel call. Stock limits on this GPU are 50 operations or 50 MB per command buffer, and the stock attention ladder stops at 1024 blocks.
- A single request on qwen36 decodes through the serial path (`_decode_with_native_cache`), the model's own forward, so a runner hook that only covers the batched path never runs for it. A step of two or more requests takes the batched path only with patch 035's flag set; without it each request gets its own forward.
- mlx-vlm's Qwen language models keep the mRoPE position delta of the last prompt in one field on the model (`_rope_deltas`: zero after text, negative after an image block). Every live request reads that field at decode.
- bf16 activations flip the greedy token on about 2 % of prose steps against a float32 forward of the same 4-bit weights, on the prefill path and the decode path alike. A text hash that differs between two arms is not evidence of anything; a hash that matches is.
- The community checkpoint keeps `mlp.gate` and `mlp.shared_expert_gate` at 8 bits, so mlx-vlm's own four-way router fold does not engage; patch 030 folds the router explicitly.


## Method

1. Census. Serve qwen36 with `SGLANG_MLX_DECODE_CENSUS=1` and run `scripts/bench/decode_ab.py` at 948 / 7,583 / 30,391 prompt tokens. Dump the step graph with `scripts/bench/mlx_decode_census.py`. Run `scripts/bench/mlx_weight_read_census.py <snapshot dir>` for the bytes a step reads.
2. Raw probes, never while a serving arm is timing: `mlx_decode_knob_sweep.sh` (process-level MLX settings), `mlx_decode_slack.py` (step time against an injected host delay), `mlx_launch_microbench.py` (per-launch prices), `mlx_decode_router_probe.py` (launch-folding arms on the real model), `mlx_decode_compile_probe.py`, `mlx_sdpa_kv_microbench.py` and `mlx_sdpa_layout_probe.py` (attention against a deep cache), `mlx_kv_int8_fidelity.py`.
3. Size attention changes from the per-layer shape: `SPLIT=1 mlx_sdpa_layout_probe.py <depths> out.jsonl 0,1536,2048,2560 262208`. Timing all ten layers in one `mx.eval` overstates the gain of larger blocks three to five times and puts the crossover near 139K keys instead of past 196,608.
4. Serving A/B for every candidate: same tree, one environment variable per arm, 3 runs per depth, 256 generated tokens, greedy with `ignore_eos`, each arm twice in mirrored order, pooled medians. Check the arm's server log for the line the hook prints; an arm whose hook never ran is a second baseline.
5. Deep arms at 121,606 and 251,474 prompt tokens: `CTX=<label+64> MEM_FRAC=0.5 CHUNKED=1024 EXTRA_ARGS=--disable-radix-cache` with `bash scripts/common/oom_guard.sh &`, one cold request per arm, wired pages sampled every 2 s. `scripts/bench/decode_step_trace.py` records per-step times when a request decodes across a threshold.
6. Gates before a default changes: `validate_capabilities.py`, the four content-aware probes, and depth recall at the deepest rung with the change engaged. For anything that is not bit-identical, grade the distribution and not the text: `scripts/eval/decode_path_precision.py` recomputes every step of a served greedy dump teacher-forced against a float32 forward of the same weights (`--fuse` installs the decode fusions first), and a change passes when its KL sits in the band the stock prefill and decode paths already span. `scripts/eval/decode_fidelity.py --restart-every 48` must be bit-identical to the uninterrupted decode, and `scripts/eval/probe_followup_fidelity.py` grades a chat-shaped follow-up served from a radix hit against the same follow-up prefilled cold.
7. Land a retained code change as a numbered opt-in patch with a CPU unit test under `scripts/eval/`, and run `scripts/test_patch_gates.sh`. Revert a neutral change. Record a disposition for every experiment in the run's `VERDICT.md`.
8. Carry a retained setting to the other presets with a one-depth A/B per preset before it becomes a default in `scripts/common.sh`.
9. Several streams. Raw first: `mlx_qmm_rows_bench.py` (cost of the 4-bit matmul against rows per forward), `mlx_verify_rows_probe.py` (a k-row forward on a cached prefix), `mlx_qmm_rows_kernel_probe.py` (a custom multi-row kernel). Served: `concurrent_decode_probe.py` streams 1, 2, 4 and 8 requests at once and reports aggregate tok/s in the window where every stream decodes, from the last stream's first chunk to the first stream's last chunk. Correctness: `scripts/eval/probe_concurrent_positions.py` interleaves a text and an image request and compares each response with its solo run, and `scripts/eval/batched_decode_fidelity.py` grades batched steps against serial steps teacher-forced on the real weights, next to a control with the wrong position delta.


## Baseline & instrument

Baseline is the same tree with no experiment variable set, measured in the same chain as the arm it is compared with. Instrument for decode speed is `decode_ab.py` (server-side step timing over 256 generated tokens, prompt cached after the first run). Instrument for memory is wired pages from `vm_stat` against the 51.8 GB wired limit. Run-to-run noise on the short-context A/B is about 1 %; a change inside that band is neutral.


## Success criteria

- A retained change wins at least 2 % at one depth, loses nowhere, and holds in both arm orders.
- Greedy output is bit-identical to baseline, or the change passes the fidelity and recall gates and says so in its patch note.
- Wired peak at 251K stays under 32 GB with the change engaged.
- Capability gates and probes pass with every retained change enabled together.
- 256K decode improves measurably, since that is the primary optimization target.


## Kill criteria

- A raw probe shows no gain: stop before any serving work and record the null.
- A serving arm is inside noise at all three depths: revert the tree.
- Any arm loses more than noise at any depth, or raises wired peak by more than 2 GB: reject.
- A change needs an edit to MLX, mlx-lm or mlx-vlm sources that cannot be carried as a local hook: do not take it, and never send it upstream.


## Deliverables

- `benchmarks/perf/2026-11-10/micro/VERDICT.md` with a disposition per experiment and a receipts map; `benchmarks/perf/2026-11-10/concurrent/VERDICT.md` for the multi-stream experiments.
- `scripts/bench/mlx_*.py`, `decode_ab.py`, `decode_step_trace.py`, `concurrent_decode_probe.py`.
- Patches 029 to 035 with unit tests and rows in `patches/README.md`.
- Defaults in `scripts/common.sh` for whatever passes the gates.
- The refreshed decode curve in `docs/long-context.md`.
- A cross-team entry in the R9700 repo's `CROSS-TEAM.md`.


## Constraints

- OOM guard for every arm at 32K and beyond; `MEM_FRAC` 0.5 for deep arms, never above 0.7.
- One serving process at a time, and no raw MLX timing while a serving arm decodes or prefills.
- Full-size model in every arm.
- No edits to files a running chain sources; change them at a pause point between steps.
- No upstream pull requests.


## Risks

- A raw probe can have the wrong shape. The joint attention benchmark predicted 11.7 ms at 251K where serving gained about 3; the per-layer probe predicts 4.0 to 4.6.
- A flag-gated hook can sit on a code path the preset never takes. The first ladder arm at 256K was a second baseline for that reason.
- Larger command buffers hold more memory in flight: 400 MB buffers add about 2 GB of wired peak at 122K, so the retained value is 200.
- More than about ten command buffers in flight stalls the queue: an operations limit of 200 loses 17 % at 30K.
- Single-turn gates never reach a radix prefix hit. Patch 033 fixes a position defect on that path that every capability gate passed over; a serving change is not validated until a follow-up turn has been graded against its cold twin.
- `decode_ab.py` builds every depth from one seeded sentence stream, so the first request at a deeper depth is a prefix hit on the previous depth's prompt. Its time to first token covers the uncached tail only; use a single-depth run for cold prefill timings.
- State that mlx-vlm keeps on the model object is shared by every live request. A gate that sends one request at a time never sees it; patch 034's defect needed a text and an image request interleaved and each response compared with its solo run.
- A per-stream rate measured over each stream's own lifetime mixes in the time a stream decodes alone. Aggregate throughput is only comparable across arms inside the window where every stream is decoding.


## Results

Receipts and the full disposition table: `benchmarks/perf/2026-11-10/micro/VERDICT.md`.

| lever | setting | served result on qwen36 |
|---|---|---|
| how launches are committed | `MLX_MAX_MB_PER_BUFFER=200` | +4.2 to +6.1 % decode to 30K, +1.4 % at 122K, cold prefill 5 to 7 % shorter at 251K, 1.1 to 1.7 GB more wired peak; on top of the preset's flags +1.8 / +2.6 / +3.6 / +4.1 / +0.1 % at 1K / 7.6K / 30K / 65K / 130K, opt-in |
| how many launches there are | `SGLANG_MLX_FUSE_MOE_ROUTER=1`, `SGLANG_MLX_FUSE_GDN_FRONT=1`, `SGLANG_MLX_FUSE_DECODE_LINEARS=1` (patches 030, 031, 032) | +11.6 / +11.0 / +10.1 % at 948 / 7,583 / 30,391 prompt tokens on top of the buffer limit: 95.9 / 90.0 / 75.1 tok/s against 85.9 / 81.1 / 68.2 |
| how attention partitions a deep cache | `SGLANG_MLX_SDPA_BLOCK_LADDER=1` (patch 029) | 37.3 against 40.8 ms per token at 251,474 prompt tokens, depth recall 6/6, stock block counts through 196,608 keys |

- The `qwen36` preset sets the three launch-folding flags and five decode parts (patch 038). On a quiet GPU, five settings in mirrored order: 97.6 / 91.6 / 76.4 tok/s cold at 959 / 7,601 / 30,408 prompt tokens against 85.2 / 80.5 / 69.0 with none of them, 61.8 against 56.9 at 65K and 45.4 against 42.8 at 130K, prefill, TTFT and wired memory equal (`benchmarks/perf/2026-11-10/flint/quiet-settings/`). The capability gates pass on that setting (`benchmarks/perf/2026-11-10/flint/gates/q36-fuse-sub5/`), depth recall is 6/6 at 245,656 prompt tokens with a 28.5 GB wired peak, and the regression baseline is armed on it: 97.4 / 80.8 / 65.0 tok/s at 1K / 8K / 32K against 86.7 / 70.6 / 58.7 for the stock flags in the same window (`benchmarks/perf/2026-11-10/flint/quiet-rearm/`).
- The three launch-folding patches pass the teacher-forced fidelity gate: KL to the float32 yardstick 0.0189 (030), 0.0206 (031), bit-identical (032), 0.0211 together, against 0.0165 for the stock decode path and 0.0194 for the stock prefill path.
- Patch 032 pays only next to the other two (+1.6 to +3.1 % over three mirrored pairs) and is neutral alone.
- With the three patches installed a raw step is 1,850 primitives and 10.15 ms against 7.9 to 8.2 ms of weight and state traffic at 220 GB/s. No module family holds more than 1 ms above its reads, the 80 layer norms knock out to zero, and the head sits on its read. Further launch folding on this checkpoint is closed; the dense presets sit on the memory wall and have no such margin.
- Patch 033 is a correctness fix the fidelity instruments found: without it a radix prefix hit on an mlx-vlm Qwen checkpoint numbers its new tokens from zero. With it a scheduled re-prompt is bit-identical over 4,598 steps and a follow-up hit sits at KL 0.0064 from its cold twin (0.0658 without), the level two cold prefills on different chunk sizes show.
- Rejected after measurement: larger operation limits per command buffer, `MLX_METAL_FAST_SYNCH`, `MLX_BFS_MAX_WIDTH`, `mx.compile` around the blocks, constant gather indices, a fused GatedDeltaNet back kernel, int8 keys and values, a layer-norm fold.
- `SGLANG_MLX_CLEAR_CACHE_STEPS=0` is free to 122K, removes a 1.7 % stall pattern at 139K and 196K, and reads 35.8 against 36.8 ms per token at 251,474 prompt tokens with identical output. Retained for long-context work; with every retained setting the 256K step is 35.8 ms against 40.8.

Several streams at once, aggregate tok/s in the window where every stream decodes (`benchmarks/perf/2026-11-10/concurrent/VERDICT.md`):

| preset | streams | one forward per request | one forward per step (patch 035) |
|---|---|---|---|
| qwen36, `MAX_RUNNING=8` | 1 / 2 / 4 / 8 | 80.6 / 97.6 / 106.4 / 75.6 | 80.4 / 117.4 / 152.4 / 189.5 |
| qwen36-27b, `MAX_RUNNING=8` | 1 / 2 / 4 / 8 | 14.0 / 14.4 / 14.9 / 14.5 | 14.1 / 25.6 / 36.0 / 37.4 |
| qwen35, its four slots | 1 / 2 / 4 | 14.0 / 14.4 / 14.8 | 14.1 / 25.6 / 35.9 |

- The stock 4-bit matmul gives the second row for free (9.29 against 9.33 ms per sweep of the MLP weights), costs 1.5 times one row at four and 3.1 times at eight, and is flat from 16 to 32. A custom multi-row kernel ran 1.16 times the stock time at 8 rows against a kill criterion of 0.75 and was rejected.
- Patch 034 is a correctness fix the interleaved probe found: with two live requests the one that prefilled first decoded under the other's position delta. Served on qwen36 with four slots, a text response left its solo run at character 264 of 1,503 and an image response at 133 of 1,304; with the patch every interleaved response equals its solo run.
- Patch 035 passes the teacher-forced gate on all three checkpoints: batched steps against serial steps read KL 0.0030 (qwen36), 0.0011 (qwen35) and 0.0006 (qwen36-27b), the distance serial steps keep from a prefill forward, against 0.010 to 0.29 for the wrong-delta control. A single stream does not move, wired memory rises 0.4 to 1.5 GB.
- The batched path changes one guarantee. A batch of one is bit-equal to the model's own forward, and a row gets the same bits at any index, batch size and neighbours, but MLX's matmul rounds one row differently from several (KL 0.0006 on the dense checkpoints, 0.0032 on qwen36). A greedy request's text therefore depends on which of its steps had company, where the serial path repeats its solo run bit for bit. On qwen35-9b-8bit the validator's greedy thinking check, which sits on near-ties (89 steps with the chosen token below 0.52), stops at 1,196 tokens alone, at 1,682 when it joins a running batch, and runs to the 2,048-token cap in 6 of 13 runs that were joined part-way. Decoding a lone request as two rows would remove the dependence at 4.6 ms a token on qwen36 and 7.2 ms on the dense 27B, and is not taken.
