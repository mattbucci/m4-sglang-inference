# M4-P: MLX decode micro-experiments (qwen36 first)


| | |
|---|---|
| **Type** | experiment lane |
| **Status** | measured; flags opt-in. The buffer limit, the block ladder and the three launch-folding patches are retained behind environment variables, patch 033 is always on, further launch folding on qwen36 is closed. Defaults in `scripts/common.sh` wait on the capability gates with the fusions on, the 256K arms and the cross-preset arms (`benchmarks/perf/2026-11-10/micro/VERDICT.md`) |
| **Execution host** | m4-box |
| **Wall clock** | about two days of exclusive serving windows (a cold 251K arm is 24 to 26 minutes of prefill; the mirrored short-context A/B is 6 minutes per arm) |
| **GPU time** | none separate from the box (Apple unified memory); raw probes and serving arms never overlap |
| **Depends on** | Patch 028 (the overlap decode loop on hybrid presets); the 256K long-context recipe in `docs/long-context.md`; the R9700 decode lane as the method donor (`~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference`, `CROSS-TEAM.md` and its `benchmarks/perf/` verdict files). |
| **Provides to** | `scripts/common.sh` defaults; patches 029 to 033; the decode curve in `docs/long-context.md`; a cross-team note on which parts of the method survive a change of GPU vendor. |

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
- qwen36 decodes through the serial path (`_decode_with_native_cache`), because mlx-vlm attention is not batchable. A runner hook that only covers the batched path never runs for it.
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

- `benchmarks/perf/2026-11-10/micro/VERDICT.md` with a disposition per experiment and a receipts map.
- `scripts/bench/mlx_*.py`, `decode_ab.py`, `decode_step_trace.py`.
- Patches 029 to 033 with unit tests and rows in `patches/README.md`.
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


## Results

Receipts and the full disposition table: `benchmarks/perf/2026-11-10/micro/VERDICT.md`.

| lever | setting | served result on qwen36 |
|---|---|---|
| how launches are committed | `MLX_MAX_MB_PER_BUFFER=200` | +4.2 to +6.1 % decode to 30K, +1.4 % at 122K, cold prefill 5 to 7 % shorter at 251K, 1.1 to 1.7 GB more wired peak |
| how many launches there are | `SGLANG_MLX_FUSE_MOE_ROUTER=1`, `SGLANG_MLX_FUSE_GDN_FRONT=1`, `SGLANG_MLX_FUSE_DECODE_LINEARS=1` (patches 030, 031, 032) | +11.6 / +11.0 / +10.1 % at 948 / 7,583 / 30,391 prompt tokens on top of the buffer limit: 95.9 / 90.0 / 75.1 tok/s against 85.9 / 81.1 / 68.2 |
| how attention partitions a deep cache | `SGLANG_MLX_SDPA_BLOCK_LADDER=1` (patch 029) | 37.3 against 40.8 ms per token at 251,474 prompt tokens, depth recall 6/6, stock block counts through 196,608 keys |

- The three launch-folding patches pass the teacher-forced fidelity gate: KL to the float32 yardstick 0.0189 (030), 0.0206 (031), bit-identical (032), 0.0211 together, against 0.0165 for the stock decode path and 0.0194 for the stock prefill path.
- Patch 032 pays only next to the other two (+1.6 to +3.1 % over three mirrored pairs) and is neutral alone.
- With the three patches installed a raw step is 1,850 primitives and 10.15 ms against 7.9 to 8.2 ms of weight and state traffic at 220 GB/s. No module family holds more than 1 ms above its reads, the 80 layer norms knock out to zero, and the head sits on its read. Further launch folding on this checkpoint is closed; the dense presets sit on the memory wall and have no such margin.
- Patch 033 is a correctness fix the fidelity instruments found: without it a radix prefix hit on an mlx-vlm Qwen checkpoint numbers its new tokens from zero. With it a scheduled re-prompt is bit-identical over 4,598 steps and a follow-up hit sits at KL 0.0064 from its cold twin (0.0658 without), the level two cold prefills on different chunk sizes show.
- Rejected after measurement: larger operation limits per command buffer, `MLX_METAL_FAST_SYNCH`, `MLX_BFS_MAX_WIDTH`, `mx.compile` around the blocks, constant gather indices, a fused GatedDeltaNet back kernel, int8 keys and values, a layer-norm fold.
- `SGLANG_MLX_CLEAR_CACHE_STEPS=0` is free to 122K and removes a 1.7 % stall pattern at 139K and 196K; the 256K arm decides it.
