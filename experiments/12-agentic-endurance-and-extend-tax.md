# agentic-endurance-and-extend-tax: retention growth over 150+ turns and append-to-cached-prefix TTFT on MLX radix hybrids


| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | ~4-6 h exclusive across phases (phase A turn-tax ~1-2 h; phase B endurance 150-200 turns + two control arms) |
| **GPU time** | exclusive serving windows, radix-ON serving (oom_guard mandatory — budget for the ~0.4 GB/request outside-pool retention) |
| **Depends on** | Exclusive serving windows.; Independent of disk-triage and docs 07/08.; R9700 checkout for the reference script (scripts/bench/measure_extend_cost.py verified present locally; fetch their repo first — the checkout may be stale).; /flush_cache endpoint (verified in use: scripts/eval/test_radix_cache_repeat.py:31). |
| **Provides to** | A max-safe-session-length bound (or validated mitigation cadence) for the rig's most productive workload (qwen36 + opencode).; Sister READMEs: the cross-rig turn-tax comparison note the R9700 audit asked all three rigs for, carried with the queued allocator-cache-cap note (experiments/README.md cross-rig edges).; A fix-experiment row in experiments/README.md if the turn tax reproduces. |

## Objective

The most productive workload — multi-turn opencode sessions on radix-on hybrids (benchmarks/radix-ab/VERDICT.md) — has two unowned risks on one axis. Memory: radix-on retention was characterized to exactly 10 requests, ~0.4 GB/request OUTSIDE the static pool; if linear, a 50-100-turn session crosses the 8 GB guard floor, and macOS has no OOM killer — the failure is a whole-box stall mid-session on the workload Matt uses daily. Latency: R9700's R97-J measured 604.6 ms TTFT to append ONE token to a 176K cached prefix (~85x decode cost; the cost is the prefix walk, not the suffix), and their audit asks all three rigs to check. No M4 receipt measures either.


## Hypothesis

Phase A: the append-to-cached-prefix turn tax reproduces on MLX (the prefix-walk cost is scheduler/radix-side, not CUDA-side) at some ratio of decode cost. Phase B: 10 requests cannot distinguish plateau from linear — the VERDICT.md open thread ("what exactly radix-on retains outside the pool across requests", lines 59-61) has no owner. Either a bound or a mitigation cadence comes out.


## Background & receipts

- Retention receipt: benchmarks/radix-ab/VERDICT.md — radix-on retains ~0.4 GB/request outside the static pool across its 10-request characterization, free+inactive troughing at 10.2 GB; open thread at lines 59-61 explicitly unowned.
- Failure mode: CLAUDE.md OOM rule — no macOS OOM killer, box stalls until reboot; guard line 8 GB free+inactive (scripts/common/oom_guard.sh).
- Turn-tax reference: R9700 R97-J — 604.6 ms TTFT for prompt+1 on a 176K cached prefix, ~85x their decode cost; reference script ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/bench/measure_extend_cost.py (verified present).
- Radix-on hybrid serving is the validated agentic default (CLAUDE.md: no_buffer strategy, greedy-determinism-validated); doc 08 is single-request deep prefill — a different axis (depth vs turns). No numbered doc, patch, or benchmark dir covers session-scale behavior.
- Mitigation instruments present: /flush_cache (test_radix_cache_repeat.py:31); SGLANG_MLX_CACHE_LIMIT_GB attribution methodology (patch 008); mem_profile.sh.


## Method

1. Phase A prep: fetch the R9700 repo, adapt their measure_extend_cost.py into scripts/bench/measure_extend_cost.py for MLX. Launch radix-ON (no --disable-radix-cache): `CTX=140000 MEM_FRAC=0.5 CHUNKED=2048 scripts/launch.sh qwen36 --kv-cache turboquant`, oom_guard.sh + mem_profile.sh armed.
2. Phase A measure: warm a radix prefix at depths {8K, 32K, 64K} with real-content prompts; re-send prompt+1 and prompt+64 tokens; record TTFT per append with the cache hit CONFIRMED from server-log cached/prefix token counts (no cached-token line = no data point). Comparator: decode ms/token at the same depth from the same server log.
3. Phase A verdict line: "turn tax reproduces on MLX: yes/no, ratio Nx decode". If it reproduces, profile whether the cost sits in the pool-gather or a contiguous-cache rebuild, and file a fix-experiment row in experiments/README.md.
4. Phase B driver: write scripts/bench/bench_session_endurance.py — shared system+repo prefix, each turn appends ~1.5K tokens of tool-output-shaped text, replies capped at 256 tokens, prefix grows to ~32K then rolls, 150-200 turns; per-turn log of TTFT, decode tok/s, server-verified prompt_tokens, cached-token count; mem_profile.sh at 1 s cadence.
5. Phase B control arms: (a) identical driver radix-OFF (isolates radix-attributable retention); (b) radix-on with `POST /flush_cache` every 25 turns (mitigation-cadence arm).
6. Classify the retention curve: plateau (report the bound and max safe session length) vs linear (report GB/turn slope; attribute via SGLANG_MLX_CACHE_LIMIT_GB variation, patch-008 methodology; validate a cadence holding free+inactive above 8 GB for 200 turns). Receipts to benchmarks/session-endurance/.
7. Deliver the cross-rig turn-tax comparison note to the sister READMEs (grep-verified in their checkouts, honoring the 3090 delivery gate) together with the queued allocator-cache-cap note from experiments/README.md cross-rig edges.


## Baseline & instrument

Phase A baseline: decode ms/token at the same depth, same server log — the tax is reported as a ratio, not an absolute. Phase B baseline: the radix-ab 10-request characterization (~0.4 GB/request, trough 10.2 GB) — the endurance run extends the same free+inactive instrument to 150-200 turns with the radix-off arm as the attribution control.


## Success criteria

- Committed table of TTFT(append-1) / TTFT(append-64) / decode-ms-per-token at >= 3 depths with server-verified cached-prefix lengths and an explicit reproduces-yes/no verdict; if the ratio exceeds ~5x decode at 32K, a fix-experiment row is filed in experiments/README.md.
- A 150+ turn session completed (or guard-killed with full receipts) and the retention curve classified: plateau → bound + max safe session length; linear → GB/turn, attribution, and a validated mitigation cadence holding free+inactive above 8 GB for 200 turns.
- Cross-rig note grep-verifiable in at least one sister README.


## Kill criteria

- TTFT numbers without cache-hit verification, or a retention verdict without the radix-off control arm, or any run without oom_guard — void.
- Guard kill during phase B is NOT a kill — it is the finding (record turn number and the per-turn memory log, proceed to the mitigation arm).
- /flush_cache proves ineffective AND cache-limit variation doesn't attribute the growth — fall back to the periodic-server-restart mitigation arm, record the cadence that survives 200 turns, file the mechanism as an upstream question.


## Deliverables

- scripts/bench/measure_extend_cost.py (MLX adaptation) + scripts/bench/bench_session_endurance.py.
- benchmarks/session-endurance/ per-turn logs, mem profiles, and the retention-curve classification.
- Turn-tax verdict table + cross-rig comparison note in sister READMEs.
- experiments/README.md fix-experiment row if the tax reproduces; README session-length guidance for the agentic workload.


## Constraints

- oom_guard.sh MANDATORY throughout — phase B deliberately walks toward the guard line; that is the one sanctioned way to find it.
- Radix-ON serving is the object under test; only the control arms differ.
- Real-content prompts for prefix warming (fleet lesson: validate on real keys/content, not synthetic filler, before trusting latency numbers).
- Exclusive windows; server-log is the source of truth for cached-token counts and decode rate.


## Risks

- Rolling the prefix at ~32K may reset radix retention and mask linear growth — the per-turn cached-token log distinguishes roll events from leak growth.
- The MLX adaptation may not hit the radix path the way the CUDA original does — per-append server-log cached-token confirmation is the gate against measuring the wrong path.
- 150-200 turns at real decode rates is hours-scale; the reply cap (256) and turn size (~1.5K) keep phase B inside one exclusive window.
