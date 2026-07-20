# prefill-budget-ledger-192k: 192K knob probes before phase-2 surgery — sub-2048 chunks, exact-CTX pool, MEM_FRAC split, GB-shortfall ledger


| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | 15-45 min per arm (4 arms + optional 128K control and 210K attempt); ledger math minutes |
| **GPU time** | exclusive serving window per arm (oom_guard mandatory) |
| **Depends on** | Exclusive serving windows only.; No patch, no download — unaffected by the disk-triage sign-off. |
| **Provides to** | Doc 08 phase 2 — either a cheap 192K win that resets its target, or the exact GB minimum-recovery number its surgery must beat.; benchmarks/longctx-bisect/ATTRIBUTION.md — new measurement rows + the stale Consequences fix. |

## Objective

The 192K death is steady budget exhaustion, not a spike (ATTRIBUTION.md:34, presize-192k-chunked2048: guard-killed at ~180K prefilled, CTX 210K, mf 0.5, chunked 2048). Three cheap no-patch levers were never probed at 192K. Probe them; whichever way the arms fall, produce the exact GB-shortfall ledger doc 08 phase 2 needs as its minimum-recovery target.


## Hypothesis

Any one of (a) sub-2048 chunks, (b) CTX pinned to exactly 196,608, (c) MEM_FRAC 0.45 may buy the ~2-3 GB the run misses; if none do, the per-arm free+inactive-at-death measurements bound phase 2's required recovery precisely. Either outcome is a win.


## Background & receipts

- Death receipt: benchmarks/longctx-bisect/ATTRIBUTION.md:34 — CTX 210K, mf 0.5, chunked 2048, patch-015 pre-size; guard-killed ~180K; "no single spike, steady exhaustion". Raw: presize-192k-chunked2048.txt.
- Lever (a), chunk floor: the campaign's load-bearing discovery is that chunk size sets the transient floor — chunked 4096 kills every deep run at ~100-113K, chunked 2048 completes 160K (ATTRIBUTION.md; CLAUDE.md long-context rule). 1024 and 512 are unexplored below the floor.
- Lever (b), pool exact-sizing: the dead run auto-sized the pool for CTX=210K against a 196,608-token (192K) label — ~14K tokens of pool (KV + mamba aux) overallocated, never used.
- Lever (c), budget split: MEM_FRAC 0.5 is the only value probed deep; 0.45 shifts budget from the pool to the per-request bf16 cache + transients.
- Ledger instruments present: scripts/common/mem_profile.sh, scripts/bench/compute_growth_rate.py, guard line = 8 GB free+inactive (scripts/common/oom_guard.sh).
- Piggybacked hygiene: ATTRIBUTION.md's Consequences paragraph still asserts "Certified ceiling: 128K" (line ~86) — verified stale, 160K is validated (presize-160k-ctx175k.txt, CLAUDE.md). Fix while editing the same file.
- Prefill-speed reference for the chunk-cost row: 157,287 tokens in 606.9 s at chunked 2048 (presize-160k-chunked2048.txt).


## Method

Every arm: exclusive window, `bash scripts/common/oom_guard.sh &` + `bash scripts/common/mem_profile.sh &` running, receipts to benchmarks/longctx-bisect/.

1. ARM 1 (pool-exact control): `CTX=196608 MEM_FRAC=0.5 CHUNKED=2048 EXTRA_ARGS="--disable-radix-cache" scripts/launch.sh qwen36 --kv-cache turboquant`; then `python scripts/bench/bench_long_context.py --contexts 196608 --output-tokens 32`. One variable changed from the dead run — isolates pool overallocation vs the CTX=210K death.
2. ARM 2 (chunk floor): same launch with CHUNKED=1024; record prefill time vs the 606.9 s 160K/2048 reference (the speed cost of halving the chunk). Descend to CHUNKED=512 only if 1024 still dies AND its mem_profile transient dip visibly shrank vs 2048 — otherwise the lever is exhausted.
3. ARM 3 (budget split): `CTX=196608 MEM_FRAC=0.45 CHUNKED=2048`, same bench line.
4. ARM 4 (ledger): run compute_growth_rate.py over each arm's mem_profile csv; for every failed arm compute free+inactive at death minus the 8 GB guard line = GB shortfall; append one row per arm to the ATTRIBUTION.md measurements table (same columns as line 34's row). Fix the stale "Certified ceiling: 128K" Consequences paragraph in the same edit.
5. If any arm completes server-verified: rerun the 128K control (`CTX=140000 MEM_FRAC=0.5 CHUNKED=2048`, must complete server-verified — doc 08's validation ladder), then attempt 210K with the winning combo. Receipts for both either way.


## Baseline & instrument

Baseline is the dead run itself (presize-192k-chunked2048.txt: guard kill at ~180K prefilled) — every arm changes exactly one knob from it. Instrument is bench_long_context.py with server-verified `usage.prompt_tokens >= 0.95x label` plus the mem_profile.sh trace; shortfall arithmetic via compute_growth_rate.py against the 8 GB guard line.


## Success criteria

PASS is either outcome:

- A 192K-labeled run completes with server-verified `usage.prompt_tokens >= 0.95x label`, no guard kill, and the 128K control still passes; or
- A committed shortfall ledger in ATTRIBUTION.md with GB-at-death per arm plus the CHUNKED=1024-vs-2048 prefill-speed cost at equal depth — quantifying exactly how much memory phase 2 must recover.

Plus, unconditionally: the stale "Certified ceiling: 128K" paragraph fixed.


## Kill criteria

- Arms run without mem_profile receipts, or results left out of ATTRIBUTION.md — the arm is void; a probe that doesn't feed the ledger is wasted box time.
- Two consecutive guard kills of the SAME arm with death depths >10K apart — stop the arm, record the nondeterminism as a finding before trusting its shortfall number.
- CHUNKED=512 prefill projects past ~30 min for the 192K label — record the speed wall and stop descending; the lever is priced out.


## Deliverables

- benchmarks/longctx-bisect/ receipts per arm (bench output + mem_profile csv), plus 128K-control and 210K-attempt receipts if step 5 triggers.
- ATTRIBUTION.md: one measurement row per arm, the GB-shortfall ledger, the Consequences-paragraph fix.
- One-paragraph verdict in ATTRIBUTION.md: which lever (if any) moves the ceiling, and the phase-2 minimum-recovery number.


## Constraints

- oom_guard.sh MANDATORY every arm (macOS has no OOM killer); never raise MEM_FRAC above 0.5 here — the lever moves DOWN for long-context (CLAUDE.md).
- One knob per arm from the dead-run baseline; no combining levers until step 5's winning-combo attempt.
- No patch-stack changes — this is the no-surgery complement to doc 08 phase 2, not a substitute.


## Risks

- Interaction effects: exact-CTX + 1024 chunks might pass where neither alone does — step 5 covers the winner-adjacent pairing; the full cross-product is out of scope (4 arms bound the box time).
- MEM_FRAC 0.45 may shrink the pool below the 196,608-token requirement and refuse to boot or cap max-total-tokens — a boot-time refusal is itself a ledger row (pool floor), not a failure.
