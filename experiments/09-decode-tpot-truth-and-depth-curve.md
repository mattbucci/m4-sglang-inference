# decode-tpot-truth-and-depth-curve: fix the contaminated TPOT instrument, publish the true decode depth curve, go/no-go the decode-topk port


| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | instrument fix: hours; each ladder pass 1-2 h exclusive (157K rung ~10 min prefill per arm); decision memo: hours |
| **GPU time** | exclusive serving windows for the ladder (oom_guard mandatory) |
| **Depends on** | Exclusive serving window on m4-box.; Independent of disk-triage.; depth-recall-probe supplies the quality-parity gate any later decode-topk port needs.; R9700 patch 069 / 3090 patch 059 checkouts (verified present locally) — read-only, step 6 only. |
| **Provides to** | Doc 08 phase 2's secondary criterion, rewritten against the true curve.; The decode-topk-mlx port decision (would be patch 017) — absorbed here as a go/no-go, not built blind.; The five doc locations currently citing 13-19 s/token, re-anchored. |

## Objective

The fleet-cited constraint "decode TPOT at depth = 13-19 s/token" is whole-request elapsed (prefill included) divided by completion tokens — an instrument artifact until proven otherwise. Fix the instrument, publish the true steady-decode s/token curve at depth, and decide the MLX decode-topk port on the measured curve instead of a number that may be ~96% prefill.


## Hypothesis

The dominant share of 13-19 s/token is prefill amortization. If true decode at 128K lands within ~2x of the 32K rate, the "decode wall" dissolves and doc 08 phase 2's secondary criterion is rewritten; if a real multi-x wall exists, the decode-topk port gets an honest baseline instead of a phantom one.


## Background & receipts

- Instrument defect, verified in-repo: scripts/bench/bench_long_context.py:58 computes `tpot = (elapsed / completion_tokens * 1000)` where `elapsed` (line 52) spans the whole request including prefill. No streaming, no TTFT, no ITL anywhere in the script.
- Arithmetic proof: benchmarks/longctx-bisect/presize-160k-chunked2048.txt line 5 — in=157,287 out=32 time=606.9s TPOT=18965.2ms. 606.9 s / 32 = 18.97 s: the receipt's TPOT is ~95-97% prefill amortization by construction. The ATTRIBUTION.md decode ladder (1.6 s @32K / 4.4 s @64K / 13 s @128K) is the same instrument at out=32 — same contamination class.
- Nothing owns the defect: commit 9b397a5 hardened only bench_all_unified.py's guards; no commit or doc notes that bench_long_context.py's TPOT includes prefill.
- Five locations repeat the number: README.md:118, CLAUDE.md:118, experiments/08-beyond-128k.md (phase 2 direction-1 rationale + secondary success criterion), experiments/README.md row 2. Doc 08 phase 2 is being planned against it.
- Fleet lesson: decode truth comes from server-log gen-throughput at real depth; client TPOT under-measures ~2x; short-depth "@256K" numbers collapse at true depth.
- Sister decode-topk precedent, twice proven against MEASURED walls: R9700 patch 069 (1.77x @256K), 3090 patch 059 (2.03x @262K, crossover ~80-90K, opt-in). An L-effort MLX port must not be built against a ~96%-prefill metric — that decision is step 6.


## Method

1. Instrument: extend scripts/bench/bench_long_context.py to `stream: true` on /v1/completions; record TTFT and per-token ITL separately (list + p50/p95 + steady mean past the first token); enforce server-verified `usage.prompt_tokens >= 0.95x label` per rung; rename the old column `amortized_s_per_token` so historical receipts stay interpretable.
2. Cross-check the streamed decode rate against server-log gen-throughput on the same run (client TPOT under-measures ~2x); require agreement within 10% before publishing any number.
3. Erratum: append one line to benchmarks/longctx-bisect/ATTRIBUTION.md naming the amortization defect and the corrected instrument; after step 4 lands, re-anchor all five citing locations to the measured curve.
4. Measure: `CTX=175000 MEM_FRAC=0.5 CHUNKED=2048 EXTRA_ARGS="--disable-radix-cache" scripts/launch.sh qwen36 --kv-cache turboquant`, oom_guard.sh + mem_profile.sh armed; ladder 32K/64K/96K/128K/157K, each at BOTH out=32 AND out=256 (separates post-prefill transients from steady decode); receipts to benchmarks/longctx-bisect/decode-curve/.
5. Attribute: same-depth decode A/B at 32K where both configs fit — radix-on (pool path) vs radix-off (ContiguousAttentionKVCache path), server-log gen-throughput both arms.
6. Decide: decision memo. If true 128K decode is within ~2x of the 32K rate (~17-21 tok/s per benchmarks/radix-ab/), the wall dissolves — rewrite doc 08 phase 2's secondary criterion. If a real multi-x wall exists, spec the MLX decode-topk port as a follow-on numbered doc with this curve as its baseline: per-block mean-pooled key summaries, opt-in budget env, sink + last-window always included, scorer validated on real keys BEFORE benchmarking (R9700 lesson).


## Baseline & instrument

Baseline is the amortized column itself (presize-160k-chunked2048.txt, ATTRIBUTION.md decode rows) — kept, renamed, contrasted with the streamed ITL curve. Ground truth is server-log gen-throughput per rung; client stream timing must agree within 10% or the server-log number wins.


## Success criteria

- Corrected instrument committed: TTFT/ITL separated, old column renamed `amortized_s_per_token`, 0.95x label guard enforced, server-log cross-check within 10%.
- True decode s/token published at 5 server-verified depths, at out=32 and out=256 each, receipts under benchmarks/longctx-bisect/decode-curve/.
- All five doc locations citing 13-19 s/token re-anchored to the curve.
- Decision memo with an explicit go/no-go on the decode-topk port, justified by the measured curve and the radix-on/off attribution.


## Kill criteria

- Any depth rung without server-verified prompt_tokens, or any curve published from the amortized metric — void; re-run or record the null.
- Client and server-log rates disagree >10% after one instrumentation fix attempt — publish server-log-derived numbers only, record the client-path discrepancy as a finding.
- Guard kill at a rung: record with mem_profile receipts, publish the partial curve with the rung marked unreachable — itself a result.


## Deliverables

- scripts/bench/bench_long_context.py streaming/ITL extension (column rename included).
- benchmarks/longctx-bisect/decode-curve/ rung receipts (out=32 and out=256).
- ATTRIBUTION.md erratum line + the five re-anchored doc locations.
- Decision memo: decode-wall verdict + decode-topk go/no-go; on "go", the follow-on numbered port spec added to experiments/.


## Constraints

- oom_guard.sh MANDATORY every rung (macOS has no OOM killer); exclusive window per pass; no concurrent benches; long urllib timeout (157K prefill ~10 min).
- Decode claims from server-log gen-throughput only, measured at real depth.
- One mechanism at a time: identical launch recipe across rungs; only depth and out-tokens vary; radix state varies only in the step 5 A/B.


## Risks

- out=256 at 157K adds minutes-scale decode on a slow rung — bounded by landing the out=32 arm first at every depth.
- Streaming may itself perturb timing on this stack — the server-log cross-check is the arbiter, not the client clock.
- Decoded-filler quality is uninspected here; depth-recall-probe owns quality-at-depth and gates any port that changes attention math.
