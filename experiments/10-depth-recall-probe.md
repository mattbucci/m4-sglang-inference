# depth-recall-probe: multi-needle recall ladder to 157K plus turboquant-vs-fp16 KV A/B


| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | probe script: hours; ladder pass ~2-3 h exclusive including the 157K rung (~10 min prefill, amortized once); KV A/B + tripwire wiring 1-2 h |
| **GPU time** | exclusive serving windows (oom_guard mandatory) |
| **Depends on** | Exclusive serving window.; Independent of disk-triage.; Should land before or alongside decode-tpot-truth-and-depth-curve (supplies its quality gate) and before doc 08 phase 2 (supplies its drift gate). |
| **Provides to** | Doc 08 phase 2's kill criterion — the named, implemented drift gate it currently presumes.; decode-tpot-truth-and-depth-curve — the quality-parity gate any decode-topk port must pass.; benchmarks/baselines.json — a standing 32K recall tripwire (schema v2). |

## Objective

The long-context campaign validates that prefill COMPLETES, never that the model retrieves anything at depth. Build the missing quality-at-depth axis: a multi-needle positional recall ladder to 157K, a turboquant-vs-fp16 KV A/B, and a standing recall tripwire — before the 192K/256K campaign optimizes a ceiling the model may not be able to use.


## Hypothesis

qwen36 under the validated deep recipe retrieves >= 5/6 uniquely-keyed facts through 128K, and turboquant costs no recall vs fp16 KV at 32K. If either fails, the finding blocks the depth campaign behind quality attribution — cheaper to learn now than after phase-2 surgery.


## Background & receipts

- Only recall instrument in-repo: scripts/eval/eval_and_chart.py:347 `needle_eval(chat_url, lengths=[1024, 4096, 16384, 65536], ...)` — one needle, one position, default cap 64K, run in practice at 1024/4096/16384 (doc 07 method step 8's `--needle-lengths`).
- The 160K receipt decodes 32 uninspected tokens of filler (benchmarks/longctx-bisect/presize-160k-chunked2048.txt) — completion, not retrieval.
- Every deep run uses `--kv-cache turboquant`; its recall cost has never been A/B'd against fp16 KV at any depth on this rig.
- Doc 08 phase 2's kill criterion ("probe verdicts flip ... vs the fp16 cache") presumes a depth-drift gate that has no implementation.
- Sister rigs treat recall-at-depth as mandatory: 3090's patch 059 promotion required perfect 256K recall; R9700 needle A/Bs carry a KV-dtype caveat.
- Filler sizing solved in-repo: the 6-chars/token + 1.1 safety estimator (scripts/bench/bench_long_context.py:27-28) lands within ~5% of target.
- Comparators exist: benchmarks/sampling-ab greedy token-identity set; /flush_cache endpoint in use (scripts/eval/test_radix_cache_repeat.py:31).


## Method

1. Write scripts/eval/probe_depth_recall.py: 6 uniquely-keyed facts (key→value pairs no filler collision can produce) at fractional positions {0.05, 0.25, 0.45, 0.65, 0.85, 0.98} in filler sized by the 6-chars/token + 1.1 estimator; greedy /v1/completions; require server-verified `usage.prompt_tokens >= 0.95x label` per prompt; reuse needle_eval's server-death-vs-wrong-answer discrimination; record the filler seed per prompt so re-runs are controlled A/Bs (R9700 replayability technique); JSON receipts to benchmarks/quality/depth-recall/.
2. Ladder 8K/32K/64K/96K/128K/157K on the validated deep recipe (`CTX=175000 MEM_FRAC=0.5 CHUNKED=2048 EXTRA_ARGS="--disable-radix-cache" scripts/launch.sh qwen36 --kv-cache turboquant`), oom_guard.sh + mem_profile.sh armed. At 157K ask all needles inside ONE prefill via a single multi-question prompt — the ~10 min prefill amortizes once, the rung stays under ~15 min.
3. KV-dtype A/B at 32K (both configs fit): turboquant vs auto KV, same filler seed — recall delta plus greedy token-identity on the benchmarks/sampling-ab determinism set. This is the fp16 reference doc 08 phase 2's kill criterion needs.
4. Wire the 32K recall point into scripts/bench/bench_regression.sh / benchmarks/baselines.json (schema v2 quality field) as a standing tripwire; negative-test both ways by perturbing a stored answer (3090 pattern) — must fire on the perturbation, pass clean after revert.
5. Name this probe in doc 08's kill criterion (replacing the unimplemented clause) and hand it to decode-tpot-truth-and-depth-curve as the quality-parity gate any decode-topk port must pass.


## Baseline & instrument

Baseline is the existing needle_eval verdict at 1024/4096/16384 (3/3 in benchmarks/quality/Qwen3.6-35B-A3B.json) — shallow, single-position. The new instrument extends the same server-death-vs-wrong-answer discrimination to 6 positions x 6 depths with per-prompt server-verified token counts; the KV A/B determinism arm reuses the sampling-ab greedy-identity method.


## Success criteria

- Recall scored at all 6 rungs (8K/32K/64K/96K/128K/157K) with server-verified depth receipts under benchmarks/quality/depth-recall/.
- turboquant-vs-fp16 delta at 32K quantified: recall count AND greedy token-identity verdict, same seed both arms.
- 32K tripwire armed in baselines.json and negative-tested both ways.
- Doc 08 kill criterion updated to name probe_depth_recall.py.
- DECISION RULE: recall >= 5/6 through 128K certifies quality-at-depth for the campaign; recall <= 3/6 at any rung <= 128K files a blocking finding re-prioritizing 192K/256K behind quality attribution (turboquant vs model vs chunked-prefill, isolated one mechanism at a time).


## Kill criteria

- Any rung reported without the server-verified token count — void, re-run.
- Guard kill at a rung: record with mem_profile receipts; the ladder below it still stands (partial certification, rung marked unreachable).
- Filler-seed replay fails to reproduce a verdict — stop, fix the instrument before publishing any recall number; a probe that can't replay can't tripwire.


## Deliverables

- scripts/eval/probe_depth_recall.py (seeded, position-swept, multi-needle).
- benchmarks/quality/depth-recall/ rung receipts + the 32K KV-dtype A/B pair.
- baselines.json schema-v2 quality field + bench_regression.sh wiring, with the both-ways negative-test receipt.
- Doc 08 kill-criterion update naming the probe; gate handoff noted in decode-tpot-truth-and-depth-curve.


## Constraints

- oom_guard.sh MANDATORY for every 32K+ rung; exclusive serving window; no concurrent benches.
- Greedy decode only — recall deltas must not be confounded with sampling.
- One mechanism at a time: KV dtype is the only variable in step 3; depth the only variable across rungs.
- Full-size qwen36 flagship only (no small-variant substitution).


## Risks

- Multi-question single-prefill at 157K may confound retrieval with instruction-following over 6 questions — the shallower rungs (same 6-question format) calibrate the format cost before blaming depth.
- fp16 (auto) KV raises pool size vs turboquant — if the auto-KV arm can't fit CTX for the 32K A/B, drop the A/B depth to 16K and record why.
- Position-fraction placement lands ±5% of target token offset (estimator error) — record realized offsets from the tokenized prompt, not intent.
