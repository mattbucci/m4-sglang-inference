# M4-B: Implement temperature/top-p/top-k/min-p sampling in the MLX backend via mlx-lm make_sampler

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | 1-2 days (implementation ~0.5 day; gates, benches, probes, patch capture ~0.5-1 day) |
| **GPU time** | m4-box only: ~4-6h of local M4 serving (baselines, A/B benches, probes on qwen36 + loop-repro preset + one VLM preset); zero 3090/R9700 GPU time |
| **Depends on** | None hard. Soft: box-time coordination with the long-context item (same single machine; run serially).; components/sglang must exist on the box: `test -d components/sglang/.git || bash scripts/setup.sh`.; mlx-lm with `make_sampler` present in the repo .venv (verify `python -c 'from mlx_lm.sample_utils import make_sampler'` and record the installed version — 0.31.3 verified; the dep is unpinned in the sglang extras so import defensively). |
| **Provides to** | In-house qwen36 quant (M4-E): the probe outcome disentangles greedy-decode vs router/DeltaNet-quant as the qwen35 infinite-`<think>` root cause before that build is judged.; M4 agentic evals: prerequisite for retiring evals/swebench/no_thinking_proxy.py and running thinking-mode SWE-bench/opencode cells (the 3090's developer-role template fix is the named follow-up donor: their scripts/eval/patch_chat_templates_developer_role.py).; Fleet parity: M4 probe/eval rows become comparable to 3090/R9700 thinking cells that run with --sampling-defaults model. |

## Objective

Remove the M4 rig's greedy-only decode limitation by wiring SGLang per-request
sampling params into the MLX backend's token-selection sites using mlx-lm's
`make_sampler`. This is the sampling-side blocker on thinking-mode parity with
the fleet — and qwen35 (Qwen3.5-27B-4bit) still infinite-loops in `<think>`
under greedy decode, forcing its thinking probe to be skipped.

## Hypothesis

Routing SGLang per-request sampling params (temp/top-p/top-k/min-p) into the
MLX backend via `make_sampler` (i) terminates the remaining Qwen3.x
infinite-`<think>` loop class — probe_thinking.py reaches `</think>` and
returns VERIFIED on a preset whose baseline first CONFIRMED the loop (qwen35 =
Qwen3.5-27B-4bit is the receipt-backed looper; qwen36 passes thinking under
greedy on the current stack and is therefore untestable for the loop claim) —
while (ii) keeping greedy-path decode tok/s within ±3% (server-log gen
throughput) and greedy outputs token-identical. Falsified on (i) if the loop
persists under sampling on a preset that demonstrably looped at baseline
(which would confirm the DeltaNet/router-quantization hypothesis instead); a
preset that does NOT loop at baseline is dropped, not counted as a pass.
Falsified on (ii) if overhead exceeds 3% on the greedy path.

## Background & receipts

- Site map: the live tree (v0.5.15.post1 + 6 patches) has **7 `mx.argmax`
  sites**, all in
  `python/sglang/srt/hardware_backend/mlx/model_runner.py`, across
  prefill_start / extend_start / decode_batch_start (incl. the serial
  `_decode_with_native_cache` path used by VLM-arch and rope-less models) /
  decode_batch_start_chained. Method step 1 makes the live grep the hard
  stop-gate — reconcile count and functions before editing. Note some sites
  are upstream-origin and some live in patch 008's serial-decode routing;
  greedy-only is partly upstream design.
- `make_sampler` verified in the installed wheel:
  `mlx_lm/sample_utils.py::make_sampler(temp, top_p, min_p,
  min_tokens_to_keep, top_k, xtc_*)` — temp==0 returns an argmax lambda;
  otherwise chains apply_top_p/apply_min_p/apply_top_k then
  `mx.random.categorical(logits * (1/temp))`. All filter ops use axis=-1, so
  (B, vocab) batched input is safe.
- Load-bearing semantics: the sampler takes LOG-PROBABILITIES, and
  apply_top_p is NOT shift-invariant (probs = mx.exp(logprobs) + cumsum must
  sum to 1). Raw logits must be normalized first: `logprobs = logits -
  mx.logsumexp(logits, axis=-1, keepdims=True)` — mlx_lm's own batched
  convention in its generate path.
- Plumbing gap: zero occurrences of temperature/top_p/sampling in the MLX
  backend's model_runner.py / tp_worker.py / scheduler_mixin.py — requests
  with temperature>0 are silently decoded greedy today (no error). tp_worker
  has the request objects in scope at every runner call site, so
  `req.sampling_params` is reachable everywhere.
- SGLang SamplingParams normalization (`sampling/sampling_params.py`):
  temperature < eps is rewritten to temperature=1.0 + top_k=1 (greedy is
  encoded as top_k==1), and top_k==-1 becomes TOP_K_ALL = 1<<30.
  GREEDY/TOP_K==1 ALIASING (document, not a bug): a genuine user request of
  top_k=1 with temperature>0 is indistinguishable from greedy and is routed
  to mx.argmax — acceptable single-user behavior, recorded in
  patches/README.md so it is not later mistaken for a sampling defect.
  Per-request sampling_seed exists; server-level random_seed is auto-filled.
- `--sampling-defaults` defaults to 'model' — once the backend honors
  sampling params, model-recommended sampling (generation_config.json)
  applies automatically to requests that omit temperature.
- Loop-repro landscape: qwen36 passes probe_thinking VERIFIED under greedy on
  the current stack, so it CANNOT establish the loop-fix claim. The
  receipt-backed INT4 looper is `mlx-community/Qwen3.5-27B-4bit` (the qwen35
  preset — its thinking probe is skipped as a known looper; the checkpoint
  audit flags its DeltaNet in_proj INT4 as a competing root cause). Use
  qwen35 if it fits RAM under MEM_FRAC 0.7; qwen35-9b-8bit is only a
  RAM-driven proxy and its baseline MUST first establish that it actually
  loops before any loop-fix claim is counted.
- Validation instrument: `scripts/eval/probe_thinking.py` (exit 0 = THINKING
  VERIFIED; checks answer + intermediate step + non-empty reasoning_content
  AND content) hardcodes temperature 0 — it needs a --temperature/--top-p
  flag to serve as the post-change probe.
- Presets: qwen36 = `mlx-community/Qwen3.6-35B-A3B-4bit`, CTX=32768,
  MAX_RUNNING=1, radix cache ON (hybrids run the normal event loop),
  `--reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-multimodal`. qwen35-9b-8bit MODEL =
  `mlx-community/Qwen3.5-9B-MLX-8bit` — receipts must cite the MLX-8bit path.
- Patch capture: next free number is **009** (current set
  002/003/005/007/008/014; setup.sh's `0[01][0-9]-*.patch` glob covers it —
  no glob change needed). Deliverable: `patches/009-mlx-sampling.patch`,
  replay-verified against pristine v0.5.15.post1.
- Full-thinking-parity donors (out of scope, named follow-ups): the 3090's
  `patch_chat_templates_developer_role.py` and retiring M4's
  `no_thinking_proxy.py` (opencode can't see reasoning_content; real
  sampling is the prerequisite).
- Perf instruments: `scripts/bench/bench_quick.sh` (note the
  --random-range-ratio caveat tracked as its own queue item) and server-log
  gen throughput (fleet-canonical). `scripts/common/oom_guard.sh` mandatory
  for ≥32K work.

## Method

1. Precheck: `test -d components/sglang/.git || bash scripts/setup.sh`, then
   `grep -n 'mx.argmax'
   components/sglang/python/sglang/srt/hardware_backend/mlx/model_runner.py`
   — reconcile the live count (7) and per-function distribution before
   editing; any mismatch is a HARD STOP. Confirm `python -c 'import mlx_lm;
   from mlx_lm.sample_utils import make_sampler; print(mlx_lm.__version__)'`
   in the repo .venv and record the version.
2. Capture baselines (see baseline field), including the HARD PRECONDITION
   that the loop-repro preset's pre-change probe shows an actual
   looping/DEGRADED transcript (no `</think>`, finish_reason=length). For
   token-identity: 3 fixed prompts, temperature=0, max_tokens=128, save
   completion token content per prompt for qwen36.
3. Implement in model_runner.py: (a) a per-rid registry `self._req_sampling:
   dict[str, SamplingSpec]` where SamplingSpec is a frozen tuple
   (temperature, top_p, top_k, min_p) derived from SGLang's SamplingParams;
   (b) mapping rule — top_k==1 → greedy (mx.argmax); else
   make_sampler(temp=sp.temperature, top_p=sp.top_p, min_p=sp.min_p,
   top_k=0 if sp.top_k >= (1<<30) else sp.top_k); cache sampler closures
   keyed by SamplingSpec; (c) `_select_tokens(logits_2d, req_ids)` →
   logprobs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
   (REQUIRED: apply_top_p exponentiates and cumsums, raw logits give wrong
   nuclei); all-greedy batch → mx.argmax unchanged; homogeneous sampled →
   one sampler call on (B,V); heterogeneous → per-row sampler calls,
   concatenated.
4. Plumb params in: add optional `sampling_params=None` kwarg to
   prefill_start (and the sync prefill wrapper), registering into
   `_req_sampling`; pass `req.sampling_params` at the tp_worker call sites
   (async + sync prefill routes); decode_batch_start / extend_start /
   decode_batch_start_chained only look up by rid. Clear registry entries in
   `remove_request` / `_cleanup_stale_rids`.
5. Replace the argmax sites with `_select_tokens` (single-request
   prefill/extend sites pass [req_id]; batched sites pass the batch's
   req_ids; the serial `_decode_with_native_cache` loop passes one rid per
   iteration). Seed once at runner init: mx.random.seed(server random_seed)
   — plumb server_args.random_seed through MlxModelRunner.__init__.
   Per-request sampling_seed is a documented non-goal (single global MLX
   random stream).
6. Gate 1 (greedy identity): relaunch qwen36, rerun the 3-prompt
   temperature=0 set — token-identical to step-2 capture. Gate 2 (sampling
   works): same prompts with temperature=1.0, 5 repeats — at least 4/5
   pairwise-distinct completions; run scripts/test/test_sampling_select.py
   unit sanity.
7. Gate 3 (thinking probe): extend probe_thinking.py with
   --temperature/--top-p; run `--temperature 0.6 --top-p 0.95` AND a no-flag
   run relying on --sampling-defaults model (verify the server log shows
   model generation_config defaults picked up) on qwen36, then on the
   loop-repro preset that CONFIRMED DEGRADED at baseline. Record
   VERIFIED/DEGRADED + finish_reason + usage for each; a preset that
   returned VERIFIED at baseline is dropped from the loop-fix claim.
8. Gate 4 (perf): server-log gen-throughput A/B at MR=1 — greedy pre vs post
   (within ±3%), and sampled vs greedy post (record the delta as the sampler
   cost). Then batched check: a MAX_RUNNING=4 preset with a 4-request mix (2
   greedy, 2 temp=1.0) — greedy rows identical across 2 reruns, sampled rows
   differ; bench_quick.sh 'post-sampling' row for the log.
9. Gate 5 (modality preservation): probe_vision.py + probe_codegen.py on
   qwen36 (both STRONG baseline); any regression blocks the patch.
10. Capture: `git diff` in components/sglang →
    `patches/009-mlx-sampling.patch`; confirm a fresh setup.sh run prints
    '✓ 009-mlx-sampling.patch'; verify replay on a pristine v0.5.15.post1
    checkout (`git apply --check`); write the patches/README.md row (site
    map + logprob-norm + param mapping + greedy/top_k==1 aliasing note),
    benchmarks/sampling-ab receipt, README Known-Issues update, tick the
    queue item.

## Baseline & instrument

Pre-change, on `bash scripts/launch.sh qwen36` (port 23334): (1)
`python scripts/eval/probe_thinking.py --port 23334` — record exit code,
finish_reason, usage; (2) greedy decode tok/s at MR=1 from the server-log
'gen throughput' lines over a fixed 3-prompt, max_tokens=512 workload (save
the log excerpt); (3) `bash scripts/bench/bench_quick.sh
"pre-sampling-baseline"` row in benchmarks.log. Then the SAME three on the
loop-repro preset: qwen35 (`mlx-community/Qwen3.5-27B-4bit`) if it fits
under MEM_FRAC 0.7, else `qwen35-9b-8bit` as an explicitly-labeled RAM
proxy. HARD PRECONDITION for the loop-fix claim: the pre-change probe on the
loop-repro preset MUST show DEGRADED/exit 1 with an actual runaway/looping
`<think>` transcript (no `</think>`, finish_reason=length at max_tokens). If
a preset returns VERIFIED under greedy at baseline, the loop-fix claim is
untestable on it — record that and DROP that preset from success criterion
(i) rather than counting a trivially-passing probe as a win.

## Success criteria

- Greedy identity: temperature=0 (top_k==1) requests produce token-identical
  completions pre/post patch on the fixed 3-prompt set (receipt: saved
  completions in benchmarks/sampling-ab.md).
- Sampling live: temp=1.0 5-repeat run yields ≥4/5 distinct completions;
  unit sanity script passes (greedy rows == argmax, sampled tokens only from
  the unfiltered set).
- probe_thinking.py returns exit 0 (THINKING VERIFIED) on qwen36 with
  model-recommended sampling, AND on a preset whose BASELINE established the
  loop the known-loop case reaches `</think>` with finish_reason=stop within
  max_tokens — OR the documented null (kill criterion 3) with transcripts.
- Perf: greedy-path decode tok/s within ±3% of baseline and sampled-path
  cost quantified, both from server-log gen throughput at MR=1.
- Batched correctness: MR=4 mixed-params batch — greedy rows reproducible
  across reruns, sampled rows vary, no cross-row token bleed.
- Modality gate: probe_vision.py + probe_codegen.py outcomes unchanged from
  the current probe matrix on the tested preset.
- patches/009-mlx-sampling.patch applies clean on pristine v0.5.15.post1; a
  fresh setup.sh prints '✓ 009-mlx-sampling.patch'; patches/README.md +
  README.md updated.

## Kill criteria

- Greedy regression unfixable: if post-patch top_k==1 requests are not
  token-identical to baseline after one debugging pass, revert the offending
  sites and record which site diverged (likely a logprob-normalization or
  lazy-eval-graph change leaking into the greedy branch).
- If heterogeneous per-row sampling in decode_batch_start cannot be made
  correct without breaking the mx.async_eval lazy contract, ship
  homogeneous-batch + MR=1 sampling only (covers the single-user mission)
  and record the batched limitation in patches/README.md.
- If the infinite-`<think>` loop persists under model-recommended sampling
  on a preset whose BASELINE established the loop, STOP probing: record the
  null (sampling is not the loop's root cause; strengthens the
  DeltaNet/router-quant hypothesis feeding M4-E), keep the sampling patch
  anyway (fleet-parity infrastructure regardless), and update Known Issues.
- If sampled-path overhead exceeds 25% decode tok/s at MR=1 after trying the
  batched-sampler and precomputed-closure variants, land greedy-default +
  document the cost; do not micro-optimize past 1 day.

## Deliverables

- `patches/009-mlx-sampling.patch` — per-site replacement of the argmax
  sites + _select_tokens/_sampling registry in model_runner.py + tp_worker
  plumbing + mx.random.seed init.
- patches/README.md new row for 009 documenting the live site map, the
  logprob-normalization requirement, the SGLang→make_sampler param mapping,
  AND the greedy/top_k==1 aliasing limitation (by design, not a bug).
- scripts/eval/probe_thinking.py extended with --temperature/--top-p flags
  (default unchanged: 0).
- benchmarks/sampling-ab.md receipt: baseline vs post-patch server-log
  gen-throughput excerpts (greedy MR=1, sampled MR=1, mixed MR=4), probe
  outputs for qwen36 + the loop-repro preset (with the baseline loop
  transcript that establishes the DEGRADED precondition), 5-repeat
  divergence check, greedy token-identity check.
- README.md updates: Known Issues greedy-only entry updated with the probe
  verdict (loop fixed, or loop persists → checkpoint-side, pointing at
  M4-E); queue item ticked.
- Unit-level sanity script scripts/test/test_sampling_select.py: synthetic
  (B,V) logits through _select_tokens for greedy/homogeneous/heterogeneous
  cases; asserts greedy rows == argmax and sampled rows land only on
  non-filtered tokens.

## Constraints

- All edits go into components/sglang on the Mac and MUST be captured as
  patches/009-mlx-sampling.patch replayed on pristine v0.5.15.post1; update
  patches/README.md.
- SGLANG_USE_MLX=1 for all serving; MEM_FRAC is a fraction of TOTAL system
  RAM, default 0.7 — do not raise (0.85 has hard-locked the box).
- macOS has no OOM killer: run `bash scripts/common/oom_guard.sh &` before
  any ≥32K-context or multi-request bench.
- One mechanism at a time: the sampling patch changes ONLY token selection;
  do not bundle bench-flag fixes, template fixes, or other work into 009.
- Greedy must stay bit-stable: any request that normalizes to top_k==1 must
  traverse a pure mx.argmax path identical to today's graph.
- Preserve thinking+image+video modalities: after the patch, re-run
  probe_vision.py and probe_video.py (or probe_all.sh) on a VLM preset —
  sampling touches every decode path including the VLM prefill sites.
- Decode tok/s claims from server-log gen throughput only (never client
  TPOT); validate behavior not exit status — read probe transcripts.
- Detach any >30min server/bench run via setsid; negative probe outcomes are
  findings with receipts.
- Coordinate with the long-context item: both need the box's only GPU; do
  not interleave bisect arms and sampling A/B in the same server session.

## Risks

- mx.compile decorators on apply_top_p/min_p/top_k capture mx.random state;
  per-shape recompiles on varying batch sizes could add first-call latency
  spikes — measure, don't assume.
- The sampler returns tokens from a filtered LOGPROB tensor; feeding raw
  logits silently yields wrong top-p nuclei (not an error) — the unit sanity
  script is the guard.
- Loop-repro preset selection: qwen35-9b-8bit may not actually loop (8-bit
  precision), making it a weak disambiguator; the baseline DEGRADED
  precondition + preferring the INT4 qwen35 looper mitigates counting a
  non-loop as a fixed loop.
- The patch edits upstream-origin lines, increasing future rebase conflict
  surface — record the site map in patches/README.md for the next
  version-rebase gate.
- Sampled thinking traces are longer and non-deterministic: probe
  max_tokens may truncate on 27B-class verbosity; bump per-run max_tokens
  rather than declaring DEGRADED on length alone.
- mlx-lm is unpinned in the sglang extras — a venv rebuild could land a
  changed sample_utils surface; import defensively and record the installed
  version in the receipt.
