# M4-A: Long-context — measure the current stack, then bisect the 128K→32K regression only if it persists

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | Phase 0 (current-stack measurement): hours. Full bisect if triggered: ~2 days elapsed (arm venv builds ~1h each; growth-rate probes minutes; conditional 128K replays ~15-30 min each incl. boot) |
| **GPU time** | none (no CUDA/ROCm); M4 unified-memory box exclusively occupied during measurement windows |
| **Depends on** | Disk headroom for arms (~25 GB each — run the queue's disk-triage item first).; External: pre-regression mlx/mlx-lm wheel availability on PyPI (the pip wheel cache is empty and no freeze of the old venv exists — PyPI upload-date recovery is the only lib-recovery path); mlx-vlm 0.4.4 from PyPI.; External: exclusive use of the M4 box during measurement windows (no evals/opencode/serving concurrently). |
| **Provides to** | M4 queue: either a recovered long-context ceiling on the current stack (closes the item without arms) or a lib/SGLang attribution for the regression.; 'Implement real sampling' and the 256K roadmap — a recovered 128K ceiling changes what contexts sampling/eval work should target.; Fleet: 3090/R9700 READMEs — a confirmed 'unpinned transitive lib silently regressed a mission metric across a venv rebuild' finding generalizes to their setup scripts; share via sister-repo README. |

## Objective

The rig's mission-defining 256K goal is blocked at a ~32K prefill ceiling. The
ceiling was measured on an earlier stack pin (v0.5.12-era), where a 128K-valid
recipe regressed to ~32K with the cause unisolated between an SGLang rebase
and MLX libraries silently floated during a same-day venv rebuild. The current
stack (v0.5.15.post1 + floated MLX libs) has NOT been measured. Phase 0
measures it; only if the regression signature persists does the bisect run.
Either outcome wins: the ceiling is re-derived on the current stack, or the
regression is attributed (recover 4× context by pinning the responsible lib,
or confirm it is SGLang-side).

## Hypothesis

The regression rode in with floated MLX library versions (mlx/mlx-lm are
UNPINNED in sglang `[srt_mps]` across the relevant tags; mlx-vlm is installed
by hand), not with SGLang-side code. Falsifiable prediction: old-SGLang +
current libs (Arm A) shows the ~0.19-0.22 MB/token prefill growth signature
and OOMs by ~40K, while newer-SGLang + old libs (Arm B) shows <0.08 MB/token
and completes a 64K prefill. The opposite pattern attributes it to the SGLang
side. A clean Phase 0 (current stack <0.08 MB/token) makes the whole question
moot — the regression is already fixed.

## Background & receipts

- 128K validation receipt: commit 8957971: qwen36, `--kv-cache-dtype
  turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5`,
  context-length 140000, prefill ~6.5 min, decode 0.10 tok/s from server-log
  gen-throughput, KV pool 322,430 slots @ 23040 B/slot; OOM guard 'never
  warned below 10 GB free' — incompatible with a 0.2 MB/token growth rate
  (128K × 0.2 MB = ~26 GB), so that stack demonstrably lacked the growth
  signature.
- The 128K-validation tree had ONLY patches 002-008 on v0.5.11 (commit
  612785ffd): `git ls-tree 8957971 patches/` shows 7 patch files. The correct
  'historical recipe' arm is a worktree at 8957971, not the last pre-rebase
  HEAD (8b877e2, 19 patches).
- Regression receipt:
  `evals/swebench/runs/qwen36-long-context-64k-OOM-2026-05-21/README.md` —
  the identical recipe on the v0.5.12 stack OOM-killed at 64K (guard fired at
  free=6.64 GB after ~40K of 64K tokens prefetched); 256/4K/16K rows
  completed. bench-results.txt archived in the same dir.
- Growth signature receipt:
  `evals/swebench/runs/qwen36-long-context-formula-2026-05-21/data.txt` —
  0.19-0.22 MB/token across three measurements; ceiling formula
  max_prefill_tokens ≈ (post_boot_free_GB − 0.5) × 5120; ceiling is
  environment-dependent (background apps ate ~2.7 GB).
- Confound mechanism verified on both sides of the interface: sglang v0.5.11
  AND v0.5.12 `python/pyproject_other.toml` [srt_mps] extras list `mlx` and
  `mlx-lm` UNPINNED (torch pinned) — fetched both tags and diffed; identical.
  The venv rebuild that coincided with the v0.5.12 rebase (commit 0f563fc)
  therefore floated mlx/mlx-lm to that day's latest. mlx-vlm was hand-bumped
  0.4.4→0.5.0 after the 128K validation but before that rebase, so it must be
  bisected separately within the lib arm.
- No freeze of the old venv exists and the pip wheel cache is empty — lib
  recovery is PyPI-upload-date selection only (Method step 2b). The
  v0.5.12-era patched TREE survives at `components/sglang.v0512.bak`.
- Current stack for reference: SGLang v0.5.15.post1 + 6 patches
  (002/003/005/007/008/014, fail-loud replay verified byte-identical against
  the pristine tag); libs mlx 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.5, torch
  2.11.0, transformers 5.12.1. The qwen36 preset enables the radix cache —
  Phase 0 and every arm must pass `EXTRA_ARGS="--disable-radix-cache"` to
  match the historical recipe.
- Known apply hazard for the old series: the v0.5.11-era
  013-mlx-vlm-pixel-values.patch has a malformed hunk header — irrelevant for
  the 8957971 arm (only 002-008 exist there) but disqualifies naive replay of
  the full 19-patch series at 8b877e2 without `git apply --recount --3way`.
- Arms are self-containable without touching the main checkout:
  scripts/common.sh at both HEAD and 8957971 derives VENV_DIR=$REPO_DIR/.venv
  and SGLANG_DIR=$REPO_DIR/components/sglang from the script's own location,
  so a `git worktree` checkout carries its own venv + sglang clone (the old
  setup.sh does a full clone at its own pin); launch.sh at 8957971 already
  has the qwen36 preset, EXTRA_ARGS pass-through, MEM_FRAC env and
  --kv-cache CLI.
- Instruments verified on disk: `scripts/common/mem_profile.sh` (CSV:
  ts,free_gb,inactive_gb,wired_gb,sglang_rss_gb,pid; PROF_INTERVAL default
  5s); `scripts/common/oom_guard.sh` (kill<8 GB, 2s interval);
  `scripts/bench/bench_long_context.py` (--port/--contexts/--output-tokens,
  urllib timeout=3600, reports server-verified usage.prompt_tokens as 'in=').
- Depth-honesty caveat: every replay must log usage.prompt_tokens per fleet
  invariant (an early prompt generator under-delivered ~1.7×; benchmarks
  labeled '128K' can be ~76K actual).
- Non-suspect: the 64K dense-SDPA ceiling is structural and predates the
  regression — qwen36 (MoE+DeltaNet hybrid) is the only valid probe model;
  dense presets cannot discriminate the arms.

## Method

1. Pre-flight (no server): record current-stack state into
   `benchmarks/longctx-bisect/phase0/`: `.venv/bin/pip freeze >
   pip-freeze.txt`, `git -C components/sglang log -1 --oneline`, `vm_stat >
   vm-stat-idle.txt`. Close background apps; target ≥12 GB free (the formula
   receipt shows ~2.7 GB of avoidable background load moves the ceiling).
2. Recover old lib versions (no server): PyPI JSON API (`curl -s
   https://pypi.org/pypi/<pkg>/json`) — pick the newest release with
   upload_time before the regression-era venv rebuild for mlx and mlx-lm, and
   mlx-vlm==0.4.4 vs 0.5.0 explicitly. Record chosen versions + evidence in
   `benchmarks/longctx-bisect/lib-recovery.md`. If no installable arm64
   wheels → Arm B blocked, see kill criteria.
3. **Phase 0 — current-stack measurement (the go/no-go):** start `bash
   scripts/common/oom_guard.sh &` then `PROF_INTERVAL=2 bash
   scripts/common/mem_profile.sh > benchmarks/longctx-bisect/phase0/mem_profile.csv &`.
   Launch: `setsid env CTX=140000 CHUNKED=2048 MEM_FRAC=0.5
   EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh qwen36
   --kv-cache turboquant > /tmp/phase0-boot.txt 2>&1 &` (the historical 128K
   recipe). After /health OK, record `vm_stat` post-boot free. Run `python
   scripts/bench/bench_long_context.py --port 23334 --contexts 4096 32768
   --output-tokens 32 | tee benchmarks/longctx-bisect/phase0/bench-results.txt`.
4. Compute the growth rate: from mem_profile.csv, MB/token = ((free+inactive
   at 32K-request start) − (free+inactive at prefill end)) × 1024 /
   usage.prompt_tokens. **Decision:** <0.08 MB/token → probe 64K, then the
   full 128K replay (prompt_tokens ≥125,000); if that completes, the
   regression is FIXED on the current stack — write ATTRIBUTION.md as
   'resolved by stack move', update the README/CLAUDE.md ceiling, close the
   item, build no arms. ≥0.15 MB/token → the regression persists; proceed to
   arms. 0.08-0.15 → re-run same-day before concluding.
5. Build Arm A (code axis: old SGLang+patches; libs: current): `git worktree
   add ../m4-armA-8957971 8957971 && cd ../m4-armA-8957971 &&
   ./scripts/setup.sh` (self-contained: own .venv + sglang clone pinned
   v0.5.11/612785ffd, patches 002-008). HARD GATE: the patch loop must report
   0 failed; on any failure use `git -C components/sglang apply --recount
   --3way <patch>` and re-verify. Then `pip install mlx-vlm` (that setup.sh
   omits it) and `pip freeze > .../armA/pip-freeze.txt`; confirm
   mlx/mlx-lm/mlx-vlm versions EQUAL phase0's freeze (that is the arm's
   definition).
6. Arm A measure (same commands as steps 3-4 from the worktree, guard +
   profiler running, receipts to .../armA/). Decision gates: growth <0.08
   MB/token at 32K → proceed to 64K probe, and if that completes, full replay
   `--contexts 131072` targeting prompt_tokens ≥125,000 and prefill wall
   ~6.5-10 min. Growth ≥0.15 → Arm A reproduces the regression with old code,
   implicating libs. If the v0.5.11-era code cannot boot against current
   mlx/mlx-lm (API drift), record the traceback as a finding and skip to Arm
   B — attribution then rests on Arm B plus Arm C.
7. Build Arm B (lib axis: old libs; code: pre-rebase-current): `git worktree
   add ../m4-armB-v0512 d414b44 && cd ../m4-armB-v0512 && ./scripts/setup.sh`
   (v0.5.12 + 14 per-feature patches; same 0-failed gate), then downgrade ONE
   mechanism at a time: first `pip install --force-reinstall mlx==<old>`
   only; measure (step 8); only if growth unchanged, additionally downgrade
   `mlx-lm==<old>`; measure; then `mlx-vlm==0.4.4`; measure. Use `--no-deps`
   only if the resolver drags unrelated packages. `pip freeze` receipt after
   every change.
8. Arm B measure after each downgrade: 32K growth-rate probe (same
   instruments, receipts to .../armB-<libset>/). Any sub-arm reaching <0.08
   MB/token → confirm with 64K, then full 128K replay. That lib is the
   culprit; note whether it was mlx core, mlx-lm, or mlx-vlm.
9. Decision tree: [Arm A high + Arm B(some libset) low] → LIB-ATTRIBUTED:
   write the pin (exact `pip install` lines after the `.[srt_mps]` install)
   as a proposed setup.sh patch; verify the pin ALSO recovers 128K on the
   CURRENT v0.5.15.post1 stack before shipping it. [Arm A low + Arm B high
   through all three downgrades] → SGLANG-SIDE: sub-bisect code by building
   Arm D worktree at 8b877e2 (v0.5.11 + all 19 patches, needs --recount for
   patch 013) to separate base-rebase from patches 009-020. [Both high] → run
   control Arm C: worktree at 8957971 + old libs (full historical state); if
   Arm C is ALSO high → the historical 128K claim is not reproducible in
   today's environment — record a null with the formula receipt as the
   explanatory frame; if Arm C is low → interaction effect, bisect lib×code
   pairs. [Both low] → re-run Phase 0 same-day to check reproducibility
   before concluding anything.
10. Write `benchmarks/longctx-bisect/ATTRIBUTION.md`: per-arm table (stack,
    lib versions, post-boot free, prompt_tokens, MB/token, max completed
    prefill), the decision-tree outcome, and the recommended action. Draft
    the README queue update + CLAUDE.md 'Long-context launch flags'
    correction. Do NOT change the main checkout's setup.sh or venv until
    ATTRIBUTION.md exists.

## Baseline & instrument

Phase 0 on the current stack (repo HEAD, existing .venv untouched): per-token
prefill memory growth at 32K, computed from mem_profile.sh CSV
(free+inactive delta over the prefill window) divided by
bench_long_context.py's server-verified usage.prompt_tokens.

## Success criteria

- Growth rate (MB/token at 32K, server-verified prompt_tokens) measured with
  archived mem_profile.csv + bench-results.txt + pip-freeze.txt + post-boot
  vm_stat for: Phase 0, and (if triggered) Arm A and every Arm B lib-set
  (thresholds: <0.08 = clean, ≥0.15 = regressed; 0.08-0.15 = inconclusive,
  re-run).
- Either: Phase 0 clean + 128K replay passes on the current stack
  (prompt_tokens ≥125,000, prefill completes without guard kill) and the
  README/CLAUDE.md ceiling is corrected; or the regression is attributed to
  exactly one axis — a named MLX package+version, or SGLang-side (base rebase
  vs patches via Arm D) — or an explicit environment-dependence null backed
  by control Arm C; written to ATTRIBUTION.md.
- If lib-attributed: a setup.sh pin patch is drafted and the pin verified to
  recover 128K on the current stack.

## Kill criteria

- Any hard system stall requiring reboot (macOS has no OOM killer) → all
  remaining probes capped at 32K growth-rate only; a second stall ends the
  experiment with whatever attribution the growth-rate data supports.
- Old-lib recovery fails (no installable pre-regression arm64 wheels on PyPI)
  → Arm B is blocked: record the null, run the code-side arms only, and state
  that lib attribution is permanently unavailable.
- Both Arm A and Arm B boot-fail after ~2h debugging each → the 2x2 is
  unconstructible; record the incompatibility null — which itself decides
  're-derive long-context on the current stack'.
- Both arms high-growth AND control Arm C high-growth → the historical 128K
  claim is not reproducible as stated; record a null with the
  environment-formula receipt as the explanation and close the queue item as
  'regression = environment + unpinned history'.
- >2 elapsed days or arms exhausted without meeting a success criterion →
  write up partial receipts as a negative result.

## Deliverables

- `benchmarks/longctx-bisect/{phase0,armA,armB-<libset>,armC,armD as
  reached}/` each with README.md, mem_profile.csv, bench-results.txt,
  pip-freeze.txt, vm-stat-postboot.txt, and a server-log tail as .txt (never
  .log — gitignored; receipts have been lost to this).
- `benchmarks/longctx-bisect/lib-recovery.md` — recovered mlx/mlx-lm/mlx-vlm
  versions with PyPI-upload-time evidence.
- `benchmarks/longctx-bisect/ATTRIBUTION.md` — per-arm results table,
  decision-tree outcome, recommendation.
- If lib-attributed: proposed setup.sh patch pinning mlx/mlx-lm alongside the
  existing mlx-vlm install line.
- README queue + CLAUDE.md 'Long-context launch flags' corrections.

## Constraints

- oom_guard.sh running BEFORE any ≥32K request (CLAUDE.md mandatory;
  defaults are post-incident-tuned — do not loosen); mem_profile.sh running
  for every measured request.
- MEM_FRAC is a fraction of TOTAL 64 GB system RAM; never above 0.7 (0.85
  has hard-locked the box); all arms use the historical 0.5.
- One mechanism per arm: each measurement changes exactly one axis (code tree
  OR one lib) vs its anchor, with pip-freeze receipts proving it; A/B at
  short+deep (4K + 32K, then 64K/128K only when growth rate × tokens
  predicts fit within post-boot free − 2 GB).
- Depth honesty: every row records server-verified usage.prompt_tokens
  ('in='), never the label; decode numbers only from server-log
  gen-throughput.
- Environment control: close background apps before every boot, record
  post-boot free via vm_stat, and compare arms primarily on MB/token
  (environment-normalized) rather than raw OOM depth.
- Arm builds only in git worktrees with their own VENV_DIR/SGLANG_DIR; the
  main checkout's .venv, components/sglang, and setup.sh stay untouched until
  ATTRIBUTION.md exists.
- Patch application in every arm hard-gates on '0 failed' — a skipped patch
  invalidates the arm.
- Detach server + any >30 min bench via setsid; receipts as .txt/.csv/.md
  only; negative results are findings and get receipt dirs too.

## Risks

- API drift: v0.5.11-era MLX-backend code may not import against current
  mlx-lm — Arm A could be unbuildable, weakening attribution to Arm B + Arm C
  only (handled in decision tree).
- Old-lib arm may under-determine: if the culprit is a transitive dep rather
  than mlx/mlx-lm/mlx-vlm, both arms read high-growth and the experiment
  lands on the interaction/null branch; full pip-freeze diffs enable a
  follow-up dep-level bisect.
- The 128K receipt is server-log-based (no archived in= counts); if its
  '128K tokens' was optimistic, the recovery target could be lower — Arm C
  re-establishes the true historical number.
- macOS stall risk on every ≥64K probe despite the guard — mitigated by
  32K-first protocol and formula-predicted fit checks; a reboot loss of a day
  is possible.
- Worktree arms double model/venv disk usage (~25 GB per arm; model weights
  shared via HF cache) — clear disk first (queue's disk-triage item).
