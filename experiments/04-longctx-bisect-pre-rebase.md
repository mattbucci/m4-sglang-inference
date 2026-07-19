# M4-A: Bisect the 128K→32K long-context regression BEFORE any rebase

> **Post-rebase deltas (2026-07-19) — read before Method step 1.** The
> v0.5.15.post1 rebase landed and validated the day after this spec was
> written; "before any rebase" and the stay-vs-rebase framing are moot (we
> stayed rebased). What survives, what changed:
>
> - **NEW STEP 0 (replaces the step-3 baseline):** measure the CURRENT
>   v0.5.15.post1 stack's 32K growth rate with the same instruments
>   (oom_guard + mem_profile + bench_long_context). If growth <0.08 MB/token
>   and a 64K/128K probe completes, **the regression is gone — close this
>   item as "fixed by rebase/lib-float", re-derive the long-context ceiling,
>   and skip all arms.** Only if the signature (≥0.15 MB/token) persists does
>   the bisect proceed as specced. Note qwen36's preset now enables the radix
>   cache + normal event loop; add `EXTRA_ARGS="--disable-radix-cache"` to
>   match the historical recipe (the spec's step-3 command already does).
> - Arms are UNAFFECTED: they build in git worktrees at old repo commits
>   whose own setup.sh does full clones at old pins — independent of the
>   main checkout's new stack.
> - The old v0.5.12 venv was rebuilt in place with **no pip freeze captured**
>   and the pip wheel cache is empty — old-lib recovery is PyPI-upload-date
>   only (step 2b). The old patched TREE survives at
>   `components/sglang.v0512.bak`.
> - Current-stack lib versions for reference: mlx 0.32.0, mlx-lm 0.31.3,
>   mlx-vlm 0.6.5, torch 2.11.0, transformers 5.12.1.
> - Disk: only ~12 GiB free — clear space (see queue's disk-triage item)
>   before building arms (~25 GB each).

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | ~2 days elapsed (arm venv builds ~1h each; growth-rate probes minutes; conditional 128K replays ~15-30 min each incl. boot) |
| **GPU time** | none (no CUDA/ROCm); M4 unified-memory box exclusively occupied ~4-6h of active measurement windows |
| **Depends on** | External: pre-2026-05-12 mlx/mlx-lm wheel availability (M4 pip cache at ~/Library/Caches/pip or PyPI); mlx-vlm 0.4.4 from PyPI.; External: exclusive use of the M4 box during measurement windows (no evals/opencode/serving concurrently).; No repo prerequisite — this is item 1 of the M4 fleet-audit queue and intentionally precedes any rebase or sampling work. |
| **Provides to** | M4 queue: the stay-vs-rebase decision gating all future version bumps (README fleet-audit bullets 1 and 6) and the port of replay/probe gates into setup.sh (pin list is an input).; M4 queue: 'Implement real sampling' and the 256K roadmap — a recovered 128K ceiling changes what contexts sampling/eval work should target.; Fleet: 3090/R9700 READMEs — a confirmed 'unpinned transitive lib silently regressed a mission metric across a venv rebuild' finding generalizes to their setup scripts; share via sister-repo README per reference-sister-teams. |

## Objective

The rig's mission-defining 256K goal is blocked at a ~32K prefill ceiling that appeared between 2026-05-11 (128K validated on v0.5.11) and 2026-05-21 (64K OOMs on v0.5.12), with the cause unisolated between the SGLang v0.5.12 rebase and MLX libraries silently floated during the same-day venv rebuild. Rebasing forward now buries the evidence; either bisect outcome wins — recover 4x context by pinning the responsible lib, or attribute it to SGLang-side changes and make stay-vs-rebase a decidable question. This item gates every other M4 recovery-queue action (README fleet-audit queue, bullet 1).


## Hypothesis

The regression rode in with the MLX library versions floated during the 2026-05-20 venv rebuild (mlx/mlx-lm are UNPINNED in sglang [srt_mps] at both v0.5.11 and v0.5.12; mlx-vlm was hand-bumped 0.4.4→0.5.0 on ~2026-05-16 and is absent from setup.sh entirely), not with SGLang v0.5.12 code. Falsifiable prediction: v0.5.12 + 14 current patches + old libs (Arm B) shows per-token prefill memory growth <0.08 MB/token and completes a 64K prefill, while the 8957971 tree (v0.5.11 + patches 002-008) + current libs (Arm A) shows the ~0.19-0.22 MB/token signature and OOMs by ~40K. The opposite pattern attributes it to the SGLang side.


## Background & receipts

- 128K validation receipt: commit 8957971 (2026-05-11 10:45): qwen36, --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5, context-length 140000, prefill ~6.5 min, decode 0.10 tok/s from server-log gen-throughput, KV pool 322,430 slots @ 23040 B/slot; OOM guard 'never warned below 10 GB free' — incompatible with a 0.2 MB/token growth rate (128K x 0.2 MB = ~26 GB), so the old stack demonstrably lacked today's growth signature.
- The 128K-validation tree had ONLY patches 002-008 on v0.5.11 (commit 612785ffd): `git ls-tree 8957971 patches/` shows 7 patch files; patches 009-020 all landed later (2026-05-12..05-19), still on v0.5.11. The correct 'historical recipe' arm is a worktree at 8957971, not the last pre-rebase HEAD (8b877e2, 19 patches).
- Regression receipt: evals/swebench/runs/qwen36-long-context-64k-OOM-2026-05-21/README.md — the identical recipe on the v0.5.12 stack OOM-killed at 64K (guard fired at free=6.64 GB after ~40K of 64K tokens prefetched); 256/4K/16K rows completed. bench-results.txt archived in the same dir.
- Growth signature receipt: evals/swebench/runs/qwen36-long-context-formula-2026-05-21/data.txt — 0.19-0.22 MB/token across three measurements; ceiling formula max_prefill_tokens ≈ (post_boot_free_GB - 0.5) × 5120; ceiling is environment-dependent (background apps ate ~2.7 GB). CLAUDE.md states 0.15-0.2 MB/token, mechanism unknown.
- Confound mechanism verified on both sides of the interface: sglang v0.5.11 AND v0.5.12 `python/pyproject_other.toml` [srt_mps] extras list `mlx` and `mlx-lm` UNPINNED (torch==2.11.0 pinned) — fetched both tags from GitHub and diffed; identical. The 2026-05-20 user-directed venv rebuild (commit 0f563fc, 'rebuild a new venv') therefore floated mlx/mlx-lm to that day's latest. mlx-vlm appears in NO install line in scripts/setup.sh (grep verified) — it was hand-installed, bumped 0.4.4→0.5.0 ~2026-05-16 (README.md line 85, line 151), i.e. AFTER the May-11 128K validation but BEFORE the rebase, so it must be bisected separately within the lib arm.
- Current pins: scripts/setup.sh line 24 SGLANG_COMMIT="v0.5.12" (= commit 127b9e328 per patches/README.md header); 14 per-feature patch files 002-016 applied by a check-apply-else-skip loop (setup.sh lines 77-93) that only WARNS on failure — the same silent-skip mechanism patch-013 forensics blames for a month of fabricated VLM output (patches/README.md line 90). Arm builds must hard-gate on 0 failed.
- Known apply hazard for the old series: the v0.5.11-era 013-mlx-vlm-pixel-values.patch has a malformed hunk header ('couldn't apply, used stash-copy' — evals/swebench/runs/qwen36-v0512-regression-PASS-2026-05-20/README.md). Irrelevant for the 8957971 arm (only 002-008 exist there) but disqualifies naive replay of the full 19-patch series at 8b877e2 without `git apply --recount --3way`.
- Arms are self-containable without touching the main checkout: scripts/common.sh at both HEAD and 8957971 derives VENV_DIR=$REPO_DIR/.venv and SGLANG_DIR=$REPO_DIR/components/sglang from the script's own location, so a `git worktree` checkout carries its own venv + sglang clone; launch.sh at 8957971 already has the qwen36 preset, EXTRA_ARGS pass-through, MEM_FRAC env and --kv-cache CLI.
- Instruments verified on disk: scripts/common/mem_profile.sh (CSV: ts,free_gb,inactive_gb,wired_gb,sglang_rss_gb,pid; PROF_INTERVAL default 5s); scripts/common/oom_guard.sh (kill<8 GB, 2s interval, pgrep anchored on 'python.*-m sglang\.launch_server'); scripts/bench/bench_long_context.py (--port/--contexts/--output-tokens, urllib timeout=3600, reports server-verified usage.prompt_tokens as 'in=').
- Depth-honesty caveat: the pre-2026-04-20 prompt generator under-delivered ~1.7x (benchmarks/qwen36-35b-a3b-4bit/results.json '128K' row = 76,262 actual input tokens); fix landed 7d9945b (2026-04-20) BEFORE the May-11 validation, but every replay must still log usage.prompt_tokens per fleet invariant.
- Non-suspect: the 64K dense-SDPA ceiling is structural and predates the regression (commit 822f011, 2026-05-12; README 'Active work' #4) — qwen36 (MoE+DeltaNet hybrid) is the only valid probe model; dense presets cannot discriminate the arms.


## Method

1. Pre-flight (no server): from /Users/letsrtfm/AI/m4-sglang-inference record current-stack state into benchmarks/longctx-bisect-2026-07/baseline/: `.venv/bin/pip freeze > pip-freeze.txt` (expect mlx-lm 0.31.2 per README bullet 2), `git -C components/sglang log -1 --oneline` and `git -C components/sglang diff --stat | tail -1` (patch presence), `vm_stat > vm-stat-idle.txt`. Close Firefox/Spotify/Steam/etc; target ≥12 GB free (the 2026-05-21 formula receipt shows ~2.7 GB of avoidable background load moved the ceiling).
2. Recover old lib versions (no server): (a) `python3 -m pip cache list mlx; python3 -m pip cache list mlx-lm; python3 -m pip cache list mlx-vlm` — the user-level wheel cache (~/Library/Caches/pip) survives venv rebuilds; correlate cached wheel versions with the May-11 era. (b) Fallback: PyPI JSON API (`curl -s https://pypi.org/pypi/<pkg>/json`) — pick the newest release with upload_time < 2026-05-12 for mlx and mlx-lm (the venv predating the rebuild), and mlx-vlm==0.4.4 vs 0.5.0 explicitly (0.5.0 arrived ~05-16, post-128K-validation). Record chosen versions + evidence in benchmarks/longctx-bisect-2026-07/lib-recovery.md. If neither source yields installable versions → Arm B blocked, see kill criteria.
3. Baseline measurement (current stack, one axis: none changed): start `bash scripts/common/oom_guard.sh &` then `PROF_INTERVAL=2 bash scripts/common/mem_profile.sh > benchmarks/longctx-bisect-2026-07/baseline/mem_profile.csv &`. Launch: `setsid env CTX=140000 CHUNKED=2048 MEM_FRAC=0.5 EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh qwen36 --kv-cache turboquant > /tmp/baseline-boot.txt 2>&1 &` (exact May-11 recipe; launch.sh maps these to --chunked-prefill-size/--mem-fraction-static/--kv-cache-dtype). After /health OK, record `vm_stat` post-boot free. Run `python scripts/bench/bench_long_context.py --port 23334 --contexts 4096 32768 --output-tokens 32 | tee benchmarks/longctx-bisect-2026-07/baseline/bench-results.txt`.
4. Compute the growth rate: from mem_profile.csv, MB/token = ((free_gb+inactive_gb at 32K-request start) − (free_gb+inactive_gb at prefill end, i.e. when sglang_rss plateaus)) × 1024 / usage.prompt_tokens (the 'in=' value in bench-results.txt, ~31,458 for the 32K row). Expected ~0.19-0.22 MB/token replicating the 2026-05-21 formula receipt. If baseline does NOT reproduce ≥0.15 MB/token, STOP — the regression is not currently reproducible; investigate environment before burning arm effort.
5. Build Arm A (code axis: old SGLang+patches; libs: current): `git worktree add ../m4-armA-8957971 8957971 && cd ../m4-armA-8957971 && ./scripts/setup.sh` (self-contained: creates its own .venv + components/sglang pinned v0.5.11/612785ffd, applies patches 002-008). HARD GATE: the patch loop must report 0 failed (silent-skip is the patch-013 fabrication mechanism); on any failure use `git -C components/sglang apply --recount --3way <patch>` and re-verify. Then `pip install mlx-vlm` (setup.sh omits it) and `pip freeze > .../armA/pip-freeze.txt`; confirm mlx/mlx-lm/mlx-vlm versions EQUAL baseline's freeze (that is the arm's definition; torch must be 2.11.0 in both).
6. Arm A measure (same commands as step 3-4 from the worktree, guard+profiler running, receipts to .../armA/). Decision gates: growth <0.08 MB/token at 32K → proceed to 64K probe (`--contexts 65536`), and if that completes, full replay `--contexts 131072 --output-tokens 32` targeting server-verified prompt_tokens ≥125,000 and prefill wall ~6.5-10 min. Growth ≥0.15 MB/token → Arm A reproduces the regression with old code, implicating libs. If the v0.5.11-era code cannot boot against 2026-07 mlx/mlx-lm (API drift), record the traceback as a finding and skip to Arm B — attribution then rests on Arm B alone plus Arm C.
7. Build Arm B (lib axis: old libs; code: current): `git worktree add ../m4-armB-v0512 d414b44 && cd ../m4-armB-v0512 && ./scripts/setup.sh` (v0.5.12 + 14 per-feature patches; same 0-failed gate), then downgrade ONE mechanism at a time per fleet invariant: first `pip install --force-reinstall mlx==<old>` only; measure (step 8); only if growth unchanged, additionally downgrade `mlx-lm==<old>`; measure; then `mlx-vlm==0.4.4`; measure. Use `--no-deps` only if the resolver drags unrelated packages. `pip freeze` receipt after every change.
8. Arm B measure after each downgrade: 32K growth-rate probe (same instruments, receipts to .../armB-<libset>/). Any sub-arm reaching <0.08 MB/token → confirm with 64K, then full 128K replay (CTX=140000 recipe, prompt_tokens ≥125,000, decode from server-log gen-throughput lines in /tmp/…boot.txt, tail copied into the receipt dir as .txt — *.log is gitignored). That lib is the culprit; note whether it was mlx core, mlx-lm, or mlx-vlm.
9. Decision tree: [Arm A high + Arm B(some libset) low] → LIB-ATTRIBUTED: write the pin (exact `pip install` lines after the `.[srt_mps]` install) as a proposed setup.sh patch; 128K recovered. [Arm A low + Arm B high through all three downgrades] → SGLANG-SIDE: sub-bisect code by building Arm D worktree at 8b877e2 (v0.5.11 + all 19 patches, needs --recount for patch 013) to separate base-rebase (v0.5.11→v0.5.12) from patches 009-020; stay-vs-rebase becomes decidable. [Both high] → run control Arm C: worktree at 8957971 + old libs (full historical state); if Arm C is ALSO high → the May-11 claim is not reproducible in today's environment — record a null with the formula receipt as the explanatory frame; if Arm C is low → interaction effect, bisect lib×code pairs. [Both low] → re-run step 3 same-day to check baseline reproducibility before concluding anything.
10. Write benchmarks/longctx-bisect-2026-07/ATTRIBUTION.md: per-arm table (stack, lib versions, post-boot free, prompt_tokens, MB/token, max completed prefill), the decision-tree outcome, and the stay-vs-rebase recommendation. Draft README fleet-queue checkbox update + CLAUDE.md 'Long-context launch flags' correction. Leave commits to the normal repo flow; do NOT change the main checkout's setup.sh or venv until attribution is written down.


## Baseline & instrument

Current v0.5.12 stack (repo HEAD d414b44, existing .venv untouched): per-token prefill memory growth at 32K, computed from scripts/common/mem_profile.sh CSV (free+inactive delta over the prefill window) divided by bench_long_context.py's server-verified usage.prompt_tokens; must reproduce the 0.19-0.22 MB/token signature recorded in evals/swebench/runs/qwen36-long-context-formula-2026-05-21/data.txt before any arm is built.


## Success criteria

- Growth rate (MB/token at 32K, server-verified prompt_tokens) measured with archived mem_profile.csv + bench-results.txt + pip-freeze.txt + post-boot vm_stat for: baseline, Arm A, and every Arm B lib-set (thresholds: <0.08 = old-stack behavior, ≥0.15 = regressed; 0.08-0.15 = inconclusive, re-run).
- Regression attributed to exactly one axis — a named MLX package+version, or SGLang-side (base rebase vs patches 009-020 via Arm D) — or an explicit environment-dependence null backed by control Arm C; written to benchmarks/longctx-bisect-2026-07/ATTRIBUTION.md.
- If lib-attributed: 128K replay passes on v0.5.12 + 14 patches + pinned libs — server-verified prompt_tokens ≥125,000 at CTX=140000/turboquant/chunked-2048/mf-0.5, prefill completes without guard kill, decode taken from server-log gen-throughput — and a setup.sh pin patch is drafted.
- If SGLang-attributed: Arm D (8b877e2) result separates base-rebase from patches 009-020, making the README queue's stay-vs-rebase question decidable with receipts.
- README fleet-audit bullet 1 checkbox resolvable + CLAUDE.md 'Long-context launch flags' section correction drafted, citing the new receipt dirs.


## Kill criteria

- Baseline fails to reproduce ≥0.15 MB/token growth at 32K on the current stack → stop before building arms; the regression is environmental, not stack — record and redirect.
- Any hard system stall requiring reboot (macOS has no OOM killer) → all remaining probes capped at 32K growth-rate only; a second stall ends the experiment with whatever attribution the growth-rate data supports.
- Old-lib recovery fails (no pip-cache hit AND no installable pre-2026-05-12 arm64 wheels on PyPI) → Arm B is blocked: record the null, run the code-side arms only, and state that lib attribution is permanently unavailable.
- Both Arm A and Arm B boot-fail after ~2h debugging each (old-code×new-libs and new-code×old-libs API drift) → the 2x2 is unconstructible; record the incompatibility null — which itself decides 'rebase forward and re-derive long-context on the new stack'.
- Both arms high-growth AND control Arm C (full historical state) high-growth → the May-11 128K claim is not reproducible as stated; record a null with the environment-formula receipt as the explanation and close the queue bullet as 'regression = environment + unpinned history'.
- >2 elapsed days or arms exhausted without meeting a success criterion → write up partial receipts as a negative result; do not extend into the rebase work.


## Deliverables

- benchmarks/longctx-bisect-2026-07/{baseline,armA,armB-<libset>,armC,armD as reached}/ each with README.md, mem_profile.csv, bench-results.txt, pip-freeze.txt, vm-stat-postboot.txt, and a server-log tail as .txt (never .log — gitignored, cf. commit 555f7c5).
- benchmarks/longctx-bisect-2026-07/lib-recovery.md — recovered mlx/mlx-lm/mlx-vlm versions with pip-cache/PyPI-upload-time evidence.
- benchmarks/longctx-bisect-2026-07/ATTRIBUTION.md — per-arm results table, decision-tree outcome, stay-vs-rebase recommendation.
- If lib-attributed: proposed patch to scripts/setup.sh pinning mlx/mlx-lm and adding the missing mlx-vlm install line (gap verified: no mlx-vlm install anywhere in setup.sh).
- Drafted README fleet-audit-queue bullet-1 resolution + CLAUDE.md 'Long-context launch flags — current state' correction.


## Constraints

- oom_guard.sh running BEFORE any ≥32K request (CLAUDE.md mandatory; defaults kill<8 GB / 2s interval are post-incident-tuned — do not loosen); mem_profile.sh running for every measured request.
- MEM_FRAC is a fraction of TOTAL 64 GB system RAM; never above 0.7 (0.85 hard-locked the box 2026-05-14); all arms use the historical 0.5.
- One mechanism per arm: each measurement changes exactly one axis (code tree OR one lib) vs its anchor, with pip-freeze receipts proving it; A/B at short+deep (4K + 32K, then 64K/128K only when the growth rate × tokens predicts fit within post-boot free − 2 GB).
- Depth honesty: every row records bench_long_context.py's server-verified usage.prompt_tokens ('in='), never the label; decode numbers only from server-log gen-throughput.
- Environment control: close background apps before every boot, record post-boot free via vm_stat, and compare arms primarily on MB/token (environment-normalized) rather than raw OOM depth — the 2026-05-21 formula receipt proves raw ceilings shift ±10K with desktop load.
- Arm builds only in git worktrees with their own VENV_DIR/SGLANG_DIR (common.sh derives both from repo root); the main checkout's .venv, components/sglang, and setup.sh stay untouched until ATTRIBUTION.md exists.
- Patch application in every arm hard-gates on '0 failed' — a skipped patch invalidates the arm (patch-013 silent-skip precedent).
- Detach server + any >30 min bench via setsid; receipts as .txt/.csv/.md only (*.log is gitignored and receipts have been lost to this before); negative results are findings and get receipt dirs too.


## Risks

- API drift: v0.5.11-era MLX-backend code may not import against 2026-07 mlx-lm (2 months of upstream churn) — Arm A could be unbuildable, weakening attribution to Arm B + Arm C only (handled in decision tree).
- Old-lib arm may under-determine: if the culprit is a transitive dep (e.g. transformers pinned differently by runtime_common between tags) rather than mlx/mlx-lm/mlx-vlm, both arms read high-growth and the experiment lands on the interaction/null branch; full pip-freeze diffs are captured to enable a follow-up dep-level bisect.
- The May-11 receipt is server-log-based (no archived bench-results with in= counts); if its '128K tokens' was optimistic, the recovery target could be lower than 128K — Arm C exists to re-establish the true historical number.
- macOS stall risk on every ≥64K probe despite the guard (guard itself was once killed by pressure, 2026-04-20 incident) — mitigated by 32K-first protocol and formula-predicted fit checks, but a reboot loss of a day is possible.
- Environment drift since May (macOS updates, jetsam thresholds) may make even the full historical state (Arm C) irreproducible — that outcome is still a decision-grade null per kill criteria.
- Worktree arms double model/venv disk usage (~20-30 GB per arm incl. sglang clone + venv; model weights shared via HF cache) — check free disk before building both arms.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
