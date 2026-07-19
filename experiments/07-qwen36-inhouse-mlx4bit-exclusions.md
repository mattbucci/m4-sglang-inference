# M4-E: In-house qwen36 MLX 4-bit with router/DeltaNet/vision exclusions

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | 2-3 days (67 GB BF16 download + hours-scale conversion + ~2h eval A/B per build + 3-instance agentic smoke) |
| **GPU time** | none (Apple unified memory — no discrete GPU; box is exclusively occupied during conversion and evals, ~1-2 days cumulative) |
| **Depends on** | None hard. Coordinate box time with M4 queue item 1 (128K->32K bisect) — this experiment must not rebase or touch patches, and both want exclusive box occupancy.; HF access to Qwen/Qwen3.6-35B-A3B (same id the 3090 calib script defaults to; ~/.secrets/hf_token pattern already used by audit script). |
| **Provides to** | scripts/launch.sh qwen36 preset (new MODEL default on promotion) — flagship for all downstream M4 agentic/eval work incl. the SWE-bench Docker-scoring handoff cell.; M4 queue sampling item: probe_thinking A/B result isolates DeltaNet-INT4 vs greedy-decode blame for the infinite-<think> loop.; Fleet: first MLX checkpoint following the GPU lanes' hazard-class recipe; conversion driver is the template for cleaning qwen36-27b (96 in_proj layers) and qwen35 next. |

## Objective

Replace the M4 flagship checkpoint mlx-community/Qwen3.6-35B-A3B-4bit — which INT4-quantizes 40 MoE router (mlp.gate) layers and 60 DeltaNet in_proj_a/b layers, both fleet-banned hazard classes — with an in-house MLX 4-bit conversion that keeps those classes plus the vision tower in BF16, mirroring the GPU lanes' proven AWQ ignore recipe. Checkpoint quality dominates M4 outcomes (Coder-30B DWQ swap was +20pp HumanEval), and qwen36 is the only model that completes the agentic loop on M4, so a cleaner flagship lifts the rig's primary metric directly.


## Hypothesis

An in-house MLX 4-bit Qwen3.6-35B-A3B with mlp.gate + linear_attn.in_proj_a/b (+ shared_expert_gate, vision tower) kept BF16 scores >= the incumbent on MMLU-100/HumanEval-20 (no regression beyond noise, expected gain), passes all modality probes, and matches or improves 3-instance SWE-bench patch-engagement; secondarily, if DeltaNet in_proj INT4 drives the infinite-<think> loop, probe_thinking.py termination improves on the in-house build under identical greedy decode.


## Background & receipts

- Hazard audit receipt: /home/letsrtfm/AI/m4-sglang-inference/benchmarks/quality/mlx-metadata-audit-2026-05-13.txt lines 19-21 — mlx-community/Qwen3.6-35B-A3B-4bit has router(bf/q)=120(0/40) and deltanet(bf/q)=180(0/60); vision is already clean at 333(167/0), so vision needs parity-preservation, not fixing.
- Audit rationale in-repo: /home/letsrtfm/AI/m4-sglang-inference/scripts/eval/audit_mlx_quant_metadata.py docstring — mlp.gate must NOT be quantized (top-k routing collapses under INT4); linear_attn.in_proj_a/b must NOT be (recurrent state error accumulates). README.md line 206 flags in_proj INT4 as 'strong candidate for the root of the known Qwen3.5 infinite-<think> loop'.
- DWQ precedent receipt: README.md 'Quality lift after swapping to the DWQ variant' table — HumanEval 75.0% -> 95.0% (+20pp) from a checkpoint swap alone, decode flat; raw scan in benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt.
- GPU-lane donor recipe: /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/quantize/quantize_qwen36_thinking_vision.py — BASE_MODEL default Qwen/Qwen3.6-35B-A3B (67 GB BF16), ignore list: lm_head, re:.*in_proj_b$, re:.*in_proj_a$, re:.*mlp\.gate$, re:.*shared_expert\..*, re:.*shared_expert_gate$, plus vision patterns (visual/vision_tower/multi_modal_projector/embed_vision).
- Narrow-ignore lesson: 3090 CLAUDE.md line 109 — broad re:.*linear_attn\..* broke the GPU AWQ loader (expects INT4 qkvz); v3 working pattern is ONLY in_proj_a$/in_proj_b$ narrow excludes. MLX analog verified safe: MLX records per-layer quantization in config.json and the incumbent already loads with a fully-BF16 vision tower (167 BF16 keys, 0 quantized) on this stack, so unquantized subsets are loader-safe — but keep the narrow pattern anyway for GPU-recipe parity and size.
- Loader is mlx-vlm, not mlx-lm: patches/README.md patch 010 lists qwen3_5_moe among mlx_vlm LanguageModel families the M4 stack loads; mlx-vlm ~0.5.0 is bundled (README 'Active work' item 2: mlx-vlm 0.4.4 -> 0.5.0). So conversion must go through mlx-vlm (or a driver on mlx.nn.quantize) to keep vision — mlx_lm.convert would drop the vision tower and violate the preserve-modalities invariant.
- Incumbent baseline receipt: /home/letsrtfm/AI/m4-sglang-inference/benchmarks/quality/Qwen3.6-35B-A3B.json (2026-05-18) — MMLU 80/100, HumanEval 17/20 (completions, no-thinking), Needle 3/3; SWE-bench patch-engagement 21/26 in README. Numbers predate the v0.5.12 rebase and the rig's dormancy — re-baseline required.
- Launch integration: scripts/launch.sh qwen36 preset uses MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}" — env override slots the in-house build in with zero code changes; evals/swebench/smoke.sh sets only PRESET/MODEL_KEY so MODEL passes through.
- Instruments verified present: scripts/eval/check_mlx_quant_scales.py (accepts local dirs via resolve_model_dir), scripts/eval/validate_capabilities.py (basic/thinking/vision, --include-vision), scripts/eval/probe_all.sh (qwen36 -> codegen+vision+video+thinking), scripts/eval/eval_and_chart.py (--tag writes benchmarks/quality/<tag>.json), scripts/eval/probe_thinking.py, scripts/common/oom_guard.sh.
- audit_mlx_quant_metadata.py is HF-remote-only (audit() fetches https://huggingface.co/<repo>/... — no local-dir branch), so post-convert recipe verification on the local output needs a local index.json check (method step 6) or an HF upload first.


## Method

1. Preflight (box): df -h $HOME — require >=100 GB free (67 GB BF16 source + ~20-21 GB output; incumbent 4-bit is ~19 GB). Confirm source repo access: hf download Qwen/Qwen3.6-35B-A3B --include config.json (3090's calib script pulls this exact id). Start the guard for every long step: bash scripts/common/oom_guard.sh & (mandatory — macOS has no OOM killer; box stalls until reboot).
2. Introspect the pinned converter before writing anything: source .venv/bin/activate; python -c "import inspect, mlx_vlm; from mlx_vlm import convert; print(mlx_vlm.__version__); print(inspect.signature(convert))" — looking for skip_vision and a quant_predicate/class_predicate parameter. Record the signature in the run log.
3. Write scripts/quantize/convert_qwen36_inhouse.py: Route A (predicate supported) — call mlx_vlm convert(hf_path='Qwen/Qwen3.6-35B-A3B', mlx_path=os.path.expanduser('~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse'), quantize=True, q_bits=4, q_group_size=64, skip_vision=True, quant_predicate=pred). Route B (no predicate in pinned mlx-vlm) — load via mlx-vlm's load_model, then mlx.nn.quantize(model, group_size=64, bits=4, class_predicate=pred), then save weights + config with per-layer quantization entries (mirror how the incumbent's config records its BF16 vision layers). Do NOT fall back to mlx_lm.convert — it drops the vision tower.
4. Predicate (the core of the experiment; mirrors 3090 v3 narrow recipe): keep BF16 (return False) iff module path endswith 'mlp.gate' (exact — must NOT match 'gate_proj' or 'shared_expert_gate' substring accidentally), endswith 'linear_attn.in_proj_a' or 'linear_attn.in_proj_b', endswith 'shared_expert_gate' (out_dim=1 scalar gate), or contains any vision prefix ('visual', 'vision_tower', 'multi_modal_projector', 'embed_vision', 'vision_model'). Everything else (incl. in_proj_qkvz, out_proj, experts, lm_head per MLX default) quantizes INT4 g64 — do NOT broadly exclude linear_attn.* (3090 lesson: only the narrow a/b gates).
5. Run the conversion detached (well over 30 min): setsid nohup python scripts/quantize/convert_qwen36_inhouse.py > /tmp/convert_qwen36_inhouse.log 2>&1 & — with oom_guard running. No serving/benches on the box during conversion.
6. Gate 1, recipe verification on the local output (audit_mlx_quant_metadata.py is HF-remote-only): python one-liner over ~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse/model.safetensors.index.json asserting ZERO keys matching ('mlp.gate.scales'|'in_proj_a.scales'|'in_proj_b.scales'|'shared_expert_gate.scales'|vision-prefix '.scales'), i.e. router 40->0, deltanet 60->0 quantized vs the incumbent's audit row; also confirm config.json quantization section carries the per-layer false entries. Save output to benchmarks/quality/ as the recipe receipt.
7. Gate 2, corruption scan: python scripts/eval/check_mlx_quant_scales.py ~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse — exit 0 required (catches all-zero/NaN scales class of silent disaster).
8. Re-baseline the incumbent on the CURRENT stack first (old numbers are v0.5.11-era; rig dormant since 2026-05-21): ./scripts/launch.sh qwen36; python scripts/eval/validate_capabilities.py --port 23334 --include-vision; python scripts/eval/eval_and_chart.py --run --port 23334 --workers 1 --tag 'Qwen36-incumbent-20260718' --mmlu-samples 100 --humaneval-samples 20 --labbench-samples 25 --needle-lengths 1024,4096,16384 --no-thinking; python scripts/eval/probe_thinking.py --port 23334 (record whether <think> terminates). Kill server between runs.
9. Gate 3, in-house boot + modality probes (single mechanism changed: checkpoint only; identical stack, patches, flags): MODEL=$HOME/AI/models/Qwen3.6-35B-A3B-4bit-inhouse ./scripts/launch.sh qwen36; validate_capabilities.py --port 23334 --include-vision must exit 0; then MODEL=... PRESETS='qwen36' bash scripts/eval/probe_all.sh — codegen+vision+video+thinking verdicts vs incumbent's (preserve-modalities invariant: all four must be probed, none regressed).
10. Gate 4, quality A/B, same instrument: MODEL=... server up, eval_and_chart.py --run --tag 'Qwen36-inhouse-20260718' with identical flags as step 8; plus probe_thinking.py on the in-house build (the infinite-<think> hypothesis test — same greedy decode both arms).
11. Gate 5, agentic smoke (per-instance restart pattern per README jetsam note is built into smoke.sh): MODEL=$HOME/AI/models/Qwen3.6-35B-A3B-4bit-inhouse PRESET=qwen36 MODEL_KEY=qwen36 INSTANCES=3 TIMEOUT=900 bash evals/swebench/smoke.sh; compare patch-engagement and resolved on the same 3 instances vs incumbent (evals/swebench/runs/qwen36-3instance-2026-05-18/ is the historical reference; re-run incumbent arm if drift suspected).
12. On pass: flip scripts/launch.sh qwen36 MODEL default to the in-house path (or upload to mattbucci/Qwen3.6-35B-A3B-4bit-mlx-clean and point at that — upstream-BF16-prune-ourselves rule; upload also lets audit_mlx_quant_metadata.py --repo verify it remotely), add the README model-table row + audit-table 'fixed' note, and record the thinking-termination outcome either way (negative result is a finding). Delete the 67 GB BF16 source only after the checkpoint is promoted.


## Baseline & instrument

Incumbent mlx-community/Qwen3.6-35B-A3B-4bit re-measured on the current v0.5.12 stack before the A/B (step 8): eval_and_chart.py --mmlu-samples 100 --humaneval-samples 20 --no-thinking + validate_capabilities.py + probe_all.sh + probe_thinking.py + 3-instance smoke.sh. Historical reference: MMLU 80/100, HumanEval 17/20, Needle 3/3 (benchmarks/quality/Qwen3.6-35B-A3B.json, 2026-05-18) and patch-engagement 21/26 (README).


## Success criteria

- Recipe gate: local index.json shows 0 quantized (.scales-bearing) layers in mlp.gate / in_proj_a / in_proj_b / shared_expert_gate / vision classes (incumbent: 40 router + 60 deltanet quantized), receipt saved under benchmarks/quality/.
- check_mlx_quant_scales.py exits 0 on the in-house dir (no all-zero/NaN/Inf scales).
- validate_capabilities.py --include-vision exits 0 on the in-house build; probe_all.sh qwen36 verdicts (codegen/vision/video/thinking) each >= incumbent's same-day verdicts.
- MMLU-100 and HumanEval-20 (eval_and_chart.py, --no-thinking, identical flags both arms) >= incumbent re-baseline minus noise (MMLU within -2pp; any HE gain is the DWQ-precedent payoff); Needle 3/3.
- 3-instance smoke.sh patch-engagement >= incumbent on the same instances.
- Decode sanity: server-log gen-throughput at short context within ~10% of incumbent (BF16 router adds compute; dead layers were free — a real cost is acceptable only if quality gates pass).
- probe_thinking.py termination behavior recorded for both arms — improvement confirms the in_proj hypothesis; no-change is a documented null that redirects blame to greedy decode (M4 queue sampling item).


## Kill criteria

- Pinned mlx-vlm supports neither skip_vision+predicate nor a workable nn.quantize driver route that keeps vision — record the signature dump as the null receipt; do NOT ship a text-only mlx_lm.convert build (violates preserve-modalities).
- Conversion OOM-stalls the box twice even with oom_guard.sh active and lazy loading — stop, record peak-memory observations, park for a bigger-RAM approach.
- Free disk < 100 GB and nothing deletable without user input — stop, status blocked, ask Matt.
- In-house build fails validate_capabilities basic check, or MMLU drops >5pp vs incumbent re-baseline, after one corrected-predicate re-convert attempt — keep incumbent, record null with receipts.
- Any gate reveals the incumbent itself no longer boots/scores on the current stack (dormant-rig rot) — stop this experiment and report; the regression-bisect queue item owns that problem.


## Deliverables

- scripts/quantize/convert_qwen36_inhouse.py (new; predicate documented inline with the 3090 narrow-ignore citation)
- ~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse/ (local checkpoint; optional HF upload mattbucci/Qwen3.6-35B-A3B-4bit-mlx-clean)
- benchmarks/quality/Qwen36-incumbent-20260718.json and benchmarks/quality/Qwen36-inhouse-20260718.json (eval_and_chart receipts)
- Recipe-verification + check_mlx_quant_scales receipts under benchmarks/quality/ (e.g. inhouse-quant-scan-2026-07.txt)
- probe-trio JSONs under benchmarks/quality/probe-trio/ + probe_thinking transcripts for both arms
- smoke.sh run dir under evals/swebench/runs/ for the 3-instance A/B
- README updates: model-table row, audit-table fixed note, launch.sh qwen36 MODEL default flip on promotion


## Constraints

- oom_guard.sh MANDATORY in background for conversion and all 32K+ serving (macOS has no OOM killer; box stalls until reboot).
- Never raise MEM_FRAC/--mem-fraction-static above 0.7 — on unified memory it is a fraction of TOTAL system RAM.
- Detach >30 min jobs via setsid (download, conversion, eval sweeps); no serving or benches concurrent with the conversion run.
- One mechanism at a time: checkpoint is the only variable; identical stack/patches/flags/greedy decode in both A/B arms.
- Preserve thinking+image+video (+audio n/a for this model) — all modality probes run on any checkpoint work; full-model rule (35B flagship, no small-variant substitution).
- Multi-instance sweeps use the per-instance server-restart pattern (jetsam at 131K).
- Decode claims from server-log gen-throughput only; no bench_serving here (its --random-range-ratio 1 fix is a separate queue item).


## Risks

- 67 GB BF16 vs 64 GB unified RAM during conversion: mlx lazy loading should keep peak bounded, but this is unverified on this box for a model this size — oom_guard + detached run + kill criterion cover the stall mode.
- mlx-vlm 0.5.0 predicate API shape is unverified from this (Linux) box — step 2's introspection gates the route choice before any code is written.
- BF16 router/gates add real decode compute (incumbent's quantized router was 'free') — decode-parity check catches a meaningful regression.
- Incumbent baselines are 2 months old on a since-rebased stack; skipping the re-baseline (step 8) would make the A/B unattributable.
- shared_expert kept INT4 (unlike the 3090's full shared_expert.* BF16 exclusion) to bound size on 64 GB — if quality gates fail marginally, widening to shared_expert.* is the one sanctioned recipe iteration before declaring the null.
- Vision-prefix name mismatch between HF BF16 source and mlx-vlm module paths could silently no-op the vision exclusion — Gate 1's index scan catches it post-convert.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
