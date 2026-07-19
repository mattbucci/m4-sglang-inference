# M4-G: Add check_tool_call boot gate to M4 validate_capabilities.py (port of 3090 check + its error-body-surfacing _http_post, greedy-adapted)

> **Post-rebase delta (2026-07-19):** stack is v0.5.15.post1, box awake.
> The "bisect exclusion" scheduling constraint reduces to: don't run
> concurrently with any other serving item on the single box. qwen36 (green
> path) and qwen3-moe (negative control) are both validated presets on the
> new stack; greedy-adapted shipping still correct (sampling not yet
> implemented).

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | ~1h code+compile-gate; ~30 min core-path validation (qwen36 green path + qwen3-moe negative control); +4-5h optional full sweep incl. qwen35 (MLX boots up to 900s each per probe_all.sh wait_ready, qwen35 decodes ~15x slower) |
| **GPU time** | No discrete GPU, but the M4 unified-memory GPU performs all decode. Operative constraint is single-box serving exclusivity: ~30 min core path, ~4-5h full sweep — no concurrent model loads, AND cannot overlap the line-9 128K->32K regression bisect, which also needs exclusive serving (hard temporal exclusion, see depends_on). |
| **Depends on** | Box-exclusive with the line-9 128K->32K regression bisect (README queue line 9): this task touches NO SGLang/MLX stack code so it is bisect-SAFE for code, BUT the M4 is a single box and both this task and the bisect need EXCLUSIVE serving — they CANNOT run concurrently. This is a hard temporal exclusion, not soft coordination: schedule around the bisect's serving windows.; Soft: rig must boot at least one preset (dormant since 2026-05-21); if the stack is frozen for bisect arms, run this task's short-context boots in a bisect-free window.; M4-B 'Implement real sampling' (README queue line 10): NOT a blocker; the check ships greedy-adapted (temperature 0) with a marked revisit to restore donor sampling params after M4-B lands. |
| **Provides to** | All future M4 SWE-bench/opencode rollout work: a seconds-scale pre-rollout gate replacing multi-hour post-mortems (int4-bakeoff class), including actionable server error bodies on template/param rejects.; Fleet-queue item 'in-house qwen36 MLX 4-bit build' (README line 13): tool_call becomes part of the mandatory capability gate for any new checkpoint per CLAUDE.md quality rules.; 3090/R9700 sister teams: validator parity — M4 matrix results (esp. the qwen3-moe parser-mismatch diagnostic and any nemotron missing-parser findings) shared via sister-repo README convention. |

## Objective

M4's capability validator gates basic/thinking/vision but has no tool-call check, while every characterized M4 agentic failure was tool-call-class (malformed tool tags, 0 tokens under tool prompts, template rejecting tool round-trips). Porting the 3090's check_tool_call gives a seconds-scale boot gate that catches wrong/missing --tool-call-parser and non-tool-emitting models at the server layer, before multi-hour SWE-bench rollouts burn wall clock on empty diffs. This directly serves the fleet-queue recovery item (README bullet, line 15) for the tool-call-heavy long-context agentic-coding target.


## Background & receipts

- M4 validator /home/letsrtfm/AI/m4-sglang-inference/scripts/eval/validate_capabilities.py has exactly 3 checks (check_basic, check_thinking, check_vision) and no tools/tool_call code; exit 0 all-pass, 1 fail, 2 server-down; flags include --no-thinking, --thinking-kwarg, --skip-thinking, --include-vision, and an already-present --model override (line 220). Default host is 'localhost' (line 219), not the donor's 127.0.0.1 (verified in full).
- Donor: /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/validate_capabilities.py. check_tool_call at lines 254-309 has a TWO-ARG signature check_tool_call(base_url, model) and hardcodes chat_template_kwargs={'enable_thinking': False} at line 285 (NOT a 3-arg thinking_kwargs form). It sends a get_weather tools spec, asserts structured tool_calls[] with function.name=='get_weather' and JSON-parseable arguments containing 'location', and emits a raw-markup-in-content hint for '<function'/'<tool_call'/'[TOOL_CALLS]'/'functools'/'<|tool'. Wiring: NON_TOOL_MODELS frozenset at donor lines 607-609 (602-606 is its comment), --skip-tools at 622-623, auto-skip + main-loop insertion at 707-709 / 734-735 / 742-745.
- Error-surfacing gap (blocking, verified): M4's _http_post (lines 34-41) imports only urllib.request and lets HTTPError propagate as an opaque 'HTTP Error 400: Bad Request' with NO body. The donor's _http_post (lines 32-50) imports urllib.error and re-raises with the server's error body (e.read(), ~500 chars). Without porting that, a tools-field 4xx under the MLX v0.5.12 stack yields no actionable body, defeating kill-criterion #1 and the devstral-template finding. This port is required and is included in this task's scope.
- Motivating receipt: /home/letsrtfm/AI/m4-sglang-inference/evals/swebench/runs/int4-bakeoff-2026-05-18/README.md — qwen3-moe emitted malformed '<|name>read>' tags neither the qwen25 nor qwen3_coder parser recognized; gemma4-31b emitted 0 tokens under tool prompts (model-side, 'FP8 retry NO'); devstral preflight 400-rejected the tool-call canary; explicitly concludes these are chat-template/parser failures, not quantization. The same receipt shows nemotron-30b ran the full 900s producing tool-call attempts (a tool-EMITTING model with no parser wired), and lists nemotron-omni as UNTESTED in agentic — so neither is a non-tool model. qwen35 is one of only TWO models that WORK (with qwen36, at TIMEOUT=1800).
- Authoritative queue bullet: /home/letsrtfm/AI/m4-sglang-inference/README.md line 15 ('Add a boot-time check_tool_call gate to validate_capabilities.py'). Still unchecked.
- M4 launch presets carry the fleet parser mapping (/home/letsrtfm/AI/m4-sglang-inference/scripts/launch.sh, verified): mistral (devstral), qwen3_coder (coder-30b, coder-next, qwen35, qwen35-9b-8bit, qwen36, qwen36-27b), gemma4 (gemma4, gemma4-31b), qwen25 (qwen3-32b, qwen3-moe); NO --tool-call-parser on exactly three presets — smol-docling (211), nemotron-omni (254), nemotron-30b (271). Served model name == preset name (SERVED_NAME="${SERVED_NAME:-$PRESET}", launch.sh line 392), so the validator's server-reported model resolves to the preset key — same auto-skip mechanism as the 3090 works unmodified.
- Server-side tools support on the MLX stack is proven: evals/swebench/run_rollouts.py preflight_canary (starts line 69; tool_calls round-trip messages ~78-85) round-trips assistant tool_calls messages, and qwen36 achieved 21/26 patch-engagement through opencode via the qwen3_coder parser (README recommended-picks table).
- Gate inherits into both existing call sites with zero changes: run_all_evals.sh line 129 invokes the validator BARE; bench_smoke.sh line 62 passes $FLAGS (= '--no-thinking' for qwen* presets, empty otherwise). Neither passes --skip-tools, so the new gate runs in both — zero-touch inheritance holds. bench_smoke default presets (coder-30b devstral qwen3-moe qwen3-32b) all have parsers wired.
- Greedy constraint (CLAUDE.md 'Greedy sampling only — MLX backend uses argmax'): greedy is currently SELF-IMPOSED by M4's own patches 004/011/016, not an MLX-backend limitation (mlx-lm ships make_sampler). The donor's temperature=0.4/top_p/top_k sampling params are therefore no-ops on M4 until the fleet-queue 'Implement real sampling' item (M4-B, README line 10) lands.
- Stack pin: scripts/setup.sh SGLANG_COMMIT="v0.5.12" (line 24); README/CLAUDE.md stale-say v0.5.11. The family->parser mapping incl. the gemma4 parser name was audited on the 3090's v0.5.12 stack 2026-05-13 (3090 CLAUDE.md, SWE-bench rollout details), so the parser names predate v0.5.12->v0.5.15 divergence.
- Rig dormant since 2026-05-21 (last code commits 2026-05-21; the only newer commit is the 2026-07-18 README queue edit) — presets last booted ~8 weeks ago.


## Method

1. Read the donor: /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/validate_capabilities.py — check_tool_call at lines 254-309 (2-arg signature, hardcoded enable_thinking:False at 285), the error-surfacing _http_post at lines 32-50, and wiring at 607-609 (NON_TOOL_MODELS), 622-623 (--skip-tools), 707-709 + 734-735 + 742-745 (auto-skip + main-loop insertion).
2. Port the donor _http_post error-body surfacing into /home/letsrtfm/AI/m4-sglang-inference/scripts/eval/validate_capabilities.py: add 'import urllib.error' and wrap the existing urlopen (M4 lines 34-41) in the donor's try/except urllib.error.HTTPError that reads e.read().decode(errors='replace')[:500] and re-raises with '{reason}: {body}'. This is load-bearing for kill-criterion #1 and the devstral 400 finding; it modestly widens the 'one mechanism' scope but is required for the gate's diagnostics to be actionable.
3. Add check_tool_call(base_url, model) as a near-verbatim 2-ARG port of the donor (get_weather tools spec; structured tool_calls assertion name=='get_weather' + JSON args containing 'location'; raw-markup hint list verbatim; max_tokens=512, timeout=150). Two MLX adaptations: (a) replace temperature=0.4/top_p=0.95/top_k=20 with temperature=0 and a comment: 'greedy is currently forced by our MLX patches (004/011/016); temperature/top_p/top_k are no-ops until the real-sampling item (M4-B) lands — re-tune then. This is a self-imposed correctness constraint, not an MLX-backend limit.'; (b) KEEP the donor's hardcoded chat_template_kwargs={'enable_thinking': False} UNCONDITIONALLY — do NOT thread the caller's thinking_kwargs. Tool-calling is orthogonal to thinking, and a greedy thinking chain (gemma4 enable_thinking:true, or run_all_evals.sh:129 which invokes the validator bare) can hit max_tokens=512 and FAIL as truncated for a non-parser reason, muddying the parser diagnostic.
4. Wire it: add NON_TOOL_MODELS = frozenset({'smol-docling'}) ONLY (a VLM OCR smoke model, legitimately non-tool). Leave nemotron-30b and nemotron-omni OFF the skip list — the int4-bakeoff receipt shows nemotron-30b emits tool-call attempts (missing-parser bug the gate MUST surface as an actionable [FAIL]/finding) and nemotron-omni is UNTESTED; any future skip-list addition requires per-model evidence of non-tool-TRAINING, never parser absence. Add --skip-tools argparse flag with auto-skip when the server-reported model is in NON_TOOL_MODELS (donor pattern; works because served name == preset name). Insert the check between basic and thinking in main() with result label 'tool_call', matching donor print format. Add a code comment: auto-skip is keyed on served model name == preset name; SERVED_NAME overrides defeat it.
5. Gate 1 (no server needed): python3 -m py_compile scripts/eval/validate_capabilities.py; then python3 scripts/eval/validate_capabilities.py --port 23334 with no server up must still exit 2, and --help must show --skip-tools.
6. Green path / positive anchor #1: bash scripts/launch.sh qwen36 (nohup + disown per probe_all.sh, wait for /health=200 up to 900s), then python3 scripts/eval/validate_capabilities.py --port 23334 --no-thinking. Require '[PASS] tool_call' with name='get_weather' and JSON location args, overall exit 0; record the tool_call check wall time alone (target seconds-scale). If an unexpected ECONNREFUSED hits the probe, check IPv4/IPv6 binding first — M4 default host is 'localhost' (line 219) vs the donor's 127.0.0.1; macOS resolves localhost fine but rule this out before theorizing.
7. Positive anchor #2: boot qwen35 (kill qwen36 first with the triple-pkill), run the validator. The receipt lists qwen35 as one of only two models that WORK, same qwen3_coder+DeltaNet+VL arch as qwen36; it gives the matrix a second PASS anchor. Accommodate its ~15x-slower decode in the wait/timeout; expect '[PASS] tool_call'.
8. Negative control (receipt-anchored, zero contrivance): boot qwen3-moe (already in the preset set, boots, has the qwen25 parser). The int4-bakeoff receipt documents it emitting malformed '<|name>read>' tags neither parser recognizes, so the gate MUST return '[FAIL] tool_call' with 'no tool_calls' and the raw-markup-in-content hint, overall exit 1 — a receipt-anchored anti-vacuousness proof. A PASS here means the gate is vacuous — STOP and rework before committing. Fallback if qwen3-moe will not boot after 8 weeks dormant: any parser-less preset (nemotron-30b or smol-docling) FAILs tool_call identically for the missing-parser reason; use one as the fallback negative control and note the substitution in the receipt.
9. Optional sweep (one preset at a time, kill between with the triple-pkill): coder-30b, devstral, qwen3-32b, gemma4, gemma4-31b, qwen35-9b-8bit, qwen36-27b (qwen3-moe already covered as negative control; qwen35/qwen36 already covered as positive anchors). Record a per-preset PASS/FAIL matrix. Reconcile against the int4-bakeoff receipt: gemma4-31b FAIL is EXPECTED (model emits 0 tokens under tool prompts — model-side, keep it running as a canary, do NOT skip-list it); nemotron-30b / nemotron-omni FAILs (missing parser) are actionable findings, not skips. Only add presets to NON_TOOL_MODELS from per-model non-tool-training evidence, never from a measured FAIL.
10. Write the receipt: benchmarks/quality/tool-call-gate-matrix-<date>.md with the per-preset matrix, per-check wall times, raw validator output snippets (including any surfaced HTTP error bodies), the qwen3-moe negative-control evidence, and the int4-bakeoff reconciliation. Commit validator edit (error-surfacing _http_post + check_tool_call + NON_TOOL_MODELS + --skip-tools + main() wiring) + receipt as one self-contained commit; tick the README fleet-queue line-15 checkbox in the same or a follow-up commit per repo README discipline.


## Baseline & instrument

Pre-change validator on a booted qwen36 (python scripts/eval/validate_capabilities.py --port 23334 --no-thinking) prints only basic (+skipped thinking) — no tool line, exit 0 even if tool parsing were broken. The tool-call failure class itself is baselined by evals/swebench/runs/int4-bakeoff-2026-05-18/README.md (multi-hour rollout post-mortems: qwen3-moe malformed tags, gemma4-31b 0-tokens, devstral 400, nemotron-30b tool-attempts-no-converge). Instrument: the validator's printed [PASS]/[FAIL] lines + exit code, plus any HTTP error body surfaced by the newly-ported _http_post.


## Success criteria

- qwen36 green path (positive anchor #1): validator prints '[PASS] tool_call' with name='get_weather' and JSON arguments containing 'location', overall exit 0; tool_call check wall time recorded and under ~60s (expected seconds-scale on MoE-3B-active greedy decode of <=512 tokens).
- qwen35 (positive anchor #2): '[PASS] tool_call' — a second receipt-backed positive, confirming the gate is not merely qwen36-specific.
- Negative control: a qwen3-moe boot yields '[FAIL] tool_call' with 'no tool_calls' and the raw-markup-in-content hint (matching the receipt's '<|name>read>' finding) and overall exit 1 — the gate demonstrably fails on a receipt-documented instance of the exact class the int4-bakeoff hit, with zero contrivance.
- Auto-skip verified: a default smol-docling boot prints the auto-skip line and is not penalized; --skip-tools flag works. nemotron-30b / nemotron-omni are NOT auto-skipped and their tool_call result is reported (PASS or actionable FAIL), not hidden.
- Error-surfacing verified: a tools-field 4xx (if any preset produces one, e.g. devstral) surfaces the server's error body in the validator output, not an opaque 'HTTP Error 400: Bad Request'.
- Zero-touch inheritance: bench_smoke.sh (line 62, $FLAGS) and run_all_evals.sh (line 129, bare) run unmodified and now exercise the gate (verify by reading their output, not just exit status).
- Receipt committed at benchmarks/quality/tool-call-gate-matrix-<date>.md with per-preset matrix reconciled against the int4-bakeoff table (gemma4-31b expected-FAIL documented as model-side canary; nemotron missing-parser FAILs documented as findings).


## Kill criteria

- If SGLang v0.5.12 argparse rejects any preset's --tool-call-parser name at boot (e.g. gemma4) or the /v1/chat/completions endpoint 4xxs on the 'tools' field under the MLX stack: record the exact server error BODY (now surfaced by the ported _http_post) as a finding in the receipt, land the gate code with py_compile + whatever presets do boot, and file the incompatibility to the version-bisect/rebase queue items — do NOT patch SGLang serving internals under this task.
- If the rig cannot boot ANY preset after 8 weeks dormant (MLX/macOS lib drift): stop GPU-side validation, land the code change gated by py_compile only, mark validation blocked-on the long-context-regression-bisect item (README queue line 9), record as partial.
- If the negative control PASSes (gate cannot distinguish a malformed-tag / parser-less serving from a real tool_calls response), STOP and rework before committing — a vacuous gate is worse than none.
- If any single preset boot exceeds the 900s wait_ready ceiling twice, mark that cell 'boot-fail' in the matrix and move on — this task does not debug preset boots.


## Deliverables

- Edited /home/letsrtfm/AI/m4-sglang-inference/scripts/eval/validate_capabilities.py: error-body-surfacing _http_post (import urllib.error) + check_tool_call (2-arg, hardcoded enable_thinking:False, temperature=0) + NON_TOOL_MODELS=frozenset({'smol-docling'}) + --skip-tools + main() wiring (single logical mechanism: a working tool-call gate; the _http_post port is a required dependency, not an unrelated bundle).
- Receipt: /home/letsrtfm/AI/m4-sglang-inference/benchmarks/quality/tool-call-gate-matrix-<date>.md — per-preset PASS/FAIL matrix, per-check timings, qwen3-moe negative-control evidence, surfaced HTTP error bodies, int4-bakeoff reconciliation.
- README fleet-audit queue line-15 checkbox ticked (same commit series).
- Raw logs preserved under /tmp during the run and quoted into the receipt (rig has no other log convention for validator runs).


## Constraints

- One server at a time; kill between presets with the bench_smoke.sh triple-pkill pattern (sglang.launch_server / sglang::scheduler / sglang::detokenizer, lines 39-41) — orphaned schedulers hold ~GBs RSS across swaps (probe_all.sh 2026-05-16 audit note).
- Never raise MEM_FRAC above the 0.7 default; probe requests are short-context (<4K) so oom_guard is not mandatory here, but do not mix this sweep with any >=32K work without it (CLAUDE.md).
- One mechanism per change: the tool-call gate + its required _http_post error-body port ship together; do NOT bundle real sampling (M4-B), version-header fixes, or setup.sh gate work into this commit.
- Validate behavior, not exit status: read the printed tool_call name/args line, any surfaced HTTP error body, and the server log — not just rc.
- READ the served-name caveat into the code comment: auto-skip is keyed on served model name == preset name; SERVED_NAME overrides defeat it.
- check_tool_call ALWAYS forces chat_template_kwargs={'enable_thinking': False} (donor behavior) regardless of caller thinking flags — this keeps the parser diagnostic clean when the gate is invoked bare (run_all_evals.sh:129).
- Full sweep (>30 min) should be driven detached (nohup + disown; macOS has no setsid — probe_all.sh lines 122-128 pattern).
- Negative results are findings: boot-fails, expected-FAILs (gemma4-31b canary), and missing-parser FAILs (nemotron) go in the matrix with receipts, not silently omitted or skip-listed.


## Risks

- 8-week dormancy: macOS/Homebrew/MLX drift may break preset boots before the gate is testable (mitigated by kill criterion: land code-only).
- gemma4/mistral/qwen25 parser behavior on the M4 v0.5.12 tree is inferred from the 3090's v0.5.12 audit (2026-05-13) + M4 launch.sh comments; no M4 receipt yet shows a structured tool_calls[] response for those presets — the sweep is the first direct measurement; treat surprises as findings (the ported error-body surfacing makes template/param rejects actionable).
- Greedy decode with thinking disabled can still degenerate on some Qwen checkpoints; max_tokens=512 caps the cost and finish_reason='length' correctly reads as FAIL.
- gemma4-31b will FAIL by design (model-side 0-tokens-under-tools) — risk is a future operator 'fixing' the noise by skip-listing it; the receipt must label it expected-fail canary explicitly. Same risk for nemotron missing-parser FAILs — they are findings, not skip candidates.
- Devstral strict user/assistant alternation broke the rollout preflight canary (a multi-message round-trip), but this probe sends a single user message + tools, which Mistral templates accept — if it 400s anyway, the ported _http_post now surfaces the template error body as a real finding, not a probe bug.
- Negative-control dependency: if qwen3-moe will not boot, the fallback parser-less negative control (nemotron-30b/smol-docling) proves a WEAKER class (missing-parser vs malformed-tag-with-parser); note the substitution and its reduced strength in the receipt rather than treating it as equivalent.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
