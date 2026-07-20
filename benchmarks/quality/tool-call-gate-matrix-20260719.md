# tool_call boot gate — validation matrix (2026-07-19)

`check_tool_call` ported from the 3090 validator (structured `tool_calls[]`
assertion + raw-markup hint + error-body-surfacing `_http_post`), wired into
`validate_capabilities.py` between basic and thinking. Deterministic
(temperature=0, enable_thinking:False forced). Stack: v0.5.15.post1 + 8
patches.

| Preset | Parser | tool_call | Wall time (full validator) | Evidence |
|--------|--------|-----------|:---:|---|
| qwen36 | qwen3_coder | **PASS** — `finish=tool_calls name='get_weather' args='{"location": "Paris"}'` | 1.0 s | positive anchor |
| qwen3-moe | qwen25 | **PASS** — identical structured response | 1.2 s | see note below |
| nemotron-30b | (none wired) | **FAIL** — `no tool_calls finish=stop raw-markup-in-content(<function)`, content = `\n<tool_call>\n<function=get_weather>\n<parameter=location>\nPar…`, exit 1 | 1.1 s | negative control |
| nemotron-30b | qwen3_coder (fix from the transcript) | **PASS** — `finish=tool_calls name='get_weather' args='{"location": "Paris"}'` | 1.0 s | fix validated by the gate |
| qwen35 | qwen3_coder | **PASS** — structured tool_calls | (re-measure window) | full-validator run |
| qwen3-32b | qwen25 | **PASS** — structured tool_calls | (re-measure window) | full-validator run |
| qwen35-9b-8bit | qwen3_coder | **PASS** — structured tool_calls | (re-measure window) | full-validator run |
| qwen36-27b | qwen3_coder | **PASS** — structured tool_calls | (re-measure window) | full-validator run |

## Negative control & anti-vacuousness

The spec's planned negative control (qwen3-moe, from the int4-bakeoff
malformed-`<|name>read>` receipt) **passes on the current stack** — that
receipt was measured on the v0.5.12-era pin under the opencode multi-tool
harness; the single-tool canary + qwen25 parser round-trips cleanly today.
Recorded as a stack improvement, with the caveat that the canary is narrower
than the agentic harness that surfaced the original failure.

Gate detection is instead proven by the parser-less class: **nemotron-30b
emits well-formed `<function=get_weather>` markup as plain content** (no
`--tool-call-parser` on the preset) and the gate fails it with the exact
diagnostic hint. This matches the int4-bakeoff observation (nemotron ran 900s
producing tool-call attempts that no parser consumed).

**Finding → fix, same session:** nemotron-30b's emitted format
(`<tool_call>` / `<function=...>` / `<parameter=...>`) is the qwen3-coder
family format; `--tool-call-parser qwen3_coder` is now wired into the preset
and the gate validates structured tool_calls (row above).

## Gate-1 checks (no server)

`py_compile` clean; no-server invocation exits 2; `--skip-tools` in help;
auto-skip list = `{smol-docling}` only (non-tool-trained VLM smoke model).
Zero-touch inheritance: `run_all_evals.sh` (bare) and `bench_smoke.sh`
(`$FLAGS`) invoke the validator unmodified and now run the gate.

## Not yet measured

coder-30b, devstral, gemma4* — coder-30b/devstral ride future serve windows;
gemma4* are blocked at boot. gemma4-31b is the expected-FAIL model-side
canary (0 tokens under tool prompts per the bakeoff receipt); it must never
be skip-listed.

Validator side-note from the re-measure windows: qwen35 and qwen35-9b-8bit
FAIL the greedy thinking check as TRUNCATED at 2048 tokens (verbosity, not
the loop — the content-aware probe_thinking passes, and model-recommended
sampling via patch 016 is the recommended thinking path). qwen36-27b passes
all three checks including thinking.
