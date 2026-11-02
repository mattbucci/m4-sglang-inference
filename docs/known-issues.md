# Known issues (staging copy)

Staging text for the GitHub issue tracker. Each bullet becomes one issue; delete
this file once they are filed. Live tracking: https://github.com/mattbucci/m4-sglang-inference/issues

- **`gemma4` / `gemma4-31b` blocked at boot.** mlx-vlm 0.6.5's Gemma 4 attention is not
  recognized by the MLX attention contract, so its layers are classed as auxiliary state and
  hybrid component extraction fails on `lm_head` (`AttributeError: 'LanguageModel' object has
  no attribute 'lm_head'`). Needs a contract entry for that module. Receipt:
  `benchmarks/quality/probe-trio/gemma4.json`.
- **`--kv-cache-dtype` is inert on the MLX backend.** The `fp8` / `turboquant` / `fp16`
  aliases (patch 002) let presets pass argparse; the pool and per-request caches store KV in
  the attention compute dtype (bf16). Working memory levers: `--max-total-tokens`, `CHUNKED`,
  `MAX_RUNNING`, request `max_tokens`, `SGLANG_MLX_CACHE_LIMIT_GB`. 4-bit KV
  (https://github.com/ml-explore/mlx/discussions/3134) is a prospect, not a shipped lever.
- **Dense-attention ceiling is structural.** `mx.fast.scaled_dot_product_attention`
  materializes the per-chunk score tensor, which reaches 30–90 GB at deep offsets; 32K is the
  practical decode ceiling for dense full-attention models. DeltaNet hybrids are the workaround.
- **Radix-on-hybrid disables the overlap schedule** (upstream `no_buffer` constraint);
  hybrids and `flint` run the normal event loop, which costs Flint ~12% of decode.
- **Flint omits opencode's required bash `description` argument** on nearly every call
  under opencode's real prompt, so opencode rejects the call with a `SchemaError` and the
  agent retries until timeout. No prompt wording fixes it (0–3 of ~12 calls; a short prompt
  gives 10/10). Use the harness proxy's backfill (`BACKFILL_BASH_DESCRIPTION=1`) or disable
  the bash tool (`tools: {bash: false}`).
- **Coder-Next-80B infeasible** on 64 GB: 42 GB of weights alone exceed the safe budget;
  model load OOMs.
- **GPU driver kernel panic on large prefill chunks**
  (`panic(cpu 6): "completeMemory() prepare count underflow" @IOGPUMemory.cpp:550`, watchdog
  reset; also https://github.com/ml-explore/mlx/issues/3186,
  https://github.com/ml-explore/mlx/issues/3346, https://github.com/ml-explore/mlx-lm/issues/883).
  Mechanism: a nonzero MLX wired limit puts every new Metal buffer into a residency set
  committed on each add and release, and that traffic trips the IOGPUFamily reference count.
  Mitigation: patch 024 skips the wired limit (`SGLANG_MLX_WIRED_LIMIT_GB`, `common.sh`
  default `0`; the identical workload then completes with decode within 1%, validator 4/4,
  `benchmarks/patch-gates/024-wired-limit-test-2026-10-31.log`) and `qwen36-27b` prefills in
  2,048-token chunks. With no wired limit weights and KV are pageable, so the OOM guard stays
  mandatory; `SGLANG_MLX_WIRED_LIMIT_GB=64` restores upstream behaviour for an A/B. A reboot
  with `Boot faults: wdog` and this panic string is this issue.
- **`mlx-community/Qwen3.8-27B-8bit` does not boot as shipped.** Its `processor_config.json`
  names `Qwen3VLImageProcessor`, which `AutoProcessor` cannot load (`Unrecognized image
  processor`). Point `MODEL=` at a mirror of the snapshot without that file;
  `preprocessor_config.json` then resolves the Qwen2VL image processor.
- **macOS has no OOM killer.** A process touching a page past physical RAM stalls the box
  until reboot. OOM guard mandatory for ≥32K work: `bash scripts/common/oom_guard.sh &`
  pkills the server when kernel-reported available memory (`kern.memorystatus_level`) drops
  below 8 GB. vm_stat free+inactive undercounts by the file pages a checkpoint load streams
  through the page cache (a 42 GB Flint load read 6.9 GB on that metric while the kernel
  reported 15 GB), which is why the guard reads the kernel level.
- **Jetsam can silently reap the scheduler mid-eval** (`Connection refused`, no traceback).
  `needle_eval` tags `server_dead=True` on connection errors; `mmlu_eval` / `humaneval_eval` /
  `labbench_eval` do not. When an eval section drops to ~0%, suspect jetsam before model
  regression; relaunch and re-run the section. Multi-instance SWE-bench sweeps at CTX=131K
  need per-instance server restart for the same reason.
- **`--mem-fraction-static` is a fraction of TOTAL system RAM** on unified memory. 0.85 has
  hard-locked the box (compressor + swap ~150 GB effective). The 0.7 default is the ceiling;
  long-context presets go down to 0.4–0.5.
- **Sampling caveats.** `top_k==1` is served greedy; repetition / frequency / presence
  penalties are not applied on the MLX sampling path.
- **`/v1/responses` is disabled at boot.** The pinned `openai==2.6.1` lacks a type the
  Responses module imports (`ResponseFunctionShellToolCall`); chat completions serve normally.
  `pip install -U openai` restores the endpoint at the cost of the pin.
- **HDMI display blackout** when heavy Metal compute starts: M4 Pro quirk, not an SGLang bug.
- **Quality-table re-run.** Every quality-table row except Qwen3.8-27B predates the current
  stack; the re-run is queued (`scripts/eval/run_all_evals.sh`), and full LAB-Bench needs a
  server restart between categories.
- **`qwen3-32b` 32K tripwire point** needs `MEM_FRAC=0.5` (the 0.45 pool caps requests at
  25,241 tokens); the baseline keeps the point from a 0.5 serve.
- **Current-stack SWE-bench predictions** (`evals/swebench/exports/qwen36-p023-predictions.jsonl`)
  await Docker scoring on the 3090; the published resolve rate is the earlier pin's.
