# nemotron-omni int4 agentic test — MLX backend bug, not model failure

## Goal

Close the last untested int4 model in `opencode.json`. nemotron-omni is the
VLM variant of nemotron-30b — same memory footprint (19.6 GB / 4
safetensors). Already cached on disk, so no download cost.

## Result: model couldn't even start decoding

```
[21:11:21] Server ready after 30s
[21:11:21] Server reports served-model-name='nemotron-omni'
[21:11:22] Proxy ready, opencode config repointed to :23335
[21:11:22] Running SWE-bench Lite rollout...
Preflight: canary chat completion against http://127.0.0.1:23334 (model=nemotron-omni)...
  preflight OK (0B content)
[1/1] astropy__astropy-12907  repo=astropy/astropy  base=d16bfe05
  env: install FAILED — falling back to no-venv prompt
[21:11:44] CRASH — IndexError in mlx/model_runner.py:1081
```

Preflight passed (small canary prompt completed). The first real opencode
chat completion request triggered an MLX decode-batch bug:

```python
File "components/sglang/python/sglang/srt/hardware_backend/mlx/model_runner.py", line 1081, in decode_batch_start
    if is_full_attn[layer_idx]:
       ~~~~~~~~~~~~^^^^^^^^^^^
IndexError: list index out of range
```

NemotronH_Nano_Omni_Reasoning_V3 has an interleaved Mamba2 + Attention + MoE
layer structure. The `is_full_attn` array used in `decode_batch_start` is
sized for a homogeneous-attention layout (likely from the launching
`model_config`'s `num_hidden_layers`), so when the decode loop indexes
deeper layers, it goes past the array bounds.

This is **a real SGLang+MLX bug — not a model failure or quantization issue**.
A patch on `model_runner.py:decode_batch_start` to make `is_full_attn`
NemotronH-aware would unblock it (probably needs to consult the
NemotronH-specific layer-type map alongside the standard attention array).

## Complete int4 agentic verdict (closed set)

| Model | Verdict | Failure class |
|---|---|---|
| `qwen36` (MoE+DeltaNet+VL) | ✓ WORKS | — |
| `qwen35` (DeltaNet+VL) @ TIMEOUT=1800 | ✓ Slow but works | — |
| `qwen36-27b` (Dense+DeltaNet+VL) | ✗ | TIMEOUT — Dense too slow for the loop |
| `qwen3-32b` (Dense base) | ✗ | TIMEOUT — model can't converge |
| `qwen3-moe` (MoE base) | ✗ | Parser mismatch — `<\|name>...` tags |
| `coder-30b` (Coder MoE) | ✗ | Chat template gives up after 1 glob |
| `devstral` (Mistral) | ✗ | Preflight rejects canary structure |
| `gemma4-31b` (Gemma 4 Dense) | ✗ | 0 tokens emitted under tool prompts |
| `nemotron-30b` (NemotronH) | ✗ | TIMEOUT — model can't converge |
| `nemotron-omni` (NemotronH+VLM) | ✗ | **MLX backend bug** (`is_full_attn` index OOB) |

**10 of 10 alternatives are exhausted.** qwen36 is the only int4 agentic
model on this stack. The failure-mode taxonomy:

- **3 stack-level / parser issues** (qwen3-moe, devstral, nemotron-omni) —
  in principle fixable by chat-template/preflight/MLX patches
- **2 chat-template / model-side gaps** (gemma4-31b, coder-30b) — won't
  emit/engage; model-side, not fixable from infrastructure
- **4 capacity / decode-rate misses** (qwen36-27b, qwen3-32b, nemotron-30b,
  and what would qwen35 be at default TIMEOUT) — model can't drive the
  loop within reasonable wall time

**FP8/8-bit retry analysis (final):**

| Failure class | Would FP8 help? |
|---|:---:|
| Stack/parser bugs (3 models) | No — quantization-independent |
| Chat-template gaps (2 models) | No — chat template not quant-sensitive |
| Decode-rate misses (4 models) | No — wall time bound, more precision doesn't help |
| MLX backend bug (nemotron-omni) | No — code path bug at any precision |

**No FP8 download has a justified information value.** The qwen36
recommendation is final at the int4 level.

## Files

- `run.log` — smoke.sh stdout/stderr
- `launch.partial.log` — first 200 lines of sglang launch (config + boot)
- `crash.log` — the IndexError + SIGQUIT stack
