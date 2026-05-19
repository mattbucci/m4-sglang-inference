# qwen3-moe parser investigation — model emits corrupted tags, not a parser-side fix

## Goal (Task #73 from MoE-rescue queue)

In the int4 bakeoff, `qwen3-moe` (Qwen3-30B-A3B-4bit-DWQ) emitted
`<|name>read>...{json}...</read>` tags that neither the configured
qwen25 parser nor qwen3_coder recognized. Hypothesis: a different
SGLang tool-call parser might handle the format, or a different system
prompt could nudge the model toward known tags.

## Root cause: model-output corruption, not parser selection

The exact emitted text (94 chars total — model produced this once then
stopped):

```
<|name>read> {"filePath":"/tmp/web-work/ast/179/ast/ast/sep.py","offset":0","limit":20}</read>
```

Two distinct problems visible:

1. **Special tokens are malformed.** The model's chat template
   (verified in `tokenizer_config.json`) defines the tool-call format
   as `<tool_call>\n{"name": "X", "arguments": {...}}\n</tool_call>`.
   The model instead emits `<|name>read>...</read>` — a half-formed
   special-token sequence that doesn't match its own training-time
   template. This is consistent with **special-token ID mis-mapping
   from the MLX 4-bit-DWQ conversion** — when the model "wants to
   emit `<tool_call>`" it produces a different token combination.

2. **The JSON itself is malformed.** Look at `"offset":0","limit":20`
   — there's a stray quote between `0` and `,`. Even if a hypothetical
   perfect parser could match the tags, the inner JSON wouldn't
   parse. The corruption isn't just at the token layer; it bleeds
   into the argument body.

3. **Hallucinated file path.** `/tmp/web-work/ast/179/ast/ast/sep.py`
   doesn't exist — the actual astropy work dir is
   `/tmp/swebench-work/astropy__astropy-12907/`. The model invented a
   plausible-looking path without first calling `glob`. This
   suggests the model isn't actually engaging with the agent loop —
   it's emitting one fabricated turn and stopping.

## Parser-selection won't fix this

The available SGLang detectors (Qwen25, Qwen3Coder, Hermes, Mistral,
Llama32, Pythonic, Gemma4, MiMo, DeepSeekV3*, GLM4*, Internlm,
Hunyuan, KimiK2, Lfm2, etc.) all match well-formed tag structures.
None of them would salvage half-tokens + malformed JSON.

## Diagnostic comparison with qwen36 (working MoE)

`qwen36` (Qwen3.6-35B-A3B-4bit-DWQ) shares MoE architecture, MLX-4bit-DWQ
quantization, and Qwen3-Coder parser configuration, yet emits clean
`<tool_call>{"name": "...", ...}</tool_call>` tags that parse to actual
tool calls and produce 506-byte patches. So the issue is specific to
**this particular checkpoint's quantization or training**, not to MoE
architecture or DWQ in general.

Possible avenues for future investigation (NOT pursued in this
iteration):
- Try the non-DWQ `mlx-community/Qwen3-30B-A3B-4bit` variant —
  though launch.sh notes it has dead layers
- Re-quantize from the original Qwen3-30B-A3B with `mlx_lm.convert`
  fresh
- Compare special-token IDs between qwen3-moe and qwen36 to find
  the mis-mapping

## Recommendation impact

**No change.** qwen3-moe stays in the "ruled out" column. The failure
mode upgrades from "parser mismatch (possibly fixable)" to "model
emits corrupted tool-call tags (not fixable at the harness layer)".

## Files

- `qwen3-moe-emission.log` — the 6-line opencode session log
  containing the model's single corrupted emission, preserved for
  forensics.
