# nemotron-omni after MLX patch — IndexError fixed, model itself still doesn't engage

## Task #74 from MoE-rescue queue

The prior `nemotron-omni-int4-2026-05-18` run crashed SGLang at the
first decode-batch with `IndexError: list index out of range` at
`model_runner.py:1081` (`is_full_attn[layer_idx]`). NemotronH's
`make_cache()` only emits caches for `M` (Mamba2) and `*` (Attention)
layers — `MoE` ("E") and `MLP` ("-") layers get nothing, so
`len(sample_cache) < num_layers`. The batched hybrid path indexed
`is_full_attn` by `layer_idx` and ran off the end of the array.

## Patch landed

`components/sglang/python/sglang/srt/hardware_backend/mlx/model_runner.py`:

```python
# Detect partial-cache layouts (NemotronH-style)
cache_layer_mismatch = len(sample_cache) != num_layers

if full_attn_idx is None or cache_layer_mismatch:
    # Pure-linear (no attention) OR partial-cache (NemotronH).
    # Either way the batched path's layer_idx indexing breaks;
    # fall back to serial-per-request decode. The model's own
    # forward() handles the cache-position vs layer-index
    # mapping internally.
    per_req_logits = []
    for i, rid in enumerate(req_ids):
        ...
```

Minimal, safe extension of the existing "pure-linear fallback" path —
no new code path, just a wider trigger condition.

## Retest result

```
[01:46:33] Server ready (no crash)
[01:46:35] opencode rollout starts
[01:46:57] Server: prefill 2048+2048+473 tokens → 200 OK
[01:46:58] Server: decode at 1.21 tok/s
[01:46:58] opencode terminates rc=0 in 20.5s
```

| Metric | Value |
|---|---|
| Server crash | **NO** ✓ (patch successful) |
| Wall | 20.5 s |
| Patch | 0 B |
| Tool calls | 0 |
| Model output | **23 tokens** then `reason: stop` |
| New failure class | Chat-template gap (model immediately says "done") |

The MLX backend bug is fixed — server decoded the full prefill +
started decode without crashing. But nemotron-omni's chat template
doesn't engage opencode's tool-call protocol. Model emits 23 output
tokens and signals stop. Same class as coder-30b ("1 glob then asks
user") and devstral ("21 KB of text, no tool calls") — the model
side is the limit now, not the harness.

## Recommendation impact

**No recommendation change** — nemotron-omni still doesn't work for
agentic on M4.

**But the MLX patch is keep-worthy**:
- Fixes a real stack-level crash that prevented NemotronH-family
  models from doing ANY batched decode
- Future NemotronH-family checkpoints (or a re-trained tool-call
  variant) could now drive opencode without crashing the server
- Single-instance decode (`batch_size==1` fast path) was already
  fine; this fixes `batch_size > 1` which the SGLang scheduler can
  trigger when 2+ requests arrive close together (e.g., preflight
  canary + first opencode request)

## Updated failure-mode taxonomy (NemotronH family)

| Model | Stack-level | Model-side |
|---|---|---|
| `nemotron-30b` (NemotronH text) | ✓ Patched | TIMEOUT — can't converge on long tasks |
| `nemotron-omni` (NemotronH+VLM) | ✓ Patched | Chat template gives up (23 tokens, stop) |

Both NemotronH variants can now decode without crashing; neither can
drive the agent loop. Model-side limitations remain.

## Files

- `run.log` — smoke.sh stdout
- `predictions.jsonl` — empty-patch record
- `astropy__astropy-12907.log` — opencode session (step_start + step_finish, 0 tool calls)
- `astropy__astropy-12907.env.log` — astropy venv install failure (irrelevant)
