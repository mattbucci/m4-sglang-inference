# Qwopus3.8-27B-Coder-Flash (in-house MLX 4-bit), sealed harness, opencode limit 131K — 3/3 patches on the canonical file

`~/AI/models/Qwopus3.8-27B-Coder-Flash-4bit-inhouse` (published as
`mattbucci/Qwopus3.8-27B-Coder-Flash-MLX-4bit`; the in-house 4-bit of
`Jackrong/Qwopus3.8-27B-Coder-Flash`) served by the `qwen36-27b` preset
(`MODEL=` override, CTX=131072, `CHUNKED=2048`, mem-fraction 0.5, radix on,
thinking always on) on SGLang v0.5.23 + the 17 patches, through the sealed
harness: `webfetch`/`websearch`/`task` off, opencode and its bash children in
the localhost-only sandbox, ancestor-only checkout. opencode `limit.context`
131072 for the `qwen36-27b` key, 1800 s per instance. OOM guard: no warnings.
No compaction in any session.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Target file |
|---|----------|-----:|-----:|------:|-------|----------:|-------------|
| 1 | `astropy__astropy-12907` | 238 s | 504 B | 3 | read 1, edit 1 | 17.7K tok | `astropy/modeling/separable.py` |
| 2 | `astropy__astropy-14182` | 1272 s | 543 B | 55 | bash 33, read 21, edit 3, glob 1, write 1 | 68.7K tok | `astropy/io/ascii/rst.py` |
| 3 | `astropy__astropy-14365` | 383 s | 617 B | 8 | bash 5, read 3, glob 1, edit 1 | 28.8K tok | `astropy/io/ascii/qdp.py` |

- 12907: `_cstack` copies the right operand's matrix (`= right`) — the
  upstream fix, found in one read and one edit.
- 14182: `RST.__init__(self, *args, **kwargs)` passes everything through to
  `FixedWidth` — the upstream `header_rows` fix in a more general form; the
  21-minute session spent its steps running the test suite with bash.
- 14365: `_line_type_re` compiled with `re.IGNORECASE` — the regex half of
  the upstream fix (the stock Qwen3.8-27B run also lower-cased the `NO`
  mask comparison).
- `audit_rollouts.py`: no web, network or history hit in the three sessions.

Same three instances as `../qwen38-27b-community-sealed-131k-2026-11-01/`
(stock Qwen3.8-27B-4bit, 3/3, 495 / 397 / 224 s).

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log` (local only)
