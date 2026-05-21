# Disabling chunked prefill BREAKS qwen36 + opencode flow (negative result)

## Hypothesis under test

Yesterday's `qwen36-django10914-N3-STABLE-RESOLVED-2026-05-21/perf-investigation.md`
hypothesized chunked-prefill scheduling as the run-to-run variance source
("cache state at chunk boundaries, attention-mask construction for
incremental prefills"). The natural experiment: disable chunking by
setting `--chunked-prefill-size 131072` (= full CTX, no chunking
needed) and see if outputs converge bit-identical.

## What happened: opencode gave up after step #1

| Metric | Value |
|---|---|
| Wall | 903.7 s (= TIMEOUT=900) |
| Diff bytes | 0 (empty) |
| Return code | 124 (SIGKILL) |
| Tool calls | 2 (`glob **/*.py`, `grep notes`) |
| Step finishes | 1 (reason=tool-calls) |
| Server POST count | 4 (health + /v1/models + 1 preflight + 1 real) |

The server itself was healthy throughout: prefill at 622-735 tok/s, decode
at 60 tok/s during the brief active window 02:11:42 - 02:12:13. After
opencode received the model's response (1 message with 2 embedded tool
calls, 67 output tokens), opencode executed `glob` and `grep` locally
**and then never made another /v1/chat/completions call**. The remaining
~14 minutes the server sat idle until the rollout hit timeout.

This is the same model + same opencode binary + same proxy as the
chunked-prefill=2048 runs that succeed. The only difference is the
chunked-prefill setting.

## Why this is surprising

- chunked-prefill=131072 ≡ no chunking (a full 131K context fits in one
  chunk). This should be a STRICT SIMPLIFICATION of the chunked-prefill=2048
  path: same input/output semantics, single big prefill instead of many
  small ones.
- The model successfully processed the 8.4K and 11.6K prompts (the two
  agentic-step contexts) — server-side fine.
- The model returned a valid response with tool calls — opencode-side
  fine through step #1.
- But step #2 never started.

## Possible explanations (not investigated)

1. **opencode tool execution hung** on filesystem op (`glob **/*.py` on
   the full django tree could enumerate 5K+ files; `grep notes *.py`
   over 5K files could take noticeable time). But 14 minutes is too long
   for an M4.
2. **Streaming response format differs** with chunked-prefill=131072 vs
   2048. Maybe the response is delivered as one big chunk that opencode
   parses differently than the 2048-chunked streaming form. Unlikely
   since the chat-completion endpoint isn't directly affected by prefill
   chunk size.
3. **Some kind of resource starvation** that doesn't trigger errors but
   prevents opencode from issuing the next request.

## Conclusion

**Don't use `--chunked-prefill-size 131072` for opencode agentic flows
on qwen36.** The current `--chunked-prefill-size 2048` in
`scripts/launch.sh` and `smoke.sh`'s `EXTRA_LAUNCH` default is load-
bearing for some reason. Without deeper instrumentation (opencode-side
trace, http2 frame logging) we can't tell why; but the empirical fact
is clear from this one run.

This is a negative result — the hypothesis "chunked prefill is the
variance source, disabling it stabilizes output" is REFUTED by the
broken agentic flow. The actual variance source remains unknown.

Future loop iterations should NOT repeat this experiment.

## Files

- `predictions.jsonl` — the empty rollout record
- `empty-prediction.diff` — 0-byte file (kept for completeness)
