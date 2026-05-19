# gemma4 MoE on django-11001 @ TIMEOUT=1800 — falls back to conversational mode

## Task #78 — generalization test for the pylint-5859 RESOLVED win

The pylint-5859 success could be a one-off. Test gemma4 MoE on
django-11001 (different ecosystem, qwen36 RESOLVED with 1368-byte
patch) to see if the recommendation generalizes.

## Result: model gives up fast (different failure mode)

```
rc=0  elapsed=123s  diff=0 B  tool_calls=4 (all bash)
```

vs pylint-5859 which used 48 tool calls (35 bash, 6 read, 3 glob,
2 grep, 1 edit, 1 write) over 1324 s.

## Failure mode: conversational fallback

Reading the opencode session:

1. Model makes a few bash exploration calls
2. Writes a "Goal" reasoning block summarizing what needs fixing
3. **Writes "Continue if you have next steps, or stop and ask for
   clarification if you are unsure how to proceed."** — model is
   asking the user for direction rather than continuing autonomously
4. Then types out a reproduction test as TEXT (not via the write
   tool) — model output:

   ```
   I will now create a reproduction test case to confirm the issue.

   ```python
   import django
   from django.db import models
   ...
   ```

This is NOT the same model behavior as on pylint-5859. There, the
model used `write` to create files and `edit` to write the fix. Here,
the model is producing chat-style output and asking for clarification.

## What this means

**gemma4 MoE's agentic engagement is INCONSISTENT across instances.**
On simpler / smaller-codebase instances (pylint-5859), it executes
the full tool-call loop. On bigger codebases (django), the chat
template lapses into conversational mode after a few tool calls.

Possible cause: context-window dynamics. The opencode prompt for
django-11001 is bigger (Django's directory tree is larger). The
input tokens at step 1 were 8095 (vs ~3-4K for pylint), step 2 hit
19761. The model may be reverting to conversational behavior under
larger context.

This is a real recommendation refinement:
- **gemma4 MoE @ T=1800 is a possible agentic option, NOT a
  reliable one.**
- Works on simple-codebase instances (verified: pylint-5859)
- Falls back to conversational on larger-codebase instances
  (verified: django-11001)
- qwen36 remains the reliable primary

## Recommendation update

| Pick | Status | Notes |
|---|---|---|
| qwen36 | **Reliable** (verified primary) | 19/26 patch-engagement, 5/13 resolved |
| qwen35 (T=1800) | **Reliable but slow** | Equivalent capability to qwen36 |
| **gemma4 MoE (T=1800)** | **Conditionally works** | Verified on pylint-5859, fails on django-11001 — possibly context-dependent |

For users who specifically want Gemma 4 family agentic: try at T=1800
and accept that instance results may vary. Don't displace qwen36 as
the default.

## Files

- `run.log` — smoke.sh output
- `predictions.jsonl` — empty-patch record (0 bytes)
- `django__django-11001.log` — opencode session showing the
  conversational fallback
- `django__django-11001.env.log` — Django 3.6 venv setup (0 pkgs needed)
