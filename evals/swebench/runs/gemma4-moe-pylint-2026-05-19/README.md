# gemma4 MoE produces its first M4 agentic patch (engages-but-no-synthesis)

## Task #76 — easier-instance follow-up to patch 020

After patch 020 unblocked gemma4 MoE from stack-level crashes, the
astropy-12907 run showed 4 tool calls but no edit. Hypothesis: that
instance is known-hard (smaller models also fail there). Test on a
SIMPLER instance — pylint-5859 — where qwen36 cleanly RESOLVED with a
656-byte patch.

## Result

```
rc=124 (TIMEOUT) elapsed=904s diff=170 B  tool_calls=43
  36 bash, 3 read, 2 glob, 2 grep
```

**This is the first non-Qwen M4 agentic patch.** 43 tool calls is
the most exploration any non-Qwen model has done in our test set.

Score: **F2P 0/1, P2P 10/10, applied=True**. The patch applies
cleanly, breaks no existing tests, but doesn't solve the F2P case.

## What gemma4 actually wrote

```diff
diff --git a/reproduction.py b/reproduction.py
new file mode 100644
+++ b/reproduction.py
@@ -0,0 +1,2 @@
+# YES: yes
+# ???: no
```

The model created a NEW reproduction file (not modified the
canonical pylint source). pylint-5859 is about pylint's fixme-note
regex matching `# YES: yes` when it shouldn't. So the model created
a test case to reproduce the bug — an intelligent first step — but
ran out of TIMEOUT=900s before writing the actual fix to
`pylint/checkers/misc.py` (where the gold patch lives).

## Comparison with the gold patch

```diff
diff --git a/pylint/checkers/misc.py b/pylint/checkers/misc.py
@@ -121,9 +121,9 @@ def open(self):
-            regex_string = rf"#\s*({notes}|{self.config.notes_rgx})\b"
+            regex_string = rf"#\s*({notes}|{self.config.notes_rgx})(?=(:|\s|\Z))"
-            regex_string = rf"#\s*({notes})\b"
+            regex_string = rf"#\s*({notes})(?=(:|\s|\Z))"
```

The gold fix changes the regex from `\b` (word boundary, which
matches at end of `YES:` between `S` and `:`) to a lookahead for
`:|\s|\Z` (only match when followed by `:`, whitespace, or end of
string). 2-line surgical fix.

The model's approach (build reproduction → study → fix) is sound but
slow. With more time it might have written the actual fix.

## Recommendation impact

**No recommendation change to "primary" set** — qwen36 still primary
because gemma4 MoE doesn't produce passing patches.

**But meaningful taxonomy refinement**: gemma4 MoE moves from
"engages, can't synthesize on hard instances" to "engages with rich
exploration (43 tool calls), creates reproduction files, but runs
out of time before writing the canonical fix even on simpler
instances." The model HAS agentic intent; it just doesn't have
enough decode budget within 15-minute windows.

A TIMEOUT=1800 retry might unlock a successful patch. Filed as a
potential follow-up.

## Files

- `run.log` — smoke.sh output
- `predictions.jsonl` — record with the 170-byte reproduction-file diff
- `pylint-dev__pylint-5859.diff` — extracted diff
- `pylint-dev__pylint-5859.log` — opencode session with 43 tool calls
- `pylint-dev__pylint-5859.env.log` — venv setup (succeeded — pylint installed)
- `score.jsonl` — local PASS/FAIL result (F2P 0/1, P2P 10/10)
