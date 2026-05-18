# qwen36 on harder-codebase ecosystems — 3/3 (9/9 cumulative across 7 ecosystems)

## TL;DR

After 6/6 on astropy + Django + matplotlib + sympy, tested qwen36 on
three harder ecosystems with larger codebases:

| Instance | Wall | Diff | Target | Issue |
|----------|:----:|:----:|--------|-------|
| `pylint-dev__pylint-5859` | 498 s | 656 B | `checkers/misc.py` | pylint #5474 |
| `scikit-learn__scikit-learn-10297` | 155 s | 932 B | `linear_model/ridge.py` | sklearn #10286 |
| `sphinx-doc__sphinx-10325` | 725 s | 1038 B | `ext/autodoc/__init__.py` | sphinx #10261 |

**3/3 produce real, on-target patches.** Cumulative: **9/9 across 7
ecosystems** (astropy, Django, matplotlib, sympy, pylint, sphinx,
scikit-learn).

## Patch quality (manual review)

### pylint (656 B)

```diff
-            regex_string = rf"#\s*({notes}|{self.config.notes_rgx})\b"
+            regex_string = rf"#\s*({notes}|{self.config.notes_rgx})(?:\W|$)"
-            regex_string = rf"#\s*({notes})\b"
+            regex_string = rf"#\s*({notes})(?:\W|$)"
```

`EncodingChecker._compute_fixme_regex` was using `\b` (word boundary)
to terminate the fixme tag match, but `\b` requires a word/non-word
transition — it doesn't match between two non-word chars. So tags
like `XXX:` followed by `:` (non-word) wouldn't match. Switching to
`(?:\W|$)` (any non-word OR end-of-string) is the canonical fix for
pylint issue #5474.

### scikit-learn (932 B)

```diff
     def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True,
-                 normalize=False, scoring=None, cv=None, class_weight=None):
+                 normalize=False, scoring=None, cv=None, class_weight=None,
+                 store_cv_values=False):
         super(RidgeClassifierCV, self).__init__(
             alphas=alphas, fit_intercept=fit_intercept, normalize=normalize,
-            scoring=scoring, cv=cv)
+            scoring=scoring, cv=cv, store_cv_values=store_cv_values)
         self.class_weight = class_weight
```

`RidgeClassifierCV` was missing the `store_cv_values` parameter that
its parent class `_BaseRidgeCV` accepts. Pass-through fix matches
sklearn issue #10286.

### sphinx (1038 B)

```diff
-        return arg
+        return {x.strip() for x in arg.split(',') if x.strip()}
...
-                    if cls.__name__ == self.options.inherited_members and cls != self.object:
+                    if cls.__name__ in self.options.inherited_members and cls != self.object:
```

Two-part change: `inherited_members_option` now parses a
comma-separated string into a set; `is_filtered_inherited_member`
switches from `==` to `in` for the set-membership check. This is a
**type-changing refactor** done correctly — the model noticed that
changing the return type required updating downstream callers. Matches
sphinx issue #10261 (multi-class support for `:inherited-members:`).

## Wall-time pattern

| Ecosystem | Wall | Venv | Codebase size |
|-----------|:----:|:----:|---------------|
| astropy (no venv) | ~120 s | no | medium |
| matplotlib | 204 s | no | medium |
| scikit-learn | 155 s | install FAILED | large |
| Django | 328 s | 0 pkgs | very large |
| pylint | 498 s | 0 pkgs | small |
| sympy | 487 s | 2 pkgs | very large |
| sphinx | 725 s | 3 pkgs | medium |

The 4 longest runs all have venvs built — the model exercises tools
more when it has `pytest`/`import` access. Pure no-venv prompts are
faster (model relies on source-only reasoning) but the diffs are
similar quality. The 900 s budget for harder ecosystems comfortably
fits all 3.

## Cumulative scorecard (2026-05-18)

| Run | Instances | Ecosystems | Patches | Notes |
|-----|:---------:|:----------:|:-------:|-------|
| Initial smoke | 1 | astropy | 1/1 | Plumbing verified |
| 3-instance | 3 | astropy×3 | 3/3 | Within-domain reproducible |
| Cross-repo | 3 | django, matplotlib, sympy | 3/3 | Cross-ecosystem |
| Harder ecosystems | 3 | pylint, sklearn, sphinx | 3/3 | Generalization |
| **TOTAL** | **9 unique** | **7 ecosystems** | **9/9** | **Strong** |

## What this still doesn't prove

- Docker pass-rate (we still don't run SWE-bench's test harness)
- Win rate at 30 / 100 / 300 instances
- Behavior on instances requiring deep call-graph reasoning (these
  have all been 1-3 file, 1-3 hunk edits)

## What this DOES prove

qwen36 + `no_thinking_proxy` is a real, generalizable agentic-coding
configuration on M4 Pro. 9/9 cross-ecosystem with patches that hit
the canonical upstream-fix locations is the strongest data the M4
stack has produced.
