# qwen36 cross-ecosystem reproducibility — 3/3 on Django + matplotlib + sympy

## TL;DR

After 3/3 on instances 1-3 (all astropy), tested qwen36 on one instance
each from **Django, matplotlib, and sympy** — the three biggest SWE-bench
Lite ecosystems by count after astropy. **3/3 produced real,
domain-appropriate patches.** Combined with the prior run, qwen36's
agentic-coding success rate is now **6/6 across 4 ecosystems**.

The recommendation generalizes off astropy.

## Result

| Instance | Repo | Wall | Diff bytes | Description |
|----------|------|:----:|:----------:|-------------|
| `django__django-10914` | django/django | 328 s | 2563 | `FILE_UPLOAD_PERMISSIONS = None → 0o644` + docs |
| `matplotlib__matplotlib-18869` | matplotlib/matplotlib | 204 s | 758 | Add `__version_info__` tuple parsing |
| `sympy__sympy-11400` | sympy/sympy | 487 s | 704 | Add `_print_sinc` (Piecewise) to ccode.py |

## Patch quality (manual review)

### Django (2563 B, multi-file)

```diff
-FILE_UPLOAD_PERMISSIONS = None
+FILE_UPLOAD_PERMISSIONS = 0o644
```

Matches Django issue #28644 — file upload mode inconsistency when
`FILE_UPLOAD_PERMISSIONS` is `None` and the temp file is on the same
filesystem as the destination. The canonical fix is exactly this:
default to `0o644`. The model also updated `docs/ref/settings.txt` to
match — multi-file editing.

### matplotlib (758 B)

```python
if name == "__version_info__":
    import re
    parts = re.split(r'[.\-+]', _version.version)
    return tuple(int(p) if p.isdigit() else p for p in parts[:3])
```

Matches matplotlib issue #17034 — adds the `__version_info__` tuple
attribute alongside `__version__`. Implemented via `module.__getattr__`
override (PEP 562 style). Clean.

### sympy (704 B)

```python
def _print_sinc(self, func):
    from sympy import Piecewise, Ne
    from sympy.functions.elementary.trigonometric import sin
    x = func.args[0]
    return self._print(Piecewise((sin(x)/x, Ne(x, S.Zero)), (S.One, True)))
```

Matches sympy issue #11400 — C code generation for `sinc` was missing.
Using `Piecewise` to handle the `sinc(0) = 1` removable singularity is
the standard treatment.

## Wall-time observations

- Astropy instances: ~120 s (smallest, no venv ever, fastest)
- Django: 328 s (with built venv `3.6, 0 pkgs`)
- matplotlib: 204 s (no venv)
- sympy: 487 s (with built venv `3.9, 2 pkgs`)

The pattern: repos with venv-built dependencies allow the model to
exercise tools more (verify imports, etc), which takes longer but
produces more substantial work. Pure-no-venv runs are faster but the
model leans on source inspection only.

Sympy at 487 s is approaching the 600 s timeout — close call. For
sympy / large-Django instances, consider `TIMEOUT=900` or `TIMEOUT=1200`
to give the model headroom.

## Cumulative scorecard (2026-05-18)

| Run | Instances | Repos | Patches | Notes |
|-----|:---------:|-------|:-------:|-------|
| Original smoke | 1 (instance 1) | astropy | 1/1 | Confirmed plumbing works |
| 3-instance reproducibility | 3 (1-3) | astropy×3 | 3/3 | Confirmed within-domain |
| Cross-ecosystem | 3 (Django/matplotlib/sympy) | django, matplotlib, sympy | 3/3 | **Confirmed generalization** |
| **Total** | **6 unique** | **4 ecosystems** | **6/6** | **Recommendation durable** |

## What this still does NOT prove

- That SWE-bench's official Docker test harness would pass these patches.
  Patches LOOK right and target the canonical bug locations / fixes from
  the issue reports, but cross-stack scoring via the 3090 stack's
  `score_docker.py` is the rigorous next step.
- That harder instances (pylint, sphinx, scikit-learn — appearing later
  in the dataset) also yield patches.
- That the win rate is maintained at 30 / 100 / 300 instances.

But: at 6/6 with this distribution, the qwen36-as-agentic-lead
recommendation is the strongest data the M4 stack has produced.

## Next experiments

1. Push `predictions.jsonl` to the 3090 stack for Docker scoring
2. Run instances 4-10 (more astropy + first wave of cross-repo)
3. Try gemma4-31b on `astropy__astropy-12907` now that `tool_call: true`
4. Try qwen35 on the same instance with `TIMEOUT=1800`
