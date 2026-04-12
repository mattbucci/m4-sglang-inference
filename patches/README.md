# Patches

Patches applied on top of SGLang `main` branch for MLX backend fixes.

## 001-mlx-request-cleanup (PR #22632)

**Source:** https://github.com/sgl-project/sglang/pull/22632
**Status:** Open (not yet merged upstream)

Fixes premature request-state cleanup in the MLX backend that causes `KeyError`
during concurrent multi-request decoding. The scheduler can legitimately exclude
a live request from an intermediate batch, but the old code deleted its MLX state
when it was absent — causing a crash when the request reappeared.

**Fix:** Replace implicit batch-membership-based cleanup with explicit lifecycle-based
cleanup. Adds `cleanup_requests()` and `clear_runtime_state()` hooks to `TpModelWorker`,
wired to scheduler `flush_cache()`, `abort_request()`, and request-finished events.

**Files changed:** 5 files, +215 / -8
- `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/managers/tp_worker.py`
- `test/registered/core/test_mlx_tp_worker.py` (new)

### Applying

```bash
cd components/sglang
git apply ../../patches/001-mlx-request-cleanup.patch
```

Or use `scripts/setup.sh` which applies all patches automatically.
