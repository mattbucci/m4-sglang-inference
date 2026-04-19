#!/usr/bin/env python3
"""Attempt to reproduce the patch-001 radix cache scatter-write bug in isolation.

The production bug: in MlxKVPool.set_kv, scatter writes (`buf[slots] = src`)
appear not to commit. After the write, reading the buffer back through
get_kv returns wrong data — V buffer is all zeros, K buffer is half-norm.

This script tries to reproduce the bug standalone in a spawned subprocess
that mirrors SGLang's scheduler subprocess context (spawn start method,
mlx_lm model loaded, mxfp8 quantization, small V magnitudes). All of
these were tested 2026-04-18 and PASSED — i.e., the bug does NOT
reproduce in isolation. Kept here as a future-investigator starting
point.

Usage:
    python scripts/test/test_mlx_scatter_repro.py
"""
import multiprocessing as mp
import sys


def child_test() -> None:
    import mlx.core as mx
    print(f'MLX {mx.__version__}', flush=True)

    # Load the same model SGLang would load — Coder-30B 4-bit MLX.
    print('Loading mlx_lm Coder-30B-4bit...', flush=True)
    from mlx_lm import load as mlx_lm_load
    model, _ = mlx_lm_load('mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit')
    mx.eval(model.parameters())
    print(f'  loaded, mem={mx.get_active_memory()/(1024**3):.2f} GB', flush=True)

    # Pool buffer like MlxKVPool: (pool_size, n_kv_heads, head_dim_packed)
    pool_size, n_kv_heads, head_dim = 1000, 4, 128
    n = 6
    src_k = mx.random.normal((n, n_kv_heads, head_dim), dtype=mx.bfloat16) * 2.14  # K-magnitude
    src_v = mx.random.normal((n, n_kv_heads, head_dim), dtype=mx.bfloat16) * 0.0066  # tiny V like attention output
    print(f'  src K[0].norm={float(mx.linalg.norm(src_k[0]).item()):.4f} V[0].norm={float(mx.linalg.norm(src_v[0]).item()):.6f}', flush=True)

    # mxfp8 quantize like quantize_pool
    k_q, k_s = mx.quantize(src_k, mode='mxfp8')
    v_q, v_s = mx.quantize(src_v, mode='mxfp8')
    mx.eval(k_q, k_s, v_q, v_s)

    # Pool buffers
    k_buf_d = mx.zeros((pool_size, n_kv_heads, head_dim // 4), dtype=mx.uint32)
    k_buf_s = mx.zeros((pool_size, n_kv_heads, head_dim // 32), dtype=mx.uint8)
    v_buf_d = mx.zeros((pool_size, n_kv_heads, head_dim // 4), dtype=mx.uint32)
    v_buf_s = mx.zeros((pool_size, n_kv_heads, head_dim // 32), dtype=mx.uint8)
    slots = mx.array([1 + i for i in range(n)], dtype=mx.int32)

    # Scatter — the operation that fails in production
    k_buf_d[slots] = k_q
    k_buf_s[slots] = k_s
    v_buf_d[slots] = v_q
    v_buf_s[slots] = v_s
    mx.eval(k_buf_d, k_buf_s, v_buf_d, v_buf_s)

    # Read back slot 1 — should match src[0]
    s1 = mx.array([1], dtype=mx.int32)
    k_back = mx.dequantize(k_buf_d[s1], k_buf_s[s1], mode='mxfp8')
    v_back = mx.dequantize(v_buf_d[s1], v_buf_s[s1], mode='mxfp8')
    k_norm = float(mx.linalg.norm(k_back).item())
    v_norm = float(mx.linalg.norm(v_back).item())
    print(f'  read-back K.norm={k_norm:.4f} V.norm={v_norm:.6f}', flush=True)

    # Production bug: V.norm would be 0.0 and K.norm would be ~half src.
    expected_k = float(mx.linalg.norm(src_k[0]).item())
    expected_v = float(mx.linalg.norm(src_v[0]).item())
    k_loss = abs(k_norm - expected_k) / expected_k
    v_loss = abs(v_norm - expected_v) / max(expected_v, 1e-9)
    print(f'  K loss: {k_loss:.1%}  V loss: {v_loss:.1%}', flush=True)
    if k_loss > 0.20 or v_loss > 0.20 or v_norm < 1e-6:
        print('  REPRO: bug detected! K or V loss > 20%, or V is zero.', flush=True)
        sys.exit(1)
    else:
        print('  CLEAN: scatter roundtrip is normal (~2-3% loss). Bug does NOT reproduce here.', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=child_test)
    p.start()
    p.join(180)
    code = p.exitcode if p.exitcode is not None else 124
    print(f'\nchild exit={code}')
    sys.exit(code)
