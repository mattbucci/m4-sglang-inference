#!/usr/bin/env python3
"""Unified benchmark for all SGLang MLX models on Apple M4 Pro.

Uses sglang.bench_serving for proper TPOT/TTFT measurement (separates
prefill from decode).  Runs single-user context sweep + concurrency
throughput sweep.  Outputs structured JSON for chart generation.

Usage:
    python bench_all_unified.py --port 23334 --name "Model Name"
    python bench_all_unified.py --port 23334 --name "Model Name" --context-max 262144
    python bench_all_unified.py --port 23334 --name "Model Name" --kv-cache fp8
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests

# Force unbuffered stdout so progress is visible when piped
sys.stdout.reconfigure(line_buffering=True)


def run_bench_serving(base_url, model, input_len, output_len, num_prompts,
                      request_rate="inf", timeout=300):
    """Run sglang.bench_serving and parse TPOT/TTFT/throughput."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--random-input", str(input_len),
        "--random-output", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--disable-tqdm",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None

    def extract(pattern):
        m = re.search(pattern, output)
        return float(m.group(1)) if m else None

    tpot = extract(r"Mean TPOT[^:]*:\s*([\d.]+)")
    ttft = extract(r"Mean TTFT[^:]*:\s*([\d.]+)")
    throughput = extract(r"Output token throughput[^:]*:\s*([\d.]+)")

    if tpot is None and throughput is None:
        return None
    return {"tpot_ms": tpot, "ttft_ms": ttft, "throughput": throughput}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--name", required=True, help="Model display name")
    p.add_argument("--output", default=None)
    p.add_argument("--context-max", type=int, default=32768)
    p.add_argument("--concurrency-max", type=int, default=16)
    p.add_argument("--kv-cache", default="auto",
                   help="KV cache mode (auto/fp8/turboquant) for results metadata")
    p.add_argument("--skip-charts", action="store_true",
                   help="Skip chart regeneration (caller will do it)")
    args = p.parse_args()

    base_url = f"http://localhost:{args.port}"

    # Verify server
    try:
        requests.get(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"Server not responding on port {args.port}")
        sys.exit(1)

    # Auto-detect model ID
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        model = r.json()["data"][0]["id"]
    except Exception:
        model = "unknown"

    print(f"=== {args.name} ===")
    print(f"Model: {model}")
    print(f"Port: {args.port}")
    print()

    # Warmup (compile MLX kernels)
    print("Warming up (3 requests)...")
    for _ in range(3):
        try:
            requests.post(f"{base_url}/v1/chat/completions", json={
                "model": "model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5, "temperature": 0,
            }, timeout=120)
        except Exception:
            pass
    print()

    # --- Context sweep (single user, 64 output tokens) ---
    ctx_levels = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                  65536, 131072, 262144]
    ctx_levels = [c for c in ctx_levels if c <= args.context_max]

    print(f"--- Context sweep (single user, 64 output tokens, sglang.bench_serving) ---")
    print(f"{'Context':>8}  {'TPOT(ms)':>10}  {'tok/s':>8}  {'TTFT(ms)':>10}")
    print("-" * 50)

    context_results = []
    for ctx in ctx_levels:
        # Scale timeout for long context — prefill is O(n^2) in attention.
        # 32K: ~4 min prefill on dense model, 64K: ~12 min, 128K: ~30 min, 256K: ~60 min
        bench_timeout = 600 if ctx <= 32768 else 1800 if ctx <= 65536 else 3600
        r = run_bench_serving(base_url, model, ctx, 64, 1,
                              request_rate=1, timeout=bench_timeout)
        if r is None:
            print(f"{ctx:>8}  {'ERR':>10}  {'—':>8}  {'—':>10}")
            context_results.append({"context": ctx, "error": "timeout_or_crash"})
            # Check if server is truly dead (use longer timeout since it
            # might be mid-prefill and temporarily unresponsive)
            server_dead = True
            for attempt in range(3):
                try:
                    requests.get(f"{base_url}/health", timeout=30)
                    server_dead = False
                    break
                except Exception:
                    pass
            if server_dead:
                print("  Server appears down, stopping context sweep")
                break
            print("  Server still alive, continuing...")
            continue

        tpot = r["tpot_ms"] or 0
        tps = 1000.0 / tpot if tpot > 0 else 0
        ttft = r["ttft_ms"] or 0
        print(f"{ctx:>8}  {tpot:>10.1f}  {tps:>8.1f}  {ttft:>10.1f}")
        context_results.append({
            "context": ctx,
            "tpot_ms": round(tpot, 1),
            "tok_per_sec": round(tps, 1),
            "ttft_ms": round(ttft, 1),
        })

    # --- Concurrency sweep (256 in / 256 out) ---
    # Skip if server died during context sweep
    server_alive = True
    try:
        requests.get(f"{base_url}/health", timeout=5)
    except Exception:
        server_alive = False

    conc_levels = [1, 2, 4, 8, 16]
    conc_levels = [c for c in conc_levels if c <= args.concurrency_max]

    print()
    if not server_alive:
        print("Server is down — skipping concurrency sweep")
        conc_levels = []
    print(f"--- Concurrency sweep (256 in / 256 out, sglang.bench_serving) ---")
    print(f"{'Conc':>5}  {'TPOT(ms)':>10}  {'tok/s':>8}  {'TTFT(ms)':>10}")
    print("-" * 45)

    throughput_results = []
    for conc in conc_levels:
        np = max(conc * 4, 4)
        rr = 1 if conc == 1 else "inf"
        r = run_bench_serving(base_url, model, 256, 256, np,
                              request_rate=rr, timeout=600)
        if r is None:
            print(f"{conc:>5}  {'ERR':>10}  {'—':>8}  {'—':>10}")
            throughput_results.append({
                "concurrency": conc, "error": "timeout_or_crash"
            })
            try:
                requests.get(f"{base_url}/health", timeout=5)
            except Exception:
                print("  Server appears down, stopping concurrency sweep")
                break
            continue

        tpot = r["tpot_ms"] or 0
        tp = r["throughput"] or 0
        ttft = r["ttft_ms"] or 0
        print(f"{conc:>5}  {tpot:>10.1f}  {tp:>8.1f}  {ttft:>10.1f}")
        throughput_results.append({
            "concurrency": conc,
            "tpot_ms": round(tpot, 1),
            "tok_per_sec": round(tp, 1),
            "ttft_ms": round(ttft, 1),
        })

    # Save results
    all_results = {
        "model": args.name,
        "model_id": model,
        "engine": "SGLang + MLX",
        "hardware": "Apple M4 Pro 64GB",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "sglang.bench_serving",
        "kv_cache_mode": args.kv_cache,
        "context_max": args.context_max,
        "context_sweep": context_results,
        "throughput_sweep": throughput_results,
    }

    safe_name = args.name.replace(" ", "_").lower()
    out_path = args.output or f"benchmarks/{safe_name}/results.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Regenerate charts (unless caller handles it)
    if not args.skip_charts:
        chart_script = os.path.join(os.path.dirname(__file__), "generate_charts.py")
        if os.path.exists(chart_script):
            print("\nRegenerating benchmark charts...")
            subprocess.run([sys.executable, chart_script], check=False)


if __name__ == "__main__":
    main()
