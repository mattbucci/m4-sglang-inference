#!/usr/bin/env python3
"""Benchmark long-context performance at various context lengths.

Tests prefill speed and decode speed at different context sizes up to 256K.
Adjusted for 64GB unified memory on Apple M4 Pro.
"""

import argparse
import json
import sys
import time
import urllib.request


def bench_completions(base_url, input_len, output_len, label=""):
    """Run a single benchmark at given input/output lengths.

    Generates a prompt that tokenizes to approximately ``input_len`` tokens.
    Empirically the word "hello world test data benchmark context length
    padding " (= 8 words, 56 chars) tokenizes to ~10 tokens on Qwen / Gemma /
    Devstral tokenizers; an earlier `[:input_len * 4]` slice under-delivered
    by ~1.7x (e.g. requested 145K → got 84K). Using 6 chars/token as the
    inverse estimate plus a small safety margin gets us within ~5% of the
    requested length.
    """
    word = "hello world test data benchmark context length padding "
    chars_per_token = 6
    safety = 1.1
    target_chars = int(input_len * chars_per_token * safety)
    repeats = max(1, target_chars // len(word) + 1)
    prompt_text = (word * repeats)[:target_chars]

    body = json.dumps({
        "model": "default",
        "prompt": prompt_text,
        "max_tokens": output_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft = None
    chunk_times = []
    usage = {}
    # 1h per request — 128K+ on M4 with turboquant can take 25+ min just for
    # prefill at our observed throughput. The full sweep can take several
    # hours, which is the expected cost of a 256K characterization on Apple.
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                chunk_times.append(now)
    elapsed = time.time() - t0

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", len(chunk_times))
    depth_verified = prompt_tokens >= 0.95 * input_len

    # True decode rate: inter-token latencies past the first token. The old
    # whole-request metric is kept as amortized_s_per_token — at out<<prefill
    # it is ~95%+ prefill amortization, NOT decode speed.
    itls = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    itls_sorted = sorted(itls)
    itl_mean = sum(itls) / len(itls) if itls else 0
    itl_p50 = itls_sorted[len(itls) // 2] if itls else 0
    itl_p95 = itls_sorted[int(len(itls) * 0.95)] if itls else 0
    decode_tok_per_s = 1.0 / itl_mean if itl_mean > 0 else 0
    amortized = (elapsed / completion_tokens) if completion_tokens > 0 else 0

    print(f"  {label:30s}  in={prompt_tokens:>7d}{'' if depth_verified else '!'}  "
          f"out={completion_tokens:>4d}  ttft={ttft or 0:>7.1f}s  "
          f"itl_mean={itl_mean * 1000:>7.1f}ms  decode={decode_tok_per_s:>5.1f} tok/s  "
          f"amortized={amortized:>6.2f} s/tok")
    return {
        "prompt_tokens": prompt_tokens,
        "depth_verified": depth_verified,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "ttft_s": ttft,
        "itl_mean_ms": itl_mean * 1000,
        "itl_p50_ms": itl_p50 * 1000,
        "itl_p95_ms": itl_p95 * 1000,
        "decode_tok_per_s": decode_tok_per_s,
        "amortized_s_per_token": amortized,
        "n_itl_samples": len(itls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--output-tokens", type=int, default=64,
                        help="Number of output tokens per test")
    parser.add_argument("--contexts", type=int, nargs="+", default=None,
                        help="Override the default context-length list (256, 1K, …, 250K). "
                             "E.g. `--contexts 128 4096 16384` for a quick sweep.")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # Check health
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5):
            pass
    except Exception as e:
        print(f"Server not ready: {e}")
        sys.exit(1)

    # Get model info
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
            models = json.loads(r.read())
        model_id = models["data"][0]["id"].split("/")[-1]
    except:
        model_id = "unknown"

    out = args.output_tokens
    print(f"Model: {model_id}")
    print(f"Output tokens per test: {out}")
    print(f"{'Label':30s}  {'Input':>9s}  {'Out':>5s}  {'TTFT':>8s}  {'ITL':>9s}  {'Decode':>11s}  {'Amortized':>10s}")
    print("-" * 95)

    # Context lengths — push to the limits of 64GB unified memory by default;
    # accept a custom list when the caller just wants a short sweep.
    if args.contexts:
        def _label(n):
            if n >= 1024 * 1024:
                return f"{n // (1024*1024)}M tokens"
            if n >= 1024:
                return f"{n // 1024}K tokens"
            return f"{n} tokens"
        tests = [(n, _label(n)) for n in args.contexts]
    else:
        tests = [
            (256, "256 tokens (baseline)"),
            (1024, "1K tokens"),
            (4096, "4K tokens"),
            (8192, "8K tokens"),
            (16384, "16K tokens"),
            (32768, "32K tokens"),
            (65536, "64K tokens"),
            (131072, "128K tokens"),
            (200000, "200K tokens"),
            (250000, "250K tokens"),
        ]

    results = []
    for input_len, label in tests:
        try:
            r = bench_completions(base_url, input_len, out, label)
            results.append((label, r))
        except Exception as e:
            print(f"  {label:30s}  ERROR: {e}")
            results.append((label, None))

    # Summary
    print(f"\n{'='*95}")
    print(f"Summary: decode at depth (output={out} tokens; amortized = whole-request/out, ~all prefill at small out)")
    for label, r in results:
        if r:
            print(f"  {label:30s}  decode={r['decode_tok_per_s']:>5.1f} tok/s  "
                  f"itl p50/p95={r['itl_p50_ms']:.0f}/{r['itl_p95_ms']:.0f}ms  "
                  f"ttft={r['ttft_s'] or 0:.1f}s  amortized={r['amortized_s_per_token']:.2f} s/tok  "
                  f"(in={r['prompt_tokens']}{'' if r['depth_verified'] else ' UNVERIFIED'})")


if __name__ == "__main__":
    main()
