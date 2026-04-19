#!/usr/bin/env python3
"""Capability validator — runs against a live SGLang/MLX server on <port>.

Adapted from the R9700 sister repo. Catches the silent regressions we keep
finding when calibrating or upgrading models:

  1. Basic   — short-answer sanity check (catches BOS doubling, <unk>/<pad> output)
  2. Thinking — model produces <think>...</think> AND terminates before max_tokens
                (Qwen3 family on greedy MLX is prone to infinite think loops)
  3. Vision   — model describes a synthetic image (skipped by default on MLX —
                MLX backend currently disables VLM; see patch 002)

Usage:
    # Launch your server, then:
    python scripts/eval/validate_capabilities.py --port 23334
    python scripts/eval/validate_capabilities.py --port 23334 --include-vision
    python scripts/eval/validate_capabilities.py --port 23334 --thinking-kwarg '{"enable_thinking":true}'

Exit 0 if all enabled checks pass, non-zero otherwise. Run after every model
launch / preset change / chat template update.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request


def _http_post(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8").strip()
        return json.loads(body) if body else {}


def _server_alive(base_url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _make_test_image() -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.stderr.write("PIL required for vision validation; pip install pillow\n")
        raise
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((64, 64, 192, 192), fill="red", outline="black", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def check_thinking(
    base_url: str,
    model: str,
    thinking_kwargs: dict | None,
    max_tokens: int = 2048,
) -> tuple[bool, str]:
    """Reasoning prompt — verify <think>...</think> structure + clean termination.

    On MLX (greedy-only), Qwen3 family is at high risk of looping `</think>\\nX\\n</think>`
    indefinitely because greedy decode hits the same token repeatedly. A finish_reason
    of "length" (truncated) is treated as a FAIL — it's the loop signature.
    """
    prompt = (
        "A ball and a bat cost $1.10 together. The bat costs $1.00 more than "
        "the ball. How much does the ball cost? Put the final numeric answer "
        "alone on the last line."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "skip_special_tokens": False,
    }
    if thinking_kwargs:
        payload["chat_template_kwargs"] = thinking_kwargs

    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=300)
    except Exception as e:
        return False, f"request failed: {e!r}"

    choice = r["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    usage = r.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)

    has_reasoning = bool(reasoning) or "<think>" in content or "<|channel>" in content
    closed = (
        bool(reasoning and content)
        or "</think>" in content
        or "<channel|>" in content
    )
    truncated = finish == "length"

    after_think = content
    if "</think>" in after_think:
        after_think = after_think.split("</think>")[-1]
    if "<channel|>" in after_think:
        after_think = after_think.split("<channel|>")[-1]
    answer_correct = bool(re.search(r"\$?0?\.05\b|\b5\s*cents?\b", after_think.lower()))

    # Detect repetition loop signature: same line repeated 5+ times
    lines = [l.strip() for l in (after_think or content).splitlines() if l.strip()]
    looped = False
    if lines:
        most_common = max(set(lines), key=lines.count)
        looped = lines.count(most_common) >= 5

    status = []
    if has_reasoning: status.append("reasoning_seen")
    if closed: status.append("terminated")
    if answer_correct: status.append("answer_ok")
    if truncated: status.append("TRUNCATED")
    if looped: status.append("LOOP_DETECTED")

    # Grading:
    #  - Thinking model: must produce <think>, close it, terminate cleanly, not loop
    #  - Non-thinking model (no markers): just needs to answer correctly + terminate
    # Both modes hard-fail on truncation or loop signature.
    if truncated or looped:
        passed = False
    elif has_reasoning:
        passed = closed and answer_correct
    else:
        passed = answer_correct
        status.append("non_thinking_model")

    msg = f"{' '.join(status) or 'no_markers':45s} ({completion_tokens} tok, finish={finish})"
    return passed, msg


def check_basic(base_url: str, model: str) -> tuple[bool, str]:
    """Short factual question — verifies the server / chat template at all."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 256,
        "temperature": 0,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=120)
    except Exception as e:
        return False, f"request failed: {e!r}"
    msg_choice = r["choices"][0]["message"]
    content = (msg_choice.get("content") or "").lower()
    reasoning = (msg_choice.get("reasoning_content") or "").lower()
    finish = r["choices"][0].get("finish_reason")

    passed = "paris" in content or "paris" in reasoning
    if "<unk>" in content or "\ufffd" in content:
        passed = False  # template/BOS bug
    if "<pad>" in content:
        passed = False  # gemma4-style pad emission
    sample = content[:60] if content else f"(reasoning){reasoning[:60]}"
    return passed, f"finish={finish} answer={sample!r}"


def check_vision(base_url: str, model: str) -> tuple[bool, str]:
    """Synthetic red circle on white. Currently most MLX models won't pass this."""
    img_bytes = _make_test_image()
    b64 = base64.b64encode(img_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=180)
    except Exception as e:
        return False, f"request failed: {e!r}"
    content = (r["choices"][0]["message"].get("content") or "").lower()
    expected = ["red", "circle", "round", "sphere", "ball", "dot", "disk", "oval"]
    hits = [w for w in expected if w in content]
    passed = len(hits) >= 1
    return passed, f"saw={hits}  response={content[:120]!r}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--host", default="localhost")
    p.add_argument("--model", default=None, help="Override model name (default: server-reported)")
    p.add_argument("--skip-thinking", action="store_true")
    p.add_argument("--include-vision", action="store_true",
                   help="Run vision check (off by default — MLX backend disables VLMs)")
    p.add_argument("--thinking-kwarg", default=None,
                   help='JSON string, e.g. \'{"enable_thinking": true}\' for Gemma4')
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    if not _server_alive(base):
        sys.stderr.write(f"Server at {base} not responding\n")
        return 2

    if args.model:
        model = args.model
    else:
        try:
            models = _http_get(f"{base}/v1/models", timeout=5)
            model = models["data"][0]["id"]
        except Exception:
            model = "default"

    thinking_kwargs = json.loads(args.thinking_kwarg) if args.thinking_kwarg else None

    print(f"=== M4 capability validator — {base}  model={model} ===")

    results: list[tuple[str, bool, str]] = []
    t0 = time.time()

    ok, msg = check_basic(base, model)
    results.append(("basic", ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] basic     {msg}")

    if not args.skip_thinking:
        ok, msg = check_thinking(base, model, thinking_kwargs)
        results.append(("thinking", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] thinking  {msg}")

    if args.include_vision:
        ok, msg = check_vision(base, model)
        results.append(("vision", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] vision    {msg}")

    elapsed = time.time() - t0
    print(f"--- {sum(ok for _, ok, _ in results)}/{len(results)} passed in {elapsed:.1f}s ---")

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
