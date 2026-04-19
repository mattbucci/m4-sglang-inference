#!/usr/bin/env python3
"""Regression test for the patch-001 radix cache repeat-prompt bug.

Discovered 2026-04-18 against `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`:
identical prompts re-sent in series return deterministic garbage on the 2nd+ call
(see project_radix_cache_repeat_bug memory + README "Known Issues").

This script sends the same simple factual prompt N times and verifies all answers
contain the expected substring. Exit 0 if all match, 1 if drift detected.

Usage:
    # Against a running server (any preset):
    python scripts/eval/test_radix_cache_repeat.py --port 23334
    python scripts/eval/test_radix_cache_repeat.py --port 23334 --iters 10
"""
import argparse
import json
import sys
import urllib.request


def _post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _flush(base_url):
    try:
        _post(f"{base_url}/flush_cache", {}, timeout=10)
    except Exception:
        pass  # endpoint may not exist on all builds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--host", default="localhost")
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--prompt", default="What is the capital of Japan? Answer in one word.")
    p.add_argument("--expect", default="tokyo", help="Case-insensitive substring all responses must contain")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    chat = f"{base}/v1/chat/completions"

    print(f"=== radix cache repeat-prompt test ===")
    print(f"  prompt: {args.prompt!r}")
    print(f"  expect: {args.expect!r}  iters: {args.iters}")
    _flush(base)

    fails = []
    for i in range(args.iters):
        try:
            r = _post(chat, {
                "model": "default",
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens, "temperature": 0,
            })
            content = (r["choices"][0]["message"]["content"] or "").strip()
            finish = r["choices"][0]["finish_reason"]
            tok = r["usage"]["completion_tokens"]
            ok = args.expect.lower() in content.lower() and finish == "stop"
            tag = "OK" if ok else "FAIL"
            print(f"  [{tag:4}] iter {i+1:2d}: tok={tok:3d} finish={finish} content={content[:80]!r}")
            if not ok:
                fails.append((i + 1, content[:120], finish, tok))
        except Exception as e:
            print(f"  [ERR ] iter {i+1:2d}: {e}")
            fails.append((i + 1, str(e), "error", 0))

    if fails:
        print(f"\nFAIL: {len(fails)}/{args.iters} iterations did not contain {args.expect!r}.")
        print("This is the patch-001 (mlx-radix-cache) repeat-prompt corruption bug.")
        print("Workaround: launch with EXTRA_ARGS=\"--disable-radix-cache\".")
        return 1
    print(f"\nPASS: all {args.iters} iterations returned consistent correct answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
