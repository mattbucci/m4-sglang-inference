#!/usr/bin/env python3
"""Default-sampling probe: chat requests that send no sampling fields.

Every other gate sends temperature 0, so none of them sees what a client that
sends nothing gets. The omp harness is such a client. The server fills the
unset fields from the checkpoint's generation_config.json; a checkpoint that
ships none leaves temperature 1.0 over the whole vocabulary, and a checkpoint
with a heavy logit tail (mlx-community/Qwen3-32B-4bit-DWQ) returns token soup
there while every greedy gate passes. Presets whose checkpoints ship no
generation config pass --sampling-defaults-fallback (patch 041).

The probe sends four short questions, each `--n` times, with no temperature,
top_p or top_k, and thinking disabled where the template allows it. A reply
passes when its text (answer plus reasoning) holds the expected keyword and is
at least 97 % printable ASCII.

  STRONG   every reply passes
  PARTIAL  some replies fail
  FAIL     no reply passes

Usage:
  python scripts/eval/probe_defaults.py [--port PORT] [--n N] [--max-tokens N]
                                        [--out receipt.json]
"""
import argparse
import json
import sys
import time
import urllib.request

# Token soup from Qwen3-32B-4bit-DWQ measured 0.91 to 0.94 printable ASCII,
# coherent replies 1.00.
MIN_ASCII_SHARE = 0.97

CASES = [
    ("What is the capital of France? Answer in one sentence.", ["paris"]),
    ("What is 17 + 25? Answer with the number.", ["42"]),
    ("Name the largest planet in the solar system in one sentence.", ["jupiter"]),
    ("Write a Python function add(a, b) that returns their sum. Code only.", ["def add", "return"]),
]


def ask(base, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ascii_share(text):
    if not text:
        return 0.0
    ok = sum(1 for c in text if c in "\n\t" or 32 <= ord(c) < 127)
    return ok / len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--model", default="default")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"

    rows, passed = [], 0
    for prompt, keys in CASES:
        for i in range(a.n):
            t0 = time.time()
            try:
                msg_obj = ask(base, a.model, prompt, a.max_tokens, a.timeout)
            except Exception as e:  # a transport failure is a failed reply
                rows.append({"prompt": prompt, "error": repr(e)[:200], "ok": False})
                print(f"  FAIL  request error {e!r}"[:160])
                continue
            choice = msg_obj["choices"][0]
            msg = choice["message"]
            text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
            share = ascii_share(text.strip())
            has_keys = all(k in text.lower() for k in keys)
            ok = has_keys and share >= MIN_ASCII_SHARE
            passed += ok
            rows.append({
                "prompt": prompt, "ok": ok, "keyword": has_keys,
                "ascii_share": round(share, 3), "finish": choice.get("finish_reason"),
                "completion_tokens": msg_obj.get("usage", {}).get("completion_tokens"),
                "seconds": round(time.time() - t0, 2), "text": text.strip()[:400],
            })
            head = " ".join(text.split())[:90]
            print(f"  {'ok  ' if ok else 'FAIL'}  ascii {share:.2f}  keyword {'yes' if has_keys else 'no '}  {head!r}")

    total = len(CASES) * a.n
    verdict = "STRONG" if passed == total else ("PARTIAL" if passed else "FAIL")
    print(f"DEFAULT SAMPLING {passed}/{total}")
    print(f"VERDICT: {verdict}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"passed": passed, "total": total, "verdict": verdict, "rows": rows}, fh, indent=1)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
