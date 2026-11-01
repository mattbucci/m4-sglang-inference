#!/usr/bin/env python3
"""Long-generation degeneration probe: greedy 2,048-token answers to fixed prompts.

Counts outputs that hit the token cap, outputs whose tail repeats, and
reasoning that never closes; the failure mode an INT4 recurrent-state gate
would show before any benchmark score moves.

    python scripts/eval/probe_loops.py --port 23334 --tag incumbent --out loops-incumbent.json
"""
import argparse
import collections
import json
import re
import sys
import time
import urllib.request

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into total seconds, with tests.",
    "Explain how a radix tree prefix cache serves a multi-turn chat; include a worked example with three turns.",
    "Implement an LRU cache in Python without using functools or OrderedDict, then explain its complexity.",
    "A train leaves at 9:40 and travels 312 km at 96 km/h with two 6-minute stops. When does it arrive? Show the arithmetic.",
    "List 25 distinct, real programming languages with the year each first appeared, oldest first.",
    "Write a bash script that finds the ten largest files under a directory, skipping symlinks, and prints sizes in MiB.",
    "Summarize the trade-offs between MoE and dense transformer models for on-device inference in about 200 words.",
    "Given the regex r'READ [TS]ERR(\\s+[0-9]+)+', explain what it matches and rewrite it to be case-insensitive in three different ways.",
    "Write a SQL query that returns the second-highest salary per department from an employees table, handling ties.",
    "Describe the steps to debug a server that returns 'context length exceeded' after a long agent session; propose two fixes.",
    "Write a short story of about 300 words about a lighthouse keeper who discovers a tide table is wrong.",
    "Implement binary search over a sorted list of floats with a tolerance, and list five edge cases with expected results.",
    "Explain why bf16 accumulation order can change the argmax of a softmax at a near-tie, with a numeric example.",
    "Convert this JSON to a Markdown table and sort by score descending: [{\"name\":\"a\",\"score\":3},{\"name\":\"b\",\"score\":7},{\"name\":\"c\",\"score\":5}]",
    "Write a Python class that models a bank account with deposits, withdrawals and an audit log; include three unit tests.",
    "What is 17 * 23 + 19 * 29 - 31 * 7? Then factor the result into primes.",
    "Outline a 7-day plan to learn the basics of Rust for someone who knows Python, one topic per day with an exercise each.",
    "Write a function that merges overlapping intervals, then trace it on [[1,4],[2,5],[7,9],[8,10],[12,13]].",
    "Explain the difference between TTFT and TPOT in LLM serving and how a prefix cache affects each.",
    "Produce a CSV with 20 rows of synthetic sensor readings: timestamp (hourly from 2026-01-01T00:00), temperature, humidity.",
    "Write a regular expression that validates IPv4 addresses and explain each component.",
    "Refactor this code for clarity: def f(l):\n  r=[]\n  for i in range(len(l)):\n    if l[i]%2==0: r.append(l[i]*l[i])\n  return r",
    "Give a step-by-step derivation of the closed form of the sum of the first n squares.",
    "Write a Dockerfile for a small FastAPI service and explain each layer's caching behavior.",
    "Explain how to detect a memory leak in a long-running Python service on macOS without an OOM killer; name the tools.",
    "Write a Python generator that yields Fibonacci numbers lazily and show how to take the first 15 that are even.",
    "Compare three ways to implement rate limiting in an HTTP API; recommend one for a single-user local server.",
    "Translate this pseudo-code into idiomatic Python: for each line in a file, if it starts with '#', skip; else split on ',' and sum the second column.",
    "Describe the git commands to split the last commit into two commits, and what happens to a branch that already pulled it.",
    "Write a haiku sequence (five haiku) about unified memory on Apple Silicon.",
    "Explain gradient checkpointing and estimate the memory saved for a 40-layer model with 2 GB of activations per layer.",
    "Implement Dijkstra's algorithm with a binary heap in Python and run it on a 5-node example graph, listing the distances.",
    "Why might a quantized MoE router pick a different expert than the full-precision router? What symptoms would appear?",
    "Write a test plan with ten cases for a function that parses command-line flags with defaults and overrides.",
    "Give the Taylor series of ln(1+x) to five terms and use it to approximate ln(1.2) to four decimals.",
    "Write a Python script that watches a directory and prints new files as they appear, using only the standard library.",
    "Explain what a sliding-window attention layer stores in its KV cache and why a 513-token window needs at most 513 entries.",
    "Draft a polite email to a vendor asking for a replacement part, including an order number and a two-week deadline.",
    "Write a function that converts a nested dict into dotted-path keys, then its inverse, with examples.",
    "List the planets of the solar system with their orbital periods in Earth years, then sort them by period descending.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--model", default="qwen36")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    rows = []
    t0 = time.time()
    for i, prompt in enumerate(PROMPTS[: a.n]):
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens, "temperature": a.temperature}
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        ch = out["choices"][0]
        msg = ch["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        text = reasoning + "\n" + content
        words = re.findall(r"\S+", text)
        grams = collections.Counter(tuple(words[j:j + 12]) for j in range(max(0, len(words) - 12)))
        max_rep = max(grams.values()) if grams else 0
        tail = text[-600:]
        tail_rep = any(tail.count(tail[k:k + 30]) >= 3 for k in range(0, max(1, len(tail) - 30), 10)) if len(tail) >= 60 else False
        usage = out.get("usage", {})
        row = {"i": i, "finish": ch.get("finish_reason"), "completion_tokens": usage.get("completion_tokens"),
               "reasoning_chars": len(reasoning), "content_chars": len(content), "max_12gram_repeat": max_rep,
               "tail_repeats": tail_rep, "secs": round(time.time() - t1, 1)}
        row["degenerate"] = row["finish"] == "length" and (max_rep >= 6 or tail_rep)
        row["unclosed_reasoning"] = row["finish"] == "length" and len(content) == 0 and len(reasoning) > 0
        rows.append(row)
        print(f"[{a.tag}] {i:2d} finish={row['finish']:6s} tok={row['completion_tokens']} rep12={max_rep} tail={tail_rep} deg={row['degenerate']} {row['secs']}s", flush=True)
    summary = {"n": len(rows), "length": sum(r["finish"] == "length" for r in rows),
               "degenerate": sum(r["degenerate"] for r in rows), "unclosed_reasoning": sum(r["unclosed_reasoning"] for r in rows),
               "mean_tokens": sum(r["completion_tokens"] or 0 for r in rows) / max(1, len(rows)), "secs": round(time.time() - t0)}
    json.dump({"tag": a.tag, "rows": rows, "summary": summary}, open(a.out, "w"), indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
