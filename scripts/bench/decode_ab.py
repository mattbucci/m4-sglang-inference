#!/usr/bin/env python3
"""Serving decode A/B: median TPOT per context depth, one JSON per arm.

Each arm is one booted server. For every depth the driver sends the same
seeded prompt N times to /v1/completions (streaming, temperature 0,
ignore_eos) and records, per run: TTFT, the server-counted prompt and
completion tokens, TPOT from the first to the last streamed token, and a
hash of the generated text. The first run at a depth is the cold one; under
the radix cache the later runs are prefix hits, which is the agentic shape.

  python scripts/bench/decode_ab.py --port 23334 --tag base \
      --tokenizer <model dir> --contexts 1024,8192,32768 --out base.json
  python scripts/bench/decode_ab.py --compare base.json cand.json

Protocol for an experiment: both arms on the same tree, run in both orders
(A B B A), compare medians, and require identical text hashes for changes
that claim greedy identity. Run the OOM guard for depths of 32K and beyond.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_prompt(n_tokens: int, tokenizer_dir: str, seed: int = 7) -> str:
    from mlx_decode_census import WORDS

    rng = random.Random(seed)
    if tokenizer_dir:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        parts, count = [], 0
        while count < n_tokens:
            sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18))).capitalize() + ". "
            parts.append(sent)
            count += len(tok.encode(sent, add_special_tokens=False).ids)
        return "".join(parts)
    chars = int(n_tokens * 4.6)
    parts, size = [], 0
    while size < chars:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18))).capitalize() + ". "
        parts.append(sent)
        size += len(sent)
    return "".join(parts)


def one_request(base: str, prompt: str, max_tokens: int, timeout: float, extra: dict) -> dict:
    body = {
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra)
    req = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    stamps, text, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
            for ch in chunk.get("choices") or []:
                if ch.get("text"):
                    stamps.append(time.perf_counter())
                    text.append(ch["text"])
    t_end = time.perf_counter()
    n = usage.get("completion_tokens", len(stamps))
    out = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": n,
        "ttft_s": round(stamps[0] - t0, 4) if stamps else None,
        "total_s": round(t_end - t0, 4),
        "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16],
    }
    if len(stamps) >= 2 and n >= 2:
        tpot = (stamps[-1] - stamps[0]) / (n - 1)
        out["tpot_ms"] = round(tpot * 1e3, 4)
        out["decode_tok_s"] = round(1.0 / tpot, 2)
    return out


def run_arm(a) -> dict:
    base = f"http://{a.host}:{a.port}"
    extra = json.loads(a.extra_body) if a.extra_body else {}
    arm = {"tag": a.tag, "max_tokens": a.max_tokens, "runs": a.runs, "contexts": []}
    one_request(base, "Warm up the server with a short request.", 16, a.timeout, extra)
    for ctx in [int(c) for c in a.contexts.split(",") if c.strip()]:
        prompt = build_prompt(ctx, a.tokenizer)
        runs = []
        for i in range(a.runs):
            r = one_request(base, prompt, a.max_tokens, a.timeout, extra)
            runs.append(r)
            print(f"  [{a.tag}] ctx {r['prompt_tokens']} run {i}: ttft {r['ttft_s']}s "
                  f"tpot {r.get('tpot_ms')} ms {r.get('decode_tok_s')} tok/s sha {r['text_sha']}",
                  flush=True)
        tp = [r["tpot_ms"] for r in runs if "tpot_ms" in r]
        row = {
            "target": ctx, "prompt_tokens": runs[0]["prompt_tokens"],
            "tpot_ms_median": round(statistics.median(tp), 4),
            "tpot_ms_min": min(tp), "tpot_ms_max": max(tp),
            "decode_tok_s_median": round(1e3 / statistics.median(tp), 2),
            "ttft_cold_s": runs[0]["ttft_s"],
            "ttft_repeat_s": round(statistics.median([r["ttft_s"] for r in runs[1:]]), 4) if len(runs) > 1 else None,
            "text_shas": sorted({r["text_sha"] for r in runs}),
            "runs": runs,
        }
        arm["contexts"].append(row)
    return arm


def compare(paths):
    arms = [json.load(open(p)) for p in paths]
    base = arms[0]
    print("| depth | " + " | ".join(f"{x['tag']} ms / tok/s" for x in arms) + " | delta vs first | same text |")
    print("|---|" + "---|" * (len(arms) + 2))
    for i, row in enumerate(base["contexts"]):
        cells, deltas, same = [], [], []
        for x in arms:
            r = x["contexts"][i]
            cells.append(f"{r['tpot_ms_median']:.2f} / {r['decode_tok_s_median']:.1f}")
            if x is not base:
                deltas.append(f"{100 * (row['tpot_ms_median'] / r['tpot_ms_median'] - 1):+.1f}%")
                same.append("yes" if r["text_shas"] == row["text_shas"] else "no")
        print(f"| {row['prompt_tokens']} | " + " | ".join(cells) + " | " + ", ".join(deltas)
              + " | " + ", ".join(same) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--contexts", default="1024,8192,32768")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokenizer", default="", help="model dir with tokenizer.json for exact depths")
    ap.add_argument("--extra-body", default="", help="JSON merged into each request")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    arm = run_arm(a)
    for row in arm["contexts"]:
        print(f"[{a.tag}] {row['prompt_tokens']:>7} tok: {row['tpot_ms_median']:.2f} ms/step "
              f"{row['decode_tok_s_median']:.1f} tok/s (min {row['tpot_ms_min']:.2f} max {row['tpot_ms_max']:.2f}) "
              f"ttft cold {row['ttft_cold_s']}s repeat {row['ttft_repeat_s']}s shas {len(row['text_shas'])}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(arm, open(a.out, "w"), indent=2)
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
