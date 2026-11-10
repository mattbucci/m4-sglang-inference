#!/usr/bin/env python3
"""Per-token decode trace of one long request: step time against context depth.

decode_ab.py reports one median per request. This driver keeps every streamed
chunk's arrival time, so a single request shows how the step time moves while
the context grows through a depth of interest, for example a kernel ladder
threshold. One request, greedy, ignore_eos, the same seeded prompt builder as
decode_ab.py.

  python scripts/bench/decode_step_trace.py --port 23334 --tokenizer <model dir> \
      --context 149800 --max-tokens 800 --tag ladder --out ladder.json
  python scripts/bench/decode_step_trace.py --show ladder.json [--window 64]
  python scripts/bench/decode_step_trace.py --compare base.json ladder.json

--show prints the median step time per window of chunks with the context depth
at the window start. --compare adds the length of the common text prefix, which
for two arms that only differ past some depth is where they first diverge.
Run the OOM guard for depths of 32K and beyond.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def trace(a) -> dict:
    from decode_ab import build_prompt

    prompt = build_prompt(a.context, a.tokenizer)
    body = {
        "model": "default", "prompt": prompt, "max_tokens": a.max_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://{a.host}:{a.port}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    chunks, usage = [], {}
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
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
                    chunks.append([round((time.perf_counter() - t0) * 1e3, 3), ch["text"]])
    return {
        "tag": a.tag, "target": a.context, "max_tokens": a.max_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens", len(chunks)),
        "ttft_s": round(chunks[0][0] / 1e3, 3) if chunks else None,
        "chunks": chunks,
    }


def windows(t: dict, window: int):
    stamps = [c[0] for c in t["chunks"]]
    per_chunk = t["completion_tokens"] / max(len(stamps), 1)
    rows = []
    for i in range(0, len(stamps) - 1, window):
        seg = stamps[i:i + window + 1]
        if len(seg) < 3:
            break
        gaps = [(b - a) / per_chunk for a, b in zip(seg, seg[1:])]
        rows.append({
            "chunk": i, "depth": t["prompt_tokens"] + int(i * per_chunk),
            "ms_median": round(statistics.median(gaps), 3), "ms_max": round(max(gaps), 2),
        })
    return rows


def show(path: str, window: int):
    t = json.load(open(path))
    print(f"{t['tag']}: prompt {t['prompt_tokens']} tokens, {t['completion_tokens']} generated in "
          f"{len(t['chunks'])} chunks, ttft {t['ttft_s']} s")
    for r in windows(t, window):
        print(f"  depth {r['depth']:>7}  median {r['ms_median']:7.3f} ms/step  max {r['ms_max']:7.2f} ms")


def compare(paths, window: int):
    a, b = (json.load(open(p)) for p in paths)
    ta, tb = [c[1] for c in a["chunks"]], [c[1] for c in b["chunks"]]
    same = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        same += 1
    print(f"common prefix: {same} of {min(len(ta), len(tb))} chunks "
          f"(prompts {a['prompt_tokens']} / {b['prompt_tokens']} tokens)")
    wa, wb = windows(a, window), windows(b, window)
    print(f"  {'depth':>7}  {a['tag']:>12}  {b['tag']:>12}  delta")
    for ra, rb in zip(wa, wb):
        d = (rb["ms_median"] / ra["ms_median"] - 1) * 100
        print(f"  {ra['depth']:>7}  {ra['ms_median']:9.3f} ms  {rb['ms_median']:9.3f} ms  {d:+6.1f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tokenizer", default="", help="model dir with tokenizer.json for exact depths")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.show:
        return show(a.show, a.window)
    if a.compare:
        return compare(a.compare, a.window)
    t = trace(a)
    if a.out:
        json.dump(t, open(a.out, "w"))
    print(f"[{t['tag']}] prompt {t['prompt_tokens']} generated {t['completion_tokens']} ttft {t['ttft_s']}s")
    for r in windows(t, a.window):
        print(f"  depth {r['depth']:>7}  median {r['ms_median']:7.3f} ms/step  max {r['ms_max']:7.2f} ms")


if __name__ == "__main__":
    main()
