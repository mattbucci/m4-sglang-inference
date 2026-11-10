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

--show prints the median and mean step time per window of generated tokens with
the context depth at the window start, then every stall (a step more than 40 ms
over the median) and what the stalls cost per step. The request asks for usage
on every chunk, because a chunk can carry more than one token. --compare adds
the length of the common text prefix, which for two arms that only differ past
some depth is where they first diverge. Run the OOM guard for depths of 32K
and beyond.
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
        # usage on every chunk: a chunk can carry more than one token (the
        # detokenizer holds text back), so chunk gaps alone are not step times
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
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
                    chunks.append([round((time.perf_counter() - t0) * 1e3, 3), ch["text"],
                                   (chunk.get("usage") or {}).get("completion_tokens")])
    return {
        "tag": a.tag, "target": a.context, "max_tokens": a.max_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens", len(chunks)),
        "ttft_s": round(chunks[0][0] / 1e3, 3) if chunks else None,
        "chunks": chunks,
    }


def steps(t: dict):
    """(depth, ms) per generated token. A chunk that carries k tokens spreads
    its gap over k steps. Traces without per-chunk token counts fall back to
    the nearest multiple of the median gap."""
    ch = t["chunks"]
    counted = all(len(c) > 2 and c[2] for c in ch)
    if not counted:
        base = statistics.median(b[0] - a[0] for a, b in zip(ch, ch[1:]))
    out, n, k = [], (ch[0][2] if counted else 1), 1
    for a, b in zip(ch, ch[1:]):
        gap = b[0] - a[0]
        if counted:
            k = max(1, b[2] - a[2])
        elif gap <= 2.6 * base:  # a longer gap is a stall: keep the previous chunk's shape
            k = max(1, round(gap / base))
        for _ in range(k):
            n += 1
            out.append((t["prompt_tokens"] + n, gap / k))
    return out, counted


def windows(t: dict, window: int):
    st, _ = steps(t)
    rows = []
    for i in range(0, len(st), window):
        seg = [ms for _, ms in st[i:i + window]]
        if len(seg) < 3:
            break
        rows.append({"depth": st[i][0], "ms_median": round(statistics.median(seg), 3),
                     "ms_mean": round(statistics.fmean(seg), 3), "ms_max": round(max(seg), 2)})
    return rows


def stalls(t: dict, over_ms: float = 40.0):
    """Steps that take over_ms longer than the median step, with the excess."""
    st, _ = steps(t)
    med = statistics.median(ms for _, ms in st)
    found = []
    for d, ms in st:
        if ms - med <= over_ms:
            continue
        if found and found[-1][0] == d - 1:  # one slow chunk spread over its tokens
            found[-1] = (found[-1][0], round(found[-1][1] + ms - med, 1))
        else:
            found.append((d, round(ms - med, 1)))
    return found


def report(t: dict, window: int):
    st, counted = steps(t)
    for r in windows(t, window):
        print(f"  depth {r['depth']:>7}  median {r['ms_median']:7.3f} ms/step  "
              f"mean {r['ms_mean']:7.3f}  max {r['ms_max']:7.2f} ms")
    found = stalls(t)
    extra = sum(x for _, x in found)
    print(f"  {len(st)} steps, median {statistics.median(ms for _, ms in st):.3f} ms, "
          f"mean {statistics.fmean(ms for _, ms in st):.3f} ms"
          + ("" if counted else "  (token counts inferred from gaps)"))
    print(f"  stalls over 40 ms: {found}  = {extra / max(len(st), 1):.3f} ms per step")


def show(path: str, window: int):
    t = json.load(open(path))
    print(f"{t['tag']}: prompt {t['prompt_tokens']} tokens, {t['completion_tokens']} generated in "
          f"{len(t['chunks'])} chunks, ttft {t['ttft_s']} s")
    report(t, window)


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
    for t in (a, b):
        st, _ = steps(t)
        print(f"  {t['tag']}: mean {statistics.fmean(ms for _, ms in st):.3f} ms per step, stalls {stalls(t)}")


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
    report(t, a.window)


if __name__ == "__main__":
    main()
