#!/usr/bin/env python3
"""Decode throughput of N concurrent streams against a running server.

Starts N streaming completions together, each on its own prompt, and counts
the tokens that arrive while all N streams are decoding: the window opens at
the last stream's first chunk and closes at the first stream's last chunk.
Prefill is outside the window, and so is a stream that decodes alone while
the others are still being prefilled (prefill is serial on this backend, so
an early stream stalls for the prefill of a later one):

  aggregate tok/s    tokens of all streams inside the window / window length
  per-stream tok/s   aggregate / N
  TPOT ms            1000 / per-stream tok/s
  TTFT s             request start to first chunk, mean over streams

Greedy, ``ignore_eos`` and a fixed ``max_tokens`` make every stream the same
length. Each level runs ``--repeats`` times on fresh prompts (no radix hit)
and reports the median. Text is not compared: two correct decode paths round
differently and greedy prose parts at the first near-tie, so fidelity is
measured teacher-forced (``scripts/eval/batched_decode_fidelity.py``).

  python scripts/bench/concurrent_decode_probe.py --port 23334 --streams 1,2,4,8 \
      --prompt-words 180 --max-tokens 200 --out out.json
"""
import argparse
import hashlib
import json
import random
import statistics
import threading
import time
import urllib.request

WORDS = (
    "river stone market lantern copper orchard winter signal harbor meadow "
    "ledger window thread anchor valley ember garden mirror timber candle "
    "bridge kettle shadow marble feather compass thunder velvet canyon whistle"
).split()


def prompt(seed, words):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    # The seed leads, so no two prompts share a prefix for the radix tree.
    return f"{seed} is the number of this notebook entry. {body}\n\nContinue the notebook in plain prose:\n"


def stream(url, model, text, max_tokens, barrier, out, i, timeout):
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    start = time.perf_counter()
    stamps, pieces, usage = [], [], None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                piece = choices[0].get("text", "") if choices else ""
                if piece:
                    stamps.append(time.perf_counter())
                    pieces.append(piece)
        out[i] = {
            "start": start,
            "first": stamps[0] if stamps else None,
            "last": stamps[-1] if stamps else None,
            "stamps": stamps,
            "text": "".join(pieces),
            "chunks": len(pieces),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        }
    except Exception as e:  # noqa: BLE001
        out[i] = {"error": repr(e)}


def run_level(url, model, n, seeds, args):
    barrier = threading.Barrier(n)
    out = [None] * n
    threads = [
        threading.Thread(
            target=stream,
            args=(url, model, prompt(seeds[i], args.prompt_words), args.max_tokens, barrier, out, i, args.timeout),
        )
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = [o["error"] for o in out if "error" in o]
    if errors:
        return {"streams": n, "error": errors[0]}
    if any(not o["stamps"] for o in out):
        return {"streams": n, "error": "a stream returned no tokens"}
    # Window in which every stream is decoding.
    t0 = max(o["first"] for o in out)
    t1 = min(o["last"] for o in out)
    inside, solo_rates = 0.0, []
    for o in out:
        tokens = o["completion_tokens"] or o["chunks"]
        per_chunk = tokens / o["chunks"]
        inside += per_chunk * sum(1 for t in o["stamps"] if t0 < t <= t1)
        solo_rates.append((tokens - 1) / (o["last"] - o["first"]))
    if t1 <= t0 or inside == 0:
        return {"streams": n, "error": "the streams never decoded together"}
    aggregate = inside / (t1 - t0)
    wall = max(o["last"] for o in out) - min(o["start"] for o in out)
    return {
        "streams": n,
        "per_stream_tok_s": aggregate / n,
        "aggregate_tok_s": aggregate,
        "tpot_ms": 1000.0 * n / aggregate,
        "ttft_s": statistics.mean(o["first"] - o["start"] for o in out),
        "window_s": t1 - t0,
        "window_tokens": inside,
        # First chunk to last chunk of each stream, stalls included.
        "first_to_last_tok_s": statistics.mean(solo_rates),
        "wall_tok_s": sum(o["completion_tokens"] or o["chunks"] for o in out) / wall,
        "prompt_tokens": [o["prompt_tokens"] for o in out],
        "completion_tokens": [o["completion_tokens"] for o in out],
        "texts": [o["text"] for o in out],
        "seeds": list(seeds[:n]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--streams", default="1,2,4,8")
    p.add_argument("--prompt-words", type=int, default=180)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))["data"][0]["id"]
    url = f"{base}/v1/completions"
    levels = [int(x) for x in args.streams.split(",")]

    # Warm the kernels and the stream path outside the measurement.
    run_level(url, model, 1, [999_001], args)
    run_level(url, model, max(levels), [999_100 + i for i in range(max(levels))], args)

    rows = []
    for rep in range(args.repeats):
        for n in levels:
            # Stream i of every level in a repeat shares its prompt words, but a
            # level-specific salt keeps the prompt off the radix tree.
            seeds = [10_000 * (rep + 1) + i for i in range(n)]
            salted = [s + 1_000_000 * n for s in seeds]
            r = run_level(url, model, n, salted, args)
            r["repeat"] = rep
            rows.append(r)
            if "error" in r:
                print(f"streams {n} rep {rep}: ERROR {r['error']}", flush=True)
                continue
            print(
                f"streams {n} rep {rep}: per-stream {r['per_stream_tok_s']:.2f} tok/s "
                f"aggregate {r['aggregate_tok_s']:.2f} tpot {r['tpot_ms']:.2f} ms "
                f"ttft {r['ttft_s']:.2f} s window {r['window_s']:.2f} s "
                f"({r['window_tokens']:.0f} tokens) first-to-last {r['first_to_last_tok_s']:.2f} tok/s "
                f"prompt {r['prompt_tokens'][0]} out {r['completion_tokens'][0]}",
                flush=True,
            )

    summary = []
    for n in levels:
        ok = [r for r in rows if r["streams"] == n and "error" not in r]
        if not ok:
            summary.append({"streams": n, "error": True})
            continue
        summary.append(
            {
                "streams": n,
                "per_stream_tok_s": round(statistics.median(r["per_stream_tok_s"] for r in ok), 2),
                "aggregate_tok_s": round(statistics.median(r["aggregate_tok_s"] for r in ok), 2),
                "tpot_ms": round(statistics.median(r["tpot_ms"] for r in ok), 2),
                "ttft_s": round(statistics.median(r["ttft_s"] for r in ok), 2),
            }
        )
    print("\nstreams  per-stream tok/s  aggregate tok/s  TPOT ms  TTFT s")
    for s in summary:
        if "error" in s:
            print(f"{s['streams']:>7}  ERROR")
        else:
            print(f"{s['streams']:>7}  {s['per_stream_tok_s']:>16.2f}  {s['aggregate_tok_s']:>15.2f}  {s['tpot_ms']:>7.2f}  {s['ttft_s']:>6.2f}")
    if args.out:
        for r in rows:
            if "texts" in r:
                r["text_sha"] = [hashlib.sha256(t.encode()).hexdigest()[:12] for t in r.pop("texts")]
        with open(args.out, "w") as f:
            json.dump({"tag": args.tag, "model": model, "args": vars(args), "summary": summary, "rows": rows}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
