#!/usr/bin/env python3
"""Published-shape sweep: the request shapes other engines publish for a
checkpoint, measured against one booted server, one JSON per arm.

  cold      one cold request per depth, then N streamed tokens. Every depth
            has its own seed, so no depth is a prefix of another and the
            radix cache cannot serve any part of it. Prefill tok/s is prompt
            tokens over TTFT; decode tok/s runs from the first to the last
            streamed token.
  repeat    the same request again (--repeat): TTFT on a whole-prompt hit and
            the decode rate on the hit path.
  runs      K runs of M tokens on one short prompt (--runs depth:tokens:count:mode,
            mode sampled or greedy): the decode-only rows, prefill excluded.
  followup  chat shape (--followup depths): a user turn of D tokens, the
            reply, then a second user turn. TTFT and cached tokens of the
            second request, which is what an agent turn pays.

Sampling for the sampled rows comes from --sampling (JSON merged into the
request); greedy rows send temperature 0. Completions rows set ignore_eos so
every run has the same token count.

  python scripts/bench/published_shape_sweep.py --port 23334 --tag int4 \
      --tokenizer <model dir> --contexts 1024,2048,4096,8192,16384,32768 --repeat \
      --runs 1024:512:5:sampled,1024:512:5:greedy,1024:1024:5:sampled \
      --followup 4096,16384 --out flint-int4.json
  python scripts/bench/published_shape_sweep.py --table a.json b.json

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

from decode_ab import build_prompt, one_request  # noqa: E402


def chat_request(base: str, messages, max_tokens: int, timeout: float, extra: dict) -> dict:
    body = {
        "model": "default", "messages": messages, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }
    body.update(extra)
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = last = None
    content, reasoning, usage, finish = [], [], {}, None
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
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                    last = time.perf_counter()
                    first = first or last
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                finish = ch.get("finish_reason") or finish
    n = usage.get("completion_tokens")
    out = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": n,
        "ttft_s": round(first - t0, 4) if first else None,
        "finish": finish,
        "content": "".join(content),
        "reasoning_chars": len("".join(reasoning)),
    }
    if first and last and n and n >= 2 and last > first:
        out["decode_tok_s"] = round((n - 1) / (last - first), 2)
    return out


def prefill_rate(r: dict):
    fresh = (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0)
    return round(fresh / r["ttft_s"], 1) if r.get("ttft_s") and fresh > 0 else None


def run(a) -> dict:
    base = f"http://{a.host}:{a.port}"
    sampling = json.loads(a.sampling)
    arm = {"tag": a.tag, "sampling": sampling, "max_tokens": a.max_tokens,
           "cold": [], "runs": [], "followup": []}
    one_request(base, "Warm up the server with a short request.", 16, a.timeout, {})
    for ctx in [int(c) for c in a.contexts.split(",") if c.strip()]:
        prompt = build_prompt(ctx, a.tokenizer, seed=1000 + ctx)
        cold = one_request(base, prompt, a.max_tokens, a.timeout, sampling)
        row = {"target": ctx, "cold": cold, "prefill_tok_s": prefill_rate(cold)}
        line = (f"[{a.tag}] cold {cold['prompt_tokens']:>7} tok: cached {cold['cached_tokens']}, "
                f"ttft {cold['ttft_s']} s, prefill {row['prefill_tok_s']} tok/s, "
                f"decode {cold.get('decode_tok_s')} tok/s")
        if a.repeat:
            row["repeat"] = rep = one_request(base, prompt, a.max_tokens, a.timeout, sampling)
            line += (f"; repeat cached {rep['cached_tokens']}, ttft {rep['ttft_s']} s, "
                     f"decode {rep.get('decode_tok_s')} tok/s")
        print(line, flush=True)
        arm["cold"].append(row)
    for spec in [s for s in a.runs.split(",") if s.strip()]:
        depth, tokens, count, mode = spec.split(":")
        extra = sampling if mode == "sampled" else {}
        prompt = build_prompt(int(depth), a.tokenizer, seed=2000 + int(depth))
        one_request(base, prompt, 16, a.timeout, extra)
        rs = [one_request(base, prompt, int(tokens), a.timeout, extra) for _ in range(int(count))]
        rates = [r["decode_tok_s"] for r in rs if "decode_tok_s" in r]
        row = {"spec": spec, "prompt_tokens": rs[0]["prompt_tokens"], "mode": mode,
               "tokens": int(tokens), "decode_tok_s": rates,
               "decode_tok_s_mean": round(statistics.mean(rates), 2),
               "decode_tok_s_median": round(statistics.median(rates), 2), "requests": rs}
        print(f"[{a.tag}] runs {spec}: prompt {row['prompt_tokens']}, decode mean "
              f"{row['decode_tok_s_mean']} median {row['decode_tok_s_median']} tok/s "
              f"({min(rates)} to {max(rates)})", flush=True)
        arm["runs"].append(row)
    for ctx in [int(c) for c in a.followup.split(",") if c.strip()]:
        text = build_prompt(ctx, a.tokenizer, seed=3000 + ctx)
        msgs = [{"role": "user", "content": text + "\n\nSummarise the text above in three sentences."}]
        first = chat_request(base, msgs, a.max_tokens, a.timeout, {})
        msgs += [{"role": "assistant", "content": first["content"]},
                 {"role": "user", "content": "Now give it a one-line title."}]
        second = chat_request(base, msgs, 64, a.timeout, {})
        fresh = (second["prompt_tokens"] or 0) - (second["cached_tokens"] or 0)
        for r in (first, second):
            r["content_chars"] = len(r.pop("content"))
        row = {"target": ctx, "first": first, "second": second, "fresh_tokens": fresh}
        print(f"[{a.tag}] followup {first['prompt_tokens']:>7} tok: first ttft {first['ttft_s']} s "
              f"({first['completion_tokens']} tok, finish {first['finish']}); second ttft "
              f"{second['ttft_s']} s, {second['cached_tokens']} of {second['prompt_tokens']} "
              f"cached, {fresh} fresh", flush=True)
        arm["followup"].append(row)
    return arm


def table(paths):
    arms = [json.load(open(p)) for p in paths]
    print("| prompt tokens | " + " | ".join(f"{x['tag']} prefill / decode / repeat decode" for x in arms) + " |")
    print("|---|" + "---|" * len(arms))
    for i, row in enumerate(arms[0]["cold"]):
        cells = []
        for x in arms:
            if i >= len(x["cold"]):
                cells.append("")
                continue
            r = x["cold"][i]
            rep = (r.get("repeat") or {}).get("decode_tok_s")
            cells.append(f"{r['prefill_tok_s']} / {r['cold'].get('decode_tok_s')} / {rep}")
        print(f"| {row['cold']['prompt_tokens']} | " + " | ".join(cells) + " |")
    for x in arms:
        for r in x["runs"]:
            print(f"{x['tag']} runs {r['spec']}: mean {r['decode_tok_s_mean']} median {r['decode_tok_s_median']} tok/s")
        for r in x["followup"]:
            s = r["second"]
            print(f"{x['tag']} followup {r['first']['prompt_tokens']}: second ttft {s['ttft_s']} s, "
                  f"{s['cached_tokens']} of {s['prompt_tokens']} cached")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", nargs="+")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--contexts", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--runs", default="")
    ap.add_argument("--followup", default="")
    ap.add_argument("--sampling", default='{"temperature": 1.0, "top_p": 0.95, "top_k": 20}')
    ap.add_argument("--tokenizer", default="", help="model dir with tokenizer.json for exact depths")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.table:
        return table(a.table)
    arm = run(a)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(arm, open(a.out, "w"), indent=2)
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
