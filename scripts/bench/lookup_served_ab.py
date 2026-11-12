#!/usr/bin/env python3
"""Served A/B for prompt-lookup drafts (patch 040) on archived agent turns.

Each arm is one booted server. For every listed turn of an archived omp
session the driver sends the conversation up to that turn as token ids to
/generate (streaming, one request at a time; greedy unless `--temperature`
is set) and records TTFT, the cached prompt tokens, the output ids, and the
decode rate from the first streamed token to the last. `--passes 2` sends the
list twice: the second pass is a radix hit on the whole prompt, a second
sample of the same decode. A sampled arm draws another text on every pass, so
it is judged on the pooled rate of several passes, never on one request.

`--followup` grades the state a lookup-decoded request leaves in the radix
cache. After each turn it sends that turn's prompt and output plus a short
user message twice, with top-k logprobs: once on the cached history (a hit
that starts from the state the turn stored when it finished) and once after
a flush (cold). KL(cold || hit) per shared step is the measure; an arm with
the flag off gives the noise class a hit is allowed. `--followup-kl` reads
the follow-ups of several arms against the first arm's cold run, so a hit
after a drafted turn stands next to a hit after an ordinary one and next to
a cold run on another chunk size.

  python scripts/bench/lookup_served_ab.py --port 23334 --model <repo or dir> \
      --session "<session.jsonl>@12,22" --passes 2 --tag off --out off.json
  python scripts/bench/lookup_served_ab.py --compare off.json rows4.json rows2.json
  python scripts/bench/lookup_served_ab.py --pool off=off-1.json,off-2.json rows4=r4-1.json,r4-2.json
  python scripts/bench/lookup_served_ab.py --followup-kl fup-off.json fup-rows4.json fup-off-chunk1024.json

Protocol: both arms on the same tree, in both orders, and check the server
log for "MLX lookup decode enabled" in every arm that sets the flag.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

FOLLOW = "<|im_end|>\n<|im_start|>user\nSay in one sentence what you just did and what comes next.<|im_end|>\n<|im_start|>assistant\n"


def stream_turn(port, ids, max_new, timeout, sampling):
    body = {"input_ids": ids, "sampling_params": {**sampling, "max_new_tokens": max_new}, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, out, meta = [], [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            meta = chunk.get("meta_info") or meta
            n = meta.get("completion_tokens", 0)
            new = chunk.get("output_ids") or []
            out = list(new) if len(new) == n else out + list(new)
            if n and (not stamps or n > stamps[-1][1]):
                stamps.append((time.perf_counter(), n))
    (t_first, n_first), (t_last, n_last) = stamps[0], stamps[-1]
    decode_s = t_last - t_first
    return {
        "ttft_s": round(t_first - t0, 3), "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"), "tokens": n_last, "first_chunk_tokens": n_first,
        "decode_s": round(decode_s, 4), "chunks": len(stamps),
        "tok_s": round((n_last - n_first) / decode_s, 3) if decode_s > 0 else None,
        "finish": (meta.get("finish_reason") or {}).get("type"), "ids": out,
    }


def followup(port, ids, new_tokens, topk):
    import probe_followup_fidelity as pff

    hit, cached = pff.generate(port, ids, new_tokens, topk)
    pff.flush(port)
    cold, _ = pff.generate(port, ids, new_tokens, topk)
    pff.flush(port)
    kls, part = [], None
    for i, (c, h) in enumerate(zip(cold, hit)):
        kls.append(pff.kl(c, h))
        if c["tid"] != h["tid"]:
            part = i
            break
    kls = [k for k in kls if k is not None]
    return {
        "prompt_tokens": len(ids), "cached_tokens": cached, "steps": len(kls),
        "kl_mean": round(statistics.fmean(kls), 6) if kls else None,
        "kl_max": round(max(kls), 6) if kls else None, "part_step": part,
        "cold_ids": [s["tid"] for s in cold], "hit_ids": [s["tid"] for s in hit],
        "prompt_sha": hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16],
        "cold_top": [s["top"] for s in cold], "hit_top": [s["top"] for s in hit],
    }


def followup_kl(paths):
    """Every arm's cold and hit follow-up against the first arm's cold run.

    A step counts while the run's own tokens equal the reference's, so both
    saw the same input. Follow-ups are matched on the prompt's hash: an arm
    whose turn came out as another text has no row.
    """
    import probe_followup_fidelity as pff

    runs = [json.load(open(p)) for p in paths]
    for ref in runs[0]["followups"]:
        ref_steps = [{"tid": t, "top": top} for t, top in zip(ref["cold_ids"], ref["cold_top"])]
        print(f"{ref['session']}@{ref['turn']}, {ref['prompt_tokens']} prompt tokens")
        for run in runs:
            f = next((x for x in run["followups"] if x.get("prompt_sha") == ref["prompt_sha"]), None)
            if f is None:
                print(f"  {run['tag']:<18} another prompt (the turn's text differs)")
                continue
            for kind in ("cold", "hit"):
                if run is runs[0] and kind == "cold":
                    continue
                kls = []
                for r, tid, top in zip(ref_steps, f[kind + "_ids"], f[kind + "_top"]):
                    k = pff.kl(r, {"top": top})
                    if k is not None:
                        kls.append(k)
                    if tid != r["tid"]:
                        break
                cached = f["cached_tokens"] if kind == "hit" else 0
                print(f"  {run['tag']:<18} {kind:<4} cached {cached:>6}  step 0 {kls[0]:.6f}  "
                      f"mean {statistics.fmean(kls):.6f}  max {max(kls):.6f}  over {len(kls)} steps")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    base = runs[0]
    key = lambda r: (r["session"], r["turn"], r["pass"])
    print(f"{'arm':<10}" + "".join(f"{k[0][:18] + '@' + str(k[1]) + ' p' + str(k[2]):>26}" for k in map(key, base["turns"])) + f"{'pooled':>16}")
    for run in runs:
        rows = {key(r): r for r in run["turns"]}
        cells = []
        for k in map(key, base["turns"]):
            r = rows.get(k)
            cells.append(f"{r['tok_s']:>8.2f} tok/s {r['tokens']:>4} tok" if r else "-")
        n = sum(r["tokens"] - r["first_chunk_tokens"] for r in run["turns"])
        dt = sum(r["decode_s"] for r in run["turns"])
        b_n = sum(r["tokens"] - r["first_chunk_tokens"] for r in base["turns"])
        b_dt = sum(r["decode_s"] for r in base["turns"])
        print(f"{run['tag']:<10}" + "".join(f"{c:>26}" for c in cells) + f"{n / dt:>8.2f} x{(n / dt) / (b_n / b_dt):.3f}")
    print()
    for run in runs:
        rows = {key(r): r for r in run["turns"]}
        notes = []
        for k in map(key, base["turns"]):
            r, b = rows.get(k), {key(x): x for x in base["turns"]}[k]
            if r is None:
                continue
            d = first_diff(r["ids"], b["ids"])
            notes.append("same" if d is None else f"differs at {d}")
        print(f"{run['tag']:<10} text against {base['tag']}: " + ", ".join(notes))
        for f in run.get("followups", []):
            print(f"{'':<10} follow-up {f['session'][:18]}@{f['turn']}: {f['cached_tokens']} of {f['prompt_tokens']} prompt tokens from the cache, "
                  f"KL(cold || hit) mean {f['kl_mean']} max {f['kl_max']} over {f['steps']} steps, "
                  + ("same greedy tokens" if f["part_step"] is None else f"tokens part at step {f['part_step']}"))


def pool(specs):
    """Pooled decode rate per label over its result files; the first label is the baseline."""
    base = None
    for spec in specs:
        label, _, files = spec.partition("=")
        runs = [json.load(open(f)) for f in files.split(",")]
        turns = [r for run in runs for r in run["turns"]]
        n = sum(r["tokens"] - r["first_chunk_tokens"] for r in turns)
        dt = sum(r["decode_s"] for r in turns)
        base = base or n / dt
        each = ", ".join(f"{run['tag']} {run['pooled_tok_s']:.2f}" for run in runs)
        print(f"{label:<12} {len(turns):>3} requests {n:>6} tokens {n / dt:>7.2f} tok/s x{(n / dt) / base:.3f}   ({each})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--model", help="served model: HF repo in the local cache, or a directory")
    ap.add_argument("--session", nargs="+", help="omp session files (.jsonl), each as <file>@<turn,turn>")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=24000)
    ap.add_argument("--gen-prefix", default="<think>\n\n</think>\n\n",
                    help="text after the assistant header, the thinking-off generation prompt")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    ap.add_argument("--top-k", type=int, default=0, help="sent when above 0")
    ap.add_argument("--top-p", type=float, default=0.0, help="sent when above 0")
    ap.add_argument("--stop", action="append", default=[],
                    help="stop string: a finish the drafts cannot foresee (tokens verified ahead are dropped)")
    ap.add_argument("--followup", action="store_true", help="grade the radix state each turn leaves (untimed arm)")
    ap.add_argument("--followup-tokens", type=int, default=48)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs="+", help="result files; the first is the baseline")
    ap.add_argument("--pool", nargs="+", help="<label>=<file,file>: pooled rate per label, the first is the baseline")
    ap.add_argument("--followup-kl", nargs="+", help="result files of --followup arms; the first one's cold run is the reference")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if a.pool:
        return pool(a.pool)
    if a.followup_kl:
        return followup_kl(a.followup_kl)
    sampling = {"temperature": a.temperature}
    if a.top_k > 0:
        sampling["top_k"] = a.top_k
    if a.top_p > 0:
        sampling["top_p"] = a.top_p
    if a.stop:
        sampling["stop"] = a.stop

    import mlx_decode_census as census
    import prompt_lookup_acceptance as pla
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(census.resolve_model_dir(a.model), "tokenizer.json"))
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids if t else []
    eos = {i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) if i is not None}
    prefix = enc(a.gen_prefix)
    turns = []
    for spec in a.session:
        path, _, own = spec.partition("@")
        seq, spans, _ = pla.load_session(path, enc)
        name = os.path.basename(os.path.dirname(path))
        for t in sorted(int(x) for x in own.split(",")):
            if t >= len(spans) or spans[t][0] + len(prefix) > a.max_context:
                print(f"skip turn {t} of {name}", flush=True)
                continue
            turns.append((name, t, seq[: spans[t][0]] + prefix))

    result = {"tag": a.tag, "model": a.model, "max_new": a.max_new, "sampling": sampling,
              "gen_prefix": a.gen_prefix, "turns": [], "followups": []}
    for p in range(1, a.passes + 1):
        for name, t, ctx in turns:
            r = stream_turn(a.port, ctx, a.max_new, a.timeout, sampling)
            r.update(session=name, turn=t, context=len(ctx))
            r["pass"] = p
            result["turns"].append(r)
            print(f"pass {p} {name} turn {t}: context {len(ctx)}, cached {r['cached_tokens']}, ttft {r['ttft_s']:.1f}s, "
                  f"{r['tokens']} tok in {r['chunks']} chunks, {r['tok_s']} tok/s, finish {r['finish']}", flush=True)
            if a.followup and p == 1:
                out = r["ids"][:-1] if r["ids"] and r["ids"][-1] in eos else r["ids"]
                f = followup(a.port, ctx + out + enc(FOLLOW) + prefix, a.followup_tokens, a.topk)
                f.update(session=name, turn=t)
                result["followups"].append(f)
                print(f"  follow-up: {f['cached_tokens']} of {f['prompt_tokens']} prompt tokens from the cache, KL(cold || hit) mean "
                      f"{f['kl_mean']} max {f['kl_max']} over {f['steps']} steps, part step {f['part_step']}", flush=True)
    n = sum(r["tokens"] - r["first_chunk_tokens"] for r in result["turns"])
    dt = sum(r["decode_s"] for r in result["turns"])
    result["pooled_tok_s"] = round(n / dt, 3)
    print(f"pooled: {n} tokens, {n / dt:.2f} tok/s", flush=True)
    if a.out:
        json.dump(result, open(a.out, "w"))


if __name__ == "__main__":
    main()
