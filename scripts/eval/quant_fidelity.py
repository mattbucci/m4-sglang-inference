#!/usr/bin/env python3
"""Quantization fidelity through the serving stack: teacher-forced NLL and KL to a reference.

Held-out prose (Salesforce/wikitext test) and code (this repo's scripts) are
tokenized once; for each sampled position the server scores the next token
(`token_ids_logprob`) and reports its top-k next-token distribution, with
`max_new_tokens` 1. Positions within a document ascend, so each request is a
radix-cache hit extending the previous prefix by `--stride` tokens.

Dump mode (default): per position, the true token's logprob and the top-k.
Reference mode (`--ref-dump ref.json`): per position, the model's exact
logprobs for the reference's top-k tokens, giving KL(ref || model) over the
reference's top-k support (both renormalized) and top-1 agreement.

    python scripts/eval/quant_fidelity.py --port 23334 --tokenizer <dir> --tag incumbent --out fid-incumbent.json
    python scripts/eval/quant_fidelity.py --port 23334 --tokenizer <dir> --tag incumbent --ref-dump fid-8bit.json --out fid-incumbent-vs8bit.json
"""
import argparse
import glob
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def post(port, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def corpus(seed, n_docs, max_tokens, tok):
    rng = random.Random(seed)
    prose = []
    try:
        import datasets
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        prose = [t for t in ds["text"] if len(t) > 2500]
    except Exception as e:
        print(f"[corpus] wikitext unavailable ({e!r}); prose from repo markdown", file=sys.stderr)
        for f in glob.glob(os.path.join(REPO, "*.md")) + glob.glob(os.path.join(REPO, "patches", "*.md")):
            prose.append(open(f, errors="replace").read())
    code_files = sorted((f for f in glob.glob(os.path.join(REPO, "scripts", "**", "*.py"), recursive=True)
                         if os.path.getsize(f) > 6000), key=os.path.getsize, reverse=True)
    rng.shuffle(prose)
    rng.shuffle(code_files)
    docs = []
    for i in range(n_docs):
        if i % 2 == 0 and prose:
            text, domain = prose[(i // 2) % len(prose)], "prose"
        else:
            text, domain = open(code_files[(i // 2) % len(code_files)], errors="replace").read(), "code"
        ids = tok(text, add_special_tokens=False)["input_ids"][:max_tokens]
        docs.append({"domain": domain, "ids": ids, "source": text[:60].replace("\n", " ")})
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir or repo of the served model")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dump", default=None, help="dump JSON of the reference model: score its top-k tokens")
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--max-doc-tokens", type=int, default=1024)
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit-positions", type=int, default=0, help="smoke tests: stop after N positions")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ref = json.load(open(a.ref_dump)) if a.ref_dump else None
    if ref:
        docs = [{"domain": d["domain"], "ids": d["ids"], "source": d["source"]} for d in ref["docs"]]
    else:
        docs = corpus(a.seed, a.docs, a.max_doc_tokens, tok)
    results = {"tag": a.tag, "ref": a.ref_dump, "stride": a.stride, "topk": a.topk, "docs": []}
    n = 0
    t0 = time.time()
    for di, d in enumerate(docs):
        ids = d["ids"]
        rows = []
        positions = list(range(a.start, len(ids) - 1, a.stride))
        if ref:
            ref_rows = {r["p"]: r for r in ref["docs"][di]["rows"]}
            positions = [p for p in positions if p in ref_rows]
        for p in positions:
            true_tid = ids[p]
            want = [true_tid]
            if ref:
                want = list(dict.fromkeys([true_tid] + [t for t, _ in ref_rows[p]["top"]]))
            body = {"input_ids": ids[:p], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True, "top_logprobs_num": a.topk, "token_ids_logprob": want}
            r = post(a.port, body)
            m = r["meta_info"]
            top = [(t[1], t[0]) for t in m["output_top_logprobs"][0]]
            scored = {t[1]: t[0] for t in m["output_token_ids_logprobs"][0]}
            row = {"p": p, "true": true_tid, "true_lp": scored.get(true_tid), "argmax": top[0][0] if top else None,
                   "top": top, "cached": m.get("cached_tokens")}
            if ref:
                row["scored"] = scored
            rows.append(row)
            n += 1
            if a.limit_positions and n >= a.limit_positions:
                break
        results["docs"].append({"domain": d["domain"], "ids": ids, "source": d["source"], "rows": rows})
        print(f"[{a.tag}] doc {di} {d['domain']:5s} {len(rows)} positions, {time.time() - t0:.0f}s elapsed", flush=True)
        if a.limit_positions and n >= a.limit_positions:
            break
    results["summary"] = summarize(results, ref)
    json.dump(results, open(a.out, "w"))
    print(json.dumps(results["summary"], indent=1))
    return 0


def summarize(results, ref):
    out = {}
    for domain in ("all", "prose", "code"):
        nll, agree, kls, ref_agree = [], [], [], []
        for di, d in enumerate(results["docs"]):
            if domain != "all" and d["domain"] != domain:
                continue
            ref_rows = {r["p"]: r for r in ref["docs"][di]["rows"]} if ref else {}
            for r in d["rows"]:
                if r["true_lp"] is None:
                    continue
                nll.append(-r["true_lp"])
                agree.append(r["argmax"] == r["true"])
                if ref and r["p"] in ref_rows and r.get("scored"):
                    rr = ref_rows[r["p"]]
                    ref_lp = {t: lp for t, lp in rr["top"]}
                    toks = [t for t in ref_lp if t in r["scored"]]
                    if len(toks) >= 2:
                        rz = math.log(sum(math.exp(ref_lp[t]) for t in toks))
                        mz = math.log(sum(math.exp(r["scored"][t]) for t in toks))
                        kls.append(sum(math.exp(ref_lp[t] - rz) * ((ref_lp[t] - rz) - (r["scored"][t] - mz)) for t in toks))
                        ref_agree.append(r["argmax"] == rr["argmax"])
        if not nll:
            continue
        s = {"positions": len(nll), "nll_mean": statistics.mean(nll), "ppl": math.exp(statistics.mean(nll)),
             "nll_se": statistics.pstdev(nll) / math.sqrt(len(nll)), "top1_acc": statistics.mean(agree)}
        if kls:
            s.update({"kl_ref_mean": statistics.mean(kls), "kl_ref_median": statistics.median(kls),
                      "kl_ref_se": statistics.pstdev(kls) / math.sqrt(len(kls)), "argmax_agree_ref": statistics.mean(ref_agree)})
        out[domain] = s
    return out


if __name__ == "__main__":
    sys.exit(main())
