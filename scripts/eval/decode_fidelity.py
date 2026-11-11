#!/usr/bin/env python3
"""Decode-path fidelity through the serving stack: greedy steps against a reference arm.

`quant_fidelity.py` scores one token per request, so every logit it reads comes
out of a prefill. A change that only touches decode steps (a fused decode
kernel, a decode-only fold) is invisible to it. This probe reads the decode
loop instead: each document's prefix is prefilled once and the server decodes
`--new-tokens` greedy steps, returning the top-k distribution of every step.

Dump mode (default): per step, the chosen token and the top-k.
Reference mode (`--ref-dump ref.json`): the same prefixes are decoded again and
compared step by step. Two arms that fed the same tokens see the same input at
every step up to and including the first one where they choose differently, so
on that stretch the distributions differ only by arithmetic. After a divergence
the prompt is extended with the reference's tokens and decoding resumes, so
every reference step is compared against a step with the same input.
`--restart-every K` re-prompts on a schedule as well, which measures a restart
by itself: the arm resumes from a prefill (or a radix hit) of tokens the
reference decoded one by one. Reported:
KL(ref || arm) over the shared top-k (both renormalized), the shift of the
reference's top-1 logprob, how often the arms choose the same token, and the
reference's top-1 to top-2 margin at each divergence. A divergence at a margin
near zero is a tie flipping; one at a wide margin is a defect. A re-prompt
restarts the arm from a prefill of tokens the reference decoded one by one, so
steps after the first divergence also carry that restart noise; the
`first_segment_*` figures cover only the steps before it.

`--split-prefill N` prefills all but the last N prefix tokens in a first
request, so the decode starts from a radix-cache hit rather than a cold
prefill. Run against the reference build, that arm is the noise floor: the
same arithmetic on a different chunk grid.

    python scripts/eval/decode_fidelity.py --port 23334 --tokenizer <dir> --tag base --out dfid-base.json
    python scripts/eval/decode_fidelity.py --port 23334 --tokenizer <dir> --tag arm --ref-dump dfid-base.json --out dfid-arm.json
    python scripts/eval/decode_fidelity.py --port 23334 --tokenizer <dir> --tag floor --ref-dump dfid-base.json --split-prefill 37 --out dfid-floor.json
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_fidelity import corpus, post  # noqa: E402


def decode(port, ids, new_tokens, topk, split=0):
    """Greedy steps after `ids`: [{"tid": token, "top": [[token, logprob], ...]}, ...]."""
    if split and len(ids) > split + 8:
        post(port, {"input_ids": ids[:-split], "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": new_tokens, "temperature": 0},
            "return_logprob": True, "top_logprobs_num": topk}
    m = post(port, body)["meta_info"]
    return [{"tid": chosen[1], "top": [[t[1], t[0]] for t in top]}
            for chosen, top in zip(m["output_token_logprobs"], m["output_top_logprobs"])]


def compare(ref_step, step):
    """One step of both arms on the same input."""
    ref_lp = {t: lp for t, lp in ref_step["top"]}
    lp = {t: l for t, l in step["top"]}
    shared = [t for t in ref_lp if t in lp]
    row = {"same": ref_step["tid"] == step["tid"], "shared": len(shared)}
    if len(shared) >= 2:
        rz = math.log(sum(math.exp(ref_lp[t]) for t in shared))
        mz = math.log(sum(math.exp(lp[t]) for t in shared))
        row["kl"] = sum(math.exp(ref_lp[t] - rz) * ((ref_lp[t] - rz) - (lp[t] - mz)) for t in shared)
    top1 = ref_step["top"][0][0]
    if top1 in lp:
        row["dlp_top1"] = abs(ref_lp[top1] - lp[top1])
    if not row["same"] and len(ref_step["top"]) >= 2:
        row["ref_margin"] = ref_step["top"][0][1] - ref_step["top"][1][1]
    return row


def against_reference(port, prefix, ref_steps, topk, split, max_resyncs, restart_every=0):
    """Rows for every reference step, plus the number of re-prompts.

    `seg` on a row counts the re-prompts before it (0 = the first request),
    `since` the steps since that request started.
    """
    rows, pos, restarts, diverged = [], 0, 0, 0
    while pos < len(ref_steps):
        want = len(ref_steps) - pos
        if restart_every:
            want = min(want, restart_every)
        fed = prefix + [s["tid"] for s in ref_steps[:pos]]
        steps = decode(port, fed, want, topk, split if pos == 0 else 0)
        if not steps:
            break
        advanced = len(steps)
        for j, step in enumerate(steps):
            row = compare(ref_steps[pos + j], step)
            row.update({"s": pos + j, "seg": restarts, "since": j})
            rows.append(row)
            if not row["same"]:
                row["arm_top"] = step["top"][:3]
                advanced = j + 1
                break
        pos += advanced
        if pos >= len(ref_steps):
            break
        if rows[-1]["same"] and advanced < want:
            break  # the arm stopped early on the same tokens
        if not rows[-1]["same"]:
            diverged += 1
            if diverged > max_resyncs:
                break
        restarts += 1
    return rows, restarts


def summarize(results):
    out = {}
    for domain in ("all", "prose", "code"):
        rows = [r for d in results["docs"] if domain in ("all", d["domain"]) for r in d["rows"]]
        if not rows:
            continue
        ref_steps = sum(d["ref_steps"] for d in results["docs"] if domain in ("all", d["domain"]))
        kls = [r["kl"] for r in rows if "kl" in r]
        dlp = [r["dlp_top1"] for r in rows if "dlp_top1" in r]
        margins = sorted(r["ref_margin"] for r in rows if "ref_margin" in r)
        diverged = sum(not r["same"] for r in rows)
        s = {"steps": len(rows), "coverage": len(rows) / ref_steps if ref_steps else None,
             "same_token": 1 - diverged / len(rows), "divergences": diverged,
             "divergences_per_1000": 1000 * diverged / len(rows)}
        if kls:
            ranked = sorted(kls)
            s.update({"kl_mean": statistics.mean(kls), "kl_median": statistics.median(kls),
                      "kl_p90": ranked[int(0.90 * (len(ranked) - 1))], "kl_p99": ranked[int(0.99 * (len(ranked) - 1))],
                      "kl_max": ranked[-1]})
        if dlp:
            s.update({"dlp_top1_mean": statistics.mean(dlp), "dlp_top1_p99": sorted(dlp)[int(0.99 * (len(dlp) - 1))],
                      "dlp_top1_max": max(dlp)})
        first = [r for r in rows if not r.get("seg")]
        first_kl = [r["kl"] for r in first if "kl" in r]
        if first_kl:
            s.update({"first_segment_steps": len(first), "first_segment_kl_mean": statistics.mean(first_kl),
                      "first_segment_kl_max": max(first_kl),
                      "first_segment_dlp_top1_max": max((r["dlp_top1"] for r in first if "dlp_top1" in r), default=None)})
        later = [r["kl"] for r in rows if r.get("seg") and "kl" in r]
        if later:
            s.update({"restarts": sum(d["resyncs"] for d in results["docs"] if domain in ("all", d["domain"])),
                      "after_restart_kl_mean": statistics.mean(later),
                      "after_restart_kl_mean_first_8": statistics.mean(
                          [r["kl"] for r in rows if r.get("seg") and "kl" in r and r["since"] < 8] or [0.0])})
        if margins:
            s.update({"wide_margin_divergences": sum(m > 1.0 for m in margins)})
            s.update({"margin_median": statistics.median(margins), "margin_max": margins[-1],
                      "margins": [round(m, 4) for m in margins]})
        out[domain] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir or repo of the served model")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dump", default=None, help="dump JSON of the reference arm")
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--prefix", type=int, default=384, help="prompt tokens per document")
    ap.add_argument("--new-tokens", type=int, default=192)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--split-prefill", type=int, default=0, help="prefill all but the last N prefix tokens first")
    ap.add_argument("--max-resyncs", type=int, default=24, help="reference mode: re-prompts after a divergence, per document")
    ap.add_argument("--restart-every", type=int, default=0, help="reference mode: also re-prompt every N steps")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    ref = json.load(open(a.ref_dump)) if a.ref_dump else None
    if ref:
        docs = [{"domain": d["domain"], "ids": d["ids"], "source": d["source"]} for d in ref["docs"]]
    else:
        from transformers import AutoTokenizer
        docs = corpus(a.seed, a.docs, a.prefix, AutoTokenizer.from_pretrained(a.tokenizer))
    results = {"tag": a.tag, "ref": a.ref_dump, "topk": a.topk, "split_prefill": a.split_prefill,
               "restart_every": a.restart_every, "docs": []}
    t0 = time.time()
    for di, d in enumerate(docs):
        entry = {"domain": d["domain"], "ids": d["ids"], "source": d["source"]}
        if ref:
            ref_steps = ref["docs"][di]["steps"]
            entry["rows"], entry["resyncs"] = against_reference(
                a.port, d["ids"], ref_steps, a.topk, a.split_prefill, a.max_resyncs, a.restart_every)
            entry["ref_steps"] = len(ref_steps)
            note = f"{len(entry['rows'])}/{len(ref_steps)} steps, {sum(not r['same'] for r in entry['rows'])} divergences"
        else:
            entry["steps"] = decode(a.port, d["ids"], a.new_tokens, a.topk, a.split_prefill)
            note = f"{len(entry['steps'])} steps"
        results["docs"].append(entry)
        print(f"[{a.tag}] doc {di} {d['domain']:5s} {note}, {time.time() - t0:.0f}s elapsed", flush=True)
    if ref:
        results["summary"] = summarize(results)
        print(json.dumps({k: {m: v for m, v in s.items() if m != "margins"}
                          for k, s in results["summary"].items()}, indent=1))
        print(f"[{a.tag}] margins at divergence (nats): {results['summary'].get('all', {}).get('margins', [])}")
    json.dump(results, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
