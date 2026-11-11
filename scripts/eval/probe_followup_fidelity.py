#!/usr/bin/env python3
"""Does a chat follow-up served from a radix prefix hit see the distribution a
cold prefill of the same prompt gives?

Per case: a first user turn holding a document of `depth` tokens is answered
cold; the follow-up turn appends that reply and a new user message holding a
second excerpt of `tail` tokens, rendered through the model's chat template.
The follow-up runs once on the cached history (a radix hit: only the tokens
after the cached prefix are forwarded) and once after a cache flush (cold).
Both are greedy with top-k logprobs, so every step the two runs share is the
same input and their distributions compare directly: KL(cold || hit) over the
shared top-k support, the step at which the greedy tokens part, and how many
prompt tokens the hit served from the cache.

`--ref <dump.json>` also compares each cold follow-up with the cold follow-up
of another run of this probe (a server on a different prefill chunk size is
the accepted noise class for a hit).

`probe_multiturn_hit_determinism.py` compares greedy token ids only; this
probe compares distributions, which sees a shift that leaves a wide-margin
argmax unchanged.

    python scripts/eval/probe_followup_fidelity.py --port 23334 --tokenizer <dir> --out followup.json
    python scripts/eval/probe_followup_fidelity.py --port 23334 --tokenizer <dir> --ref followup-chunk2048.json --out followup.json
"""
import argparse
import json
import math
import os
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_fidelity import corpus, post  # noqa: E402

SYSTEM = "You are a careful assistant. Answer briefly and only from the text you are given."
ASK_1 = "\n\nDescribe the text above in one short sentence."
ASK_2 = "\n\nIn two sentences: what does this second excerpt cover, and how does it relate to the first text?"


def flush(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/flush_cache", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


def generate(port, ids, new_tokens, topk):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": new_tokens, "temperature": 0},
            "return_logprob": True, "top_logprobs_num": topk}
    m = post(port, body, timeout=3600)["meta_info"]
    steps = [{"tid": chosen[1], "top": [[t[1], t[0]] for t in top]}
             for chosen, top in zip(m["output_token_logprobs"], m["output_top_logprobs"])]
    return steps, m.get("cached_tokens")


def kl(ref_step, step):
    ref_lp = {t: lp for t, lp in ref_step["top"]}
    lp = {t: l for t, l in step["top"]}
    shared = [t for t in ref_lp if t in lp]
    if len(shared) < 2:
        return None
    rz = math.log(sum(math.exp(ref_lp[t]) for t in shared))
    mz = math.log(sum(math.exp(lp[t]) for t in shared))
    return sum(math.exp(ref_lp[t] - rz) * ((ref_lp[t] - rz) - (lp[t] - mz)) for t in shared)


def against(ref, run):
    """Steps the two greedy runs share (same input): KLs, and where they part."""
    kls, agree = [], 0
    for r, s in zip(ref, run):
        k = kl(r, s)
        if k is not None:
            kls.append(k)
        if r["tid"] != s["tid"]:
            break
        agree += 1
    return {"agree": agree, "steps": min(len(ref), len(run)), "kl_first": kls[0] if kls else None,
            "kl_mean": statistics.fmean(kls) if kls else None, "kl_max": max(kls) if kls else None}


def render(tok, messages, thinking):
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=thinking)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def summarize(rows, key):
    out = {}
    for tail in sorted({r["tail"] for r in rows}):
        sel = [r[key] for r in rows if r["tail"] == tail and r.get(key)]
        if not sel:
            continue
        first = [s["kl_first"] for s in sel if s["kl_first"] is not None]
        mean = [s["kl_mean"] for s in sel if s["kl_mean"] is not None]
        out[str(tail)] = {
            "cases": len(sel),
            "identical_tokens": sum(s["agree"] == s["steps"] for s in sel),
            "agree_steps_mean": statistics.fmean(s["agree"] for s in sel),
            "kl_first_median": statistics.median(first) if first else None,
            "kl_first_mean": statistics.fmean(first) if first else None,
            "kl_mean": statistics.fmean(mean) if mean else None,
            "kl_max": max((s["kl_max"] for s in sel if s["kl_max"] is not None), default=None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir or repo of the served model")
    ap.add_argument("--depths", default="1024,6144", help="first-turn document tokens")
    ap.add_argument("--tails", default="32,256,1024,3000", help="follow-up excerpt tokens")
    ap.add_argument("--seeds", type=int, default=3, help="document pairs per depth and tail")
    ap.add_argument("--reply-tokens", type=int, default=48)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--thinking", action="store_true", help="render the follow-up with thinking enabled")
    ap.add_argument("--ref", default=None, help="dump of another run: compare the cold follow-ups")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    depths = [int(x) for x in a.depths.split(",")]
    tails = [int(x) for x in a.tails.split(",")]
    docs = corpus(0, 2 * a.seeds * len(depths) * len(tails), max(depths + tails), tok)
    ref = {(r["depth"], r["tail"], r["seed"]): r for r in json.load(open(a.ref))["rows"]} if a.ref else {}
    rows, n = [], 0
    for depth in depths:
        for tail in tails:
            for seed in range(a.seeds):
                first, second = docs[n], docs[n + 1]
                n += 2
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": tok.decode(first["ids"][:depth]) + ASK_1}]
                ids1 = render(tok, msgs, False)
                flush(a.port)
                reply, _ = generate(a.port, ids1, a.reply_tokens, 1)
                text = tok.decode([s["tid"] for s in reply], skip_special_tokens=True).strip() or "Noted."
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": tok.decode(second["ids"][:tail]) + ASK_2}]
                ids2 = render(tok, msgs, a.thinking)
                hit, cached = generate(a.port, ids2, a.new_tokens, a.topk)
                hit2, cached2 = generate(a.port, ids2, a.new_tokens, a.topk)
                flush(a.port)
                cold, cached_cold = generate(a.port, ids2, a.new_tokens, a.topk)
                row = {"depth": depth, "tail": tail, "seed": seed, "domains": [first["domain"], second["domain"]],
                       "first_prompt_tokens": len(ids1), "followup_prompt_tokens": len(ids2),
                       "cached": cached, "forwarded": len(ids2) - (cached or 0), "cached_repeat": cached2,
                       "cached_cold": cached_cold, "ids_sha": hash(tuple(ids2)) & 0xFFFFFFFF,
                       "hit_vs_cold": against(cold, hit), "repeat_vs_cold": against(cold, hit2),
                       "cold": cold, "hit": hit}
                r = ref.get((depth, tail, seed))
                if r and len(r["cold"]) and r["followup_prompt_tokens"] == len(ids2):
                    row["cold_vs_ref_cold"] = against(r["cold"], cold)
                    row["hit_vs_ref_cold"] = against(r["cold"], hit)
                rows.append(row)
                hc = row["hit_vs_cold"]
                extra = ""
                if "cold_vs_ref_cold" in row:
                    rc = row["cold_vs_ref_cold"]
                    extra = f" | cold vs ref cold: agree {rc['agree']}/{rc['steps']} KL first {rc['kl_first']:.2e} mean {rc['kl_mean']:.2e}"
                print(f"depth {depth} tail {tail} seed {seed}: prompt {len(ids2)}, cached {cached}, forwarded {row['forwarded']}; "
                      f"hit vs cold: agree {hc['agree']}/{hc['steps']} KL first {hc['kl_first']:.2e} mean {hc['kl_mean']:.2e} "
                      f"max {hc['kl_max']:.2e}{extra}", flush=True)
    summary = {k: summarize(rows, k) for k in ("hit_vs_cold", "repeat_vs_cold", "cold_vs_ref_cold", "hit_vs_ref_cold")}
    summary = {k: v for k, v in summary.items() if v}
    json.dump({"tag": a.tag, "args": vars(a), "summary": summary, "rows": rows}, open(a.out, "w"))
    print()
    for key, by_tail in summary.items():
        print(key)
        for tail, s in by_tail.items():
            print(f"  tail {tail:>5}: cases {s['cases']}, identical tokens {s['identical_tokens']}, "
                  f"agree steps {s['agree_steps_mean']:.1f}, KL first median {s['kl_first_median']:.2e} "
                  f"mean {s['kl_first_mean']:.2e}, KL mean {s['kl_mean']:.2e}, max {s['kl_max']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
