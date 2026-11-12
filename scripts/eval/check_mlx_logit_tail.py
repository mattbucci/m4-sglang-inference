#!/usr/bin/env python3
"""Tail mass of an MLX checkpoint's next-token distribution. No server.

A healthy checkpoint puts all but a sliver of its probability on a few
hundred tokens. `mlx-community/Qwen3-32B-4bit-DWQ` does not: its logits
span about half the usual range, and at temperature 1.0 it holds 11 to 56 %
of its probability beyond the top 1000 tokens. Greedy decode and every
greedy gate are unaffected. A request sampled at temperature 1.0 without
top-k returns token soup, and that is what a chat request with no sampling
fields gets when the checkpoint ships no generation_config.json.

The script teacher-forces a prose sentence and a code line after a chat
prompt, in one raw forward each, and reports per position the probability
beyond the top 1000 tokens at temperature 1.0 and at the fallback
temperature (float32 softmax). It also says whether the checkpoint ships a
generation_config.json, since the two together make the hazard.

  HEAVY TAIL  mean mass beyond the top 1000 above --threshold (exit 1)
  OK          otherwise (exit 0)
  NOT LOADED  neither mlx-lm nor mlx-vlm loads the checkpoint raw (exit 2)

Usage:
  python scripts/eval/check_mlx_logit_tail.py mlx-community/Qwen3-32B-4bit-DWQ
  python scripts/eval/check_mlx_logit_tail.py <local dir> [--json out.json] [-v]
"""
import argparse
import glob
import json
import os
import sys

import mlx.core as mx

CASES = [
    ("Write one sentence about the capital of France.",
     "The capital of France is Paris, a city renowned for its historical landmarks"),
    ("Write a Python function add(a, b) that returns their sum. Code only.",
     "def add(a, b):\n    return a + b"),
]


def load(path):
    """(callable ids -> logits, tokenizer, loader name)."""
    try:
        from mlx_lm import load as lm_load

        model, tok = lm_load(path)
        return (lambda ids: model(ids)), tok, "mlx-lm"
    except Exception as first:
        try:
            from mlx_vlm import load as vlm_load

            model, processor = vlm_load(path)
        except Exception:
            raise first
        lang = model.language_model

        def forward(ids):
            out = lang(ids)
            return out.logits if hasattr(out, "logits") else out

        return forward, processor.tokenizer, "mlx-vlm"


def has_generation_config(path):
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "generation_config.json"))
    hub = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub")
    snaps = glob.glob(os.path.join(hub, "models--" + path.replace("/", "--"), "snapshots", "*"))
    return any(os.path.exists(os.path.join(s, "generation_config.json")) for s in snaps)


def chat_prompt(tok, question):
    msgs = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def beyond(row, temperature, k=1000):
    scaled = row / temperature
    probs = mx.exp(scaled - mx.logsumexp(scaled))
    top = mx.topk(probs, k)
    return 1.0 - float(top.sum()), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="mean mass beyond the top 1000 at temperature 1.0 that fails")
    ap.add_argument("--temperature", type=float, default=0.6, help="second temperature to report")
    ap.add_argument("--json", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    try:
        forward, tok, loader = load(a.model)
    except Exception as e:
        # Architectures only the server's MLX backend registers (Flint) land here.
        print(f"{a.model}: not loadable by mlx-lm or mlx-vlm ({type(e).__name__})")
        print("VERDICT: NOT LOADED")
        return 2
    rows = []
    for question, reply in CASES:
        p_ids = tok.encode(chat_prompt(tok, question), add_special_tokens=False)
        r_ids = tok.encode(reply, add_special_tokens=False)
        logits = forward(mx.array([p_ids + r_ids]))[0]
        for i, t in enumerate(r_ids):
            row = logits[len(p_ids) - 1 + i].astype(mx.float32)
            tail_1, probs = beyond(row, 1.0)
            tail_t, _ = beyond(row, a.temperature)
            rows.append({
                "token": tok.decode([t]), "p_token": round(float(probs[t]), 4),
                "top1": round(float(probs.max()), 4), "beyond_1000": round(tail_1, 4),
                "beyond_1000_at_t": round(tail_t, 4), "max_logit": round(float(row.max()), 2),
                "median_logit": round(float(mx.sort(row)[row.shape[0] // 2]), 2),
            })
            if a.verbose:
                r = rows[-1]
                print(f"  {r['token']!r:16s} p={r['p_token']:.4f} top1={r['top1']:.4f} "
                      f"beyond1000={r['beyond_1000']:.4f} at T={a.temperature}: {r['beyond_1000_at_t']:.4f} "
                      f"max logit {r['max_logit']:.2f} median {r['median_logit']:.2f}")

    n = len(rows)
    mean_tail = sum(r["beyond_1000"] for r in rows) / n
    max_tail = max(r["beyond_1000"] for r in rows)
    mean_tail_t = sum(r["beyond_1000_at_t"] for r in rows) / n
    spans = sorted(r["max_logit"] - r["median_logit"] for r in rows)
    gen = has_generation_config(a.model)
    heavy = mean_tail > a.threshold
    summary = {
        "model": a.model, "loader": loader, "positions": n,
        "generation_config": gen, "mean_beyond_1000": round(mean_tail, 4),
        "max_beyond_1000": round(max_tail, 4),
        f"mean_beyond_1000_at_{a.temperature}": round(mean_tail_t, 4),
        "median_logit_span": round(spans[n // 2], 2),
        "verdict": "HEAVY TAIL" if heavy else "OK",
    }
    print(f"{a.model}  ({loader}, {n} positions)")
    print(f"  generation_config.json: {'present' if gen else 'MISSING'}")
    print(f"  mass beyond the top 1000 at temperature 1.0: mean {mean_tail:.4f}, max {max_tail:.4f}")
    print(f"  mass beyond the top 1000 at temperature {a.temperature}: mean {mean_tail_t:.4f}")
    print(f"  max logit minus median logit, median over positions: {spans[n // 2]:.1f}")
    if heavy and not gen:
        print("  a chat request with no sampling fields decodes at temperature 1.0 over this tail:")
        print("  give the preset --sampling-defaults-fallback (launch.sh SAMPLING_FALLBACK)")
    print(f"VERDICT: {summary['verdict']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({**summary, "rows": rows}, fh, indent=1)
    return 1 if heavy else 0


if __name__ == "__main__":
    sys.exit(main())
