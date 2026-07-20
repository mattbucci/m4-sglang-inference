#!/usr/bin/env python3
"""Multi-needle positional recall ladder (depth-recall-probe).

Embeds 6 uniquely-keyed facts (vault -> 6-digit code) at fractional positions
{0.05, 0.25, 0.45, 0.65, 0.85, 0.98} inside seeded varied-vocabulary filler
sized by the 6-chars/token + 1.1 estimator, then asks all six questions in
ONE prompt (one prefill per rung). Greedy, enable_thinking:false; every rung
requires server-verified ``usage.prompt_tokens >= 0.95 x label`` or the rung
is VOID. The filler + codes derive from --seed, so re-runs are controlled
A/Bs (same seed => byte-identical prompt).

Receipts: benchmarks/quality/depth-recall/rung<label>-<tag>-seed<seed>.json

Usage:
  python scripts/eval/probe_depth_recall.py --port 23334 \
      --labels 8192,32768,65536,98304,131072,163840 --seed 42 --tag turboquant
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

VAULTS = ["Kestrel", "Meridian", "Obsidian", "Cinnabar", "Halcyon", "Zephyr"]
POSITIONS = [0.05, 0.25, 0.45, 0.65, 0.85, 0.98]
CHARS_PER_TOKEN = 6
SAFETY = 1.1

VOCAB = (
    "system record window process value table market region station final "
    "report number ancient garden silver monitor velvet crimson harbor lantern "
    "meadow copper thunder whisper granite orchid saffron timber quartz ember "
    "willow falcon marble canyon drift signal outpost archive channel beacon "
    "summit hollow prairie cascade tundra reef atlas cipher lattice prism"
).split()


def make_filler(target_chars: int, rng: random.Random) -> str:
    out = []
    n = 0
    while n < target_chars:
        k = rng.randint(8, 14)
        sentence = " ".join(rng.choice(VOCAB) for _ in range(k)).capitalize() + ". "
        out.append(sentence)
        n += len(sentence)
    return "".join(out)[:target_chars]


def build_prompt(label: int, seed: int):
    rng = random.Random(seed)
    codes = {v: f"{rng.randint(100000, 999999)}" for v in VAULTS}
    target_chars = int(label * CHARS_PER_TOKEN * SAFETY)
    filler = make_filler(target_chars, rng)
    facts = [
        f" The access code for vault {v} is {codes[v]}. "
        for v in VAULTS
    ]
    # insert facts at fractional char offsets, descending so offsets stay valid
    pieces = filler
    inserts = sorted(zip(POSITIONS, facts), key=lambda x: -x[0])
    realized = {}
    for frac, fact in inserts:
        off = int(len(filler) * frac)
        pieces = pieces[:off] + fact + pieces[off:]
        realized[fact.split()[5]] = frac  # vault name is the 6th word
    questions = "\n".join(
        f"{i + 1}. What is the access code for vault {v}?"
        for i, v in enumerate(VAULTS)
    )
    prompt = (
        "Here is a document:\n\n" + pieces +
        "\n\nAnswer the following questions using ONLY the document above. "
        "For each question reply with just the numeric code, one per line, "
        "in order.\n" + questions
    )
    return prompt, codes, realized


def run_rung(base_url: str, label: int, seed: int, tag: str, outdir: str,
             max_tokens: int = 200) -> dict:
    prompt, codes, realized = build_prompt(label, seed)
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            data = json.loads(r.read())
    except Exception as e:
        rec = {"label": label, "seed": seed, "tag": tag,
               "verdict": "SERVER_ERROR", "error": repr(e)}
        path = f"{outdir}/rung{label}-{tag}-seed{seed}.json"
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  rung {label:>7}  SERVER_ERROR  {e!r}  -> {path}")
        return rec
    elapsed = time.time() - t0
    prompt_tokens = data["usage"]["prompt_tokens"]
    content = data["choices"][0]["message"]["content"] or ""
    verified = prompt_tokens >= 0.95 * label
    found = {v: bool(re.search(rf"\b{codes[v]}\b", content)) for v in VAULTS}
    score = sum(found.values())
    rec = {
        "label": label, "seed": seed, "tag": tag,
        "prompt_tokens": prompt_tokens,
        "depth_verified": verified,
        "verdict": ("VOID_DEPTH" if not verified else f"{score}/6"),
        "score": score, "found": found, "codes": codes,
        "positions_char_frac": realized,
        "elapsed_s": round(elapsed, 1),
        "completion": content[:400],
        "finish_reason": data["choices"][0].get("finish_reason"),
    }
    path = f"{outdir}/rung{label}-{tag}-seed{seed}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  rung {label:>7}  in={prompt_tokens:>7}  verified={verified}  "
          f"score={score}/6  {elapsed:.0f}s  -> {path}")
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--labels", default="8192,32768,65536,98304,131072,163840")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="turboquant",
                   help="Receipt tag (KV dtype / config arm)")
    p.add_argument("--outdir", default="benchmarks/quality/depth-recall")
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    base = f"http://localhost:{args.port}"
    results = []
    for label in [int(x) for x in args.labels.split(",")]:
        results.append(run_rung(base, label, args.seed, args.tag,
                                args.outdir, args.max_tokens))
    bad = [r for r in results
           if r.get("verdict") in ("SERVER_ERROR", "VOID_DEPTH")]
    weak = [r for r in results if r.get("score", 0) <= 3 and r not in bad]
    print(f"== {len(results)} rungs, {len(bad)} void/error, "
          f"{len(weak)} at <=3/6 ==")
    return 1 if bad or weak else 0


if __name__ == "__main__":
    sys.exit(main())
