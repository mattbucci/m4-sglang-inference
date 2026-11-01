#!/usr/bin/env python3
"""Structured reads on DiffusionGemma through SGLang (patch 025).

A read seeds the diffusion canvas with an answer template, pins every
template position, leaves the answer slots as noise, runs a few denoising
steps, and reads each slot's distribution over its labels from the final
canvas's logprobs. Every label must be a single token, and the template
must fit the served canvas (256 tokens on DiffusionGemma).

Question kinds: noul (yes/no), choice (options), score (ordered levels).

    python scripts/eval/structured_reads.py --tokenizer google/diffusiongemma-26B-A4B-it \\
        --question "Is the patch complete?" --kind noul --context "$(cat notes.txt)"
    python scripts/eval/structured_reads.py ... --kind choice --options low medium high
    python scripts/eval/structured_reads.py ... --kind score --levels 1 2 3 4 5 --steps 4
"""
import argparse
import json
import math
import sys
import urllib.request


def single_token(tok, label: str) -> int:
    ids = tok(" " + label, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        ids = tok(label, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise SystemExit(f"label {label!r} is not a single token ({ids})")
    return ids[0]


def build_read(tok, canvas_len: int, slots: list[tuple[str, list[str]]]):
    """Canvas ids, pinned positions, slot positions and label ids per slot."""
    ids: list[int] = []
    slot_positions: list[int] = []
    label_ids: list[list[int]] = []
    for name, labels in slots:
        ids += tok(f"{name}:", add_special_tokens=False)["input_ids"]
        slot_positions.append(len(ids))
        label_ids.append([single_token(tok, l) for l in labels])
        ids.append(label_ids[-1][0])  # placeholder, left as a free slot
        ids += tok("\n", add_special_tokens=False)["input_ids"]
    if len(ids) > canvas_len:
        raise SystemExit(f"template is {len(ids)} tokens, canvas is {canvas_len}")
    pad = tok("\n", add_special_tokens=False)["input_ids"][0]
    ids += [pad] * (canvas_len - len(ids))
    pinned = [p for p in range(canvas_len) if p not in slot_positions]
    return ids, pinned, slot_positions, label_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:30000")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--kind", choices=["noul", "choice", "score"], default="noul")
    ap.add_argument("--options", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--canvas-length", type=int, default=256, help="the served canvas (DiffusionGemma: 256)")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--constrained", action="store_true", default=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    labels = {"noul": ["yes", "no"], "choice": a.options or [], "score": a.levels or []}[a.kind]
    if not labels:
        raise SystemExit("choice needs --options, score needs --levels")
    slots = [("answer", labels)]
    canvas, pinned, slot_positions, label_ids = build_read(tok, a.canvas_length, slots)
    messages = [
        {"role": "system", "content": "Answer the question about the context by filling the answer slot with one of: " + ", ".join(labels) + "."},
        {"role": "user", "content": (a.context + "\n\n" if a.context else "") + a.question},
    ]
    prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    union = sorted({t for ids in label_ids for t in ids})
    body = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": a.canvas_length,
            "custom_params": {"diffusion": {"seed_canvas": canvas, "pinned": pinned, "max_steps": a.steps,
                                            "read_only": True, "constrained": a.constrained}},
        },
        "return_logprob": True,
        "token_ids_logprob": union,
    }
    req = urllib.request.Request(a.server + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    rows = out["meta_info"]["output_token_ids_logprobs"]
    for (name, labs), pos, ids in zip(slots, slot_positions, label_ids):
        row = {int(t[1]): t[0] for t in rows[pos]}  # [logprob, token_id, text]
        lps = [row[t] for t in ids]
        z = math.log(sum(math.exp(x) for x in lps))
        probs = {l: math.exp(x - z) for l, x in zip(labs, lps)}
        best = max(probs, key=probs.get)
        print(json.dumps({"slot": name, "answer": best, "probs": {k: round(v, 4) for k, v in probs.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
