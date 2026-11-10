#!/usr/bin/env python3
"""Does a request that resumes a finished request's exact tokens continue from
the state that belongs to those tokens?

A finished request donates its hybrid (DeltaNet / Mamba / windowed-attention)
state to the radix tree under the key prompt + output[:-1]. A follow-up whose
input is prompt + output hits that leaf, recomputes one token and decodes on.
If the donated state is the state after exactly those tokens, the follow-up
sees the distributions of one uninterrupted run, up to the bf16 noise between
a cache hit and a cold prefill. A state that is ahead of or behind its key
(the hazard when the decode loop keeps a lookahead step in flight at finish
time, or keys fewer tokens than it decoded) moves the very first resumed
logprob by far more than that noise.

Per generation length N (the finished request's output length), for a finish
on length and a finish on a stop token:
  whole  = generate(P, N + M)          cold, uninterrupted
  first  = generate(P, N)              cold, finishes
  resume = generate(P + first, M)      served from what the finish donated
The probe checks which prefix served the resume and compares its logprobs with
the uninterrupted run up to the first token where the greedy paths part (a
near-tie can flip on hit-versus-cold noise; after that the texts differ and
the logprobs no longer describe the same positions).

--ref <json from another arm> additionally requires the resumed tokens and
logprobs to match that arm's: a server reproduces its own resumes bit for bit,
so two servers that donate the same state resume identically. This is the
conclusive check. A state one token ahead of its key moves some resumed
logprobs by only 0.01-0.3, inside the hit-versus-cold noise the standalone
thresholds have to allow, and fails the --ref comparison in every case.

    python scripts/eval/probe_finish_state_resume.py --port 23334 --tokenizer <model dir>
"""
import argparse
import json
import random
import sys
import urllib.request

WORDS = ("system record window process value table market region station final report ancient garden "
         "silver monitor velvet crimson harbor lantern meadow copper thunder whisper granite orchid saffron "
         "timber quartz").split()


def prompt_ids(tokenizer_dir, n_tokens, seed):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")
    rng = random.Random(seed)
    text = ""
    while True:
        text += " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 14))).capitalize() + ". "
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) >= n_tokens:
            return ids[:n_tokens]


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw.decode("utf-8", errors="replace")}


def generate(base, ids, n, stop_token=None):
    # ignore_eos also disables stop_token_ids, so the stop finish leaves it off.
    params = {"max_new_tokens": n, "temperature": 0, "ignore_eos": stop_token is None}
    if stop_token is not None:
        params["stop_token_ids"] = [stop_token]
    out = post(base, "/generate", {"input_ids": ids, "sampling_params": params, "return_logprob": True})
    meta = out["meta_info"]
    pairs = [(int(t[1]), float(t[0])) for t in meta.get("output_token_logprobs") or []]
    return {"ids": [p[0] for p in pairs], "lp": [p[1] for p in pairs],
            "cached": meta.get("cached_tokens"), "prompt_tokens": meta.get("prompt_tokens"),
            "finish": (meta.get("finish_reason") or {}).get("type")}


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_case(base, prompt, n, tail, finish):
    post(base, "/flush_cache", {})
    whole = generate(base, prompt, n + tail)
    stop_token = None
    if finish == "stop":
        stop_token = whole["ids"][n - 1]
        if stop_token in whole["ids"][:n - 1]:
            return None  # the stop token would end the request early
    post(base, "/flush_cache", {})
    first = generate(base, prompt, n + 8 if finish == "stop" else n, stop_token)
    # An end-of-sequence token can end the stop-finish request sooner; it is
    # still a finish on a stop token, so the case runs at the length it reached.
    n = len(first["ids"])
    if n < 2:
        return None  # no decode step ran, so the finish donates nothing
    resume = generate(base, prompt + first["ids"], tail)

    want_tail = whole["ids"][n:n + tail]
    diverge = first_diff(resume["ids"], want_tail)
    upto = len(resume["ids"]) if diverge is None else diverge + 1
    deltas = [abs(a - b) for a, b in zip(resume["lp"][:upto], whole["lp"][n:n + upto])]
    return {
        "gen_len": n,
        "finish": finish,
        "finish_reason": first["finish"],
        "first_matches_whole": first["ids"] == whole["ids"][:n],
        "prompt_tokens": resume["prompt_tokens"],
        "cached_tokens": resume["cached"],
        "leaf_tokens": len(prompt) + n - 1,
        "leaf_hit": resume["cached"] == len(prompt) + n - 1,
        "tail_first_diff": diverge,
        "compared_tokens": upto,
        "first_token_logprob_delta": deltas[0] if deltas else None,
        "max_logprob_delta": max(deltas) if deltas else None,
        "resume_ids": resume["ids"],
        "resume_logprobs": resume["lp"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tokenizer", required=True, help="model directory holding tokenizer.json")
    ap.add_argument("--depth", type=int, default=1500, help="prompt tokens")
    ap.add_argument("--gen-lens", default="2,3,5,17,64", help="finished-request output lengths")
    ap.add_argument("--tail", type=int, default=48, help="tokens decoded after the resume")
    ap.add_argument("--finishes", default="length,stop", help="how the finished request ends")
    ap.add_argument("--max-first-delta", type=float, default=1.0,
                    help="largest hit-versus-cold logprob difference accepted on the first resumed token")
    ap.add_argument("--max-logprob-delta", type=float, default=1.0,
                    help="largest hit-versus-cold logprob difference accepted before the paths part "
                         "(the noise compounds along the tail)")
    ap.add_argument("--no-require-leaf", action="store_true",
                    help="accept a resume served from a shorter prefix (finish donates nothing)")
    ap.add_argument("--ref", help="JSON from another arm; resumed tokens and logprobs must match it")
    ap.add_argument("--ref-tol", type=float, default=1e-4)
    ap.add_argument("--out")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    ref = {}
    if args.ref:
        with open(args.ref) as f:
            ref = {(r["gen_len"], r["finish"]): r for r in json.load(f)["rows"]}

    rows, ok = [], True
    for n in (int(x) for x in args.gen_lens.split(",")):
        prompt = prompt_ids(args.tokenizer, args.depth, seed=100 + n)
        for finish in args.finishes.split(","):
            row = run_case(base, prompt, n, args.tail, finish)
            if row is None:
                print(f"gen {n:>3} {finish:<6}: skipped (no usable stop token at this length)")
                continue
            checks = {
                "first": row["first_matches_whole"],
                "leaf": row["leaf_hit"] or args.no_require_leaf,
                "state": row["max_logprob_delta"] is not None
                and row["first_token_logprob_delta"] <= args.max_first_delta
                and row["max_logprob_delta"] <= args.max_logprob_delta,
            }
            n = row["gen_len"]
            other = ref.get((n, finish))
            if other is not None and row["leaf_hit"] and other.get("leaf_hit"):
                lp_gap = max((abs(a - b) for a, b in zip(row["resume_logprobs"], other["resume_logprobs"])),
                             default=0.0)
                row["ref_ids_equal"] = row["resume_ids"] == other["resume_ids"]
                row["ref_max_logprob_delta"] = lp_gap
                checks["ref"] = row["ref_ids_equal"] and lp_gap <= args.ref_tol
            row["checks"] = checks
            row["pass"] = all(checks.values())
            ok = ok and row["pass"]
            rows.append(row)
            where = "same text" if row["tail_first_diff"] is None else f"parts at {row['tail_first_diff']}"
            line = (f"gen {n:>3} {finish:<6}: cached {row['cached_tokens']} (leaf {row['leaf_tokens']}) "
                    f"{where}, logprob delta first {row['first_token_logprob_delta']:.4f} "
                    f"max {row['max_logprob_delta']:.4f} over {row['compared_tokens']}")
            if "ref" in checks:
                line += f", vs ref ids {'equal' if row['ref_ids_equal'] else 'DIFFER'} lp {row['ref_max_logprob_delta']:.2e}"
            failed = [k for k, v in checks.items() if not v]
            print(line + (" PASS" if row["pass"] else f" FAIL({','.join(failed)})"))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"depth": args.depth, "tail": args.tail, "rows": rows}, f, indent=1)
    print("FINISH STATE RESUME:", "PASS" if ok and rows else "FAIL")
    sys.exit(0 if ok and rows else 1)


if __name__ == "__main__":
    main()
