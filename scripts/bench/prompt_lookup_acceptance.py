#!/usr/bin/env python3
"""Would a prompt-lookup draft pay on agentic output? An offline replay, no server and no GPU.

A draft-and-verify decoder guesses the next tokens by finding the last n-gram of the
context earlier in the context and copying what followed it, then checks the guess in one
forward of several rows. `mlx_verify_rows_probe.py` gives what such a forward costs; this
script gives the other half, how often the guess is right, on real coding-agent turns: it
replays archived omp sessions (`evals/swebench/runs/<cell>/sessions`), re-renders every
assistant turn as the model emitted it (text, then each tool call in the Qwen3.6 template's
XML parameter form, then the end-of-turn token), checks the token count of each turn against
the count the server reported, tokenizes with the checkpoint's tokenizer, and walks each turn under a drafting
policy with a cost table in units of one ordinary decode step.

A forward with k rows carries the last accepted token and k-1 draft tokens. It yields the
leading draft tokens that match plus one token of the model's own, at cost c(k). With no
match the decoder takes an ordinary step: at cost 1 if those steps stay pipelined (`piped`),
at the cost of a one-row forward that waits for its token if they do not (`waits`, the
simple decoder `mlx_lookup_decode_probe.py` runs). Two rollback models bound a hybrid
checkpoint: `free` (a KV trim, or recurrent state kept per row) and `carry` (the state is
restored from before the forward and the accepted rows ride in the next forward).

    python scripts/bench/prompt_lookup_acceptance.py --tokenizer <snapshot dir> \
        --runs evals/swebench/runs/qwen36-26inst-omp-2026-11-08 --out <json>
"""
import argparse
import collections
import glob
import json
import os

# ms per forward of k tokens on a cached 8,192-token prefix, and the pipelined one-token
# step, from mlx_verify_rows_probe.py (benchmarks/perf/2026-11-10/concurrent/VERDICT.md).
MEASURED = {
    "qwen36": {"step": 12.06, "rows": {1: 14.95, 2: 18.54, 4: 26.46, 8: 50.33, 16: 74.23}},
    "qwen36+fused": {"step": 11.12, "rows": {1: 13.90, 2: 17.87, 4: 25.52, 8: 51.09, 16: 75.01}},
    "qwen36-27b": {"step": 70.25, "rows": {1: 73.95, 2: 77.04, 4: 108.03, 8: 221.15, 16: 325.24}},
}


def cost_fn(table):
    ks = sorted(table["rows"])
    def c(k):
        if k in table["rows"]:
            return table["rows"][k] / table["step"]
        lo = max(x for x in ks if x < k) if k > ks[0] else ks[0]
        hi = min((x for x in ks if x > k), default=ks[-1])
        if hi == lo:  # past the last measured point: extend the last slope
            a, b = ks[-2], ks[-1]
            return (table["rows"][b] + (k - b) * (table["rows"][b] - table["rows"][a]) / (b - a)) / table["step"]
        f = (k - lo) / (hi - lo)
        return (table["rows"][lo] * (1 - f) + table["rows"][hi] * f) / table["step"]
    return c


def render_assistant(m):
    """An assistant turn as the Qwen3.6 chat template writes it (and as the model emitted it): the text,
    then every tool call in the XML parameter form with raw string values, then the end-of-turn token."""
    out, prev = "", None
    for c in m.get("content") or []:
        if c.get("type") == "text":
            out += c.get("text", "")
            prev = "text"
        elif c.get("type") == "toolCall":
            out += ("\n\n" if prev == "text" and out.strip() else "\n" if prev == "call" else "")
            out += "<tool_call>\n<function=" + str(c.get("name")) + ">\n"
            for k, v in (c.get("arguments") or {}).items():
                out += "<parameter=" + k + ">\n" + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) + "\n</parameter>\n"
            out += "</function>\n</tool_call>"
            prev = "call"
    return out + "<|im_end|>"


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def load_session(path, enc):
    """-> token list of the whole conversation, the (start, end) span of every assistant turn, and per
    turn the output-token count the server reported (None when the row has no usage)."""
    seq, spans, served = [], [], []
    prev = None
    for line in open(path):
        if not line.strip():
            continue
        m = json.loads(line).get("message") or {}
        role = m.get("role")
        if role == "assistant":
            seq.extend(enc(("<|im_end|>\n" if prev in ("user", "toolResult") else "\n") + "<|im_start|>assistant\n"))
            ids = enc(render_assistant(m))
            spans.append((len(seq), len(seq) + len(ids)))
            served.append((m.get("usage") or {}).get("output"))
            seq.extend(ids)
        elif role == "user":
            seq.extend(enc("\n<|im_start|>user\n" + text_of(m.get("content"))))
        elif role == "toolResult":
            head = "" if prev == "toolResult" else "\n<|im_start|>user"
            seq.extend(enc(head + "\n<tool_response>\n" + text_of(m.get("content")) + "\n</tool_response>"))
        else:
            continue
        prev = role
    return seq, spans, served


class Lookup:
    """Most recent earlier occurrence of the context's last n-gram, longest n first."""
    def __init__(self, seq, ns):
        self.seq, self.ns = seq, sorted(ns, reverse=True)
        self.idx = {n: {} for n in ns}
        self.reg = 0  # n-grams ending before this position are registered

    def find(self, e):
        seq = self.seq
        while self.reg < e - 1:  # register every n-gram that ends at or before e-2
            p = self.reg
            for n in self.ns:
                if p + 1 >= n:
                    self.idx[n][tuple(seq[p + 1 - n:p + 1])] = p + 1
            self.reg += 1
        for n in self.ns:
            if e >= n:
                p = self.idx[n].get(tuple(seq[e - n:e]))
                if p is not None:
                    back = n  # how far the match really extends, a confidence signal
                    while back < 64 and p - back - 1 >= 0 and seq[p - back - 1] == seq[e - back - 1]:
                        back += 1
                    return p, back
        return None, 0


def rows_for(policy, back, level, avail):
    """Rows to send (1 = ordinary step). `avail` is how many draft tokens the match can supply."""
    kind, kmax = policy["kind"], policy["kmax"]
    if avail <= 0 or back < policy["min_back"]:
        return 1
    if kind == "fixed":
        k = kmax
    elif kind == "adaptive":  # double the draft after a fully accepted one, restart after a miss
        k = min(kmax, 2 ** (level + 1))
    elif kind == "backlen":  # a longer backward match earns a longer draft
        k = 2
        for need, rows in policy["ladder"]:
            if back >= need:
                k = rows
        k = min(k, kmax)
    return max(2, min(k, avail + 1))


def simulate(sessions, ns, policy, cost, rollback):
    tot = collections.Counter()
    for seq, spans, _ in sessions:
        lk = Lookup(seq, ns)
        for s, e_turn in spans:
            i, level, carry = s, 0, 0
            while i < e_turn:
                p, back = lk.find(i)
                avail = min(i - p, e_turn + 64) if p is not None else 0
                k = rows_for(policy, back, level, avail)
                if k == 1:
                    tot["cost"] += cost(1 + carry) if carry else 1.0
                    tot["cost_sync"] += cost(1 + carry)
                    tot["plain_steps"] += 1
                    i += 1; carry = 0
                    continue
                d = k - 1
                draft = seq[p:p + d]
                m = 0
                while m < d and i + m < e_turn and draft[m] == seq[i + m]:
                    m += 1
                tot["cost"] += cost(k + carry)
                tot["cost_sync"] += cost(k + carry)
                tot["forwards"] += 1; tot["drafted"] += d; tot["accepted"] += m
                tot["first_ok"] += 1 if m >= 1 else 0
                full = m == d
                carry = 0 if (full or rollback == "free") else m + 1
                level = level + 1 if full else 0
                i += min(m + 1, e_turn - i)
            tot["tokens"] += e_turn - s
    return tot


def oracle(sessions, ns, cost, ks):
    """Upper bound for this lookup: at each position pick the row count with the best tokens per cost,
    knowing how many draft tokens will match."""
    tot = collections.Counter()
    for seq, spans, _ in sessions:
        lk = Lookup(seq, ns)
        for s, e_turn in spans:
            i = s
            while i < e_turn:
                p, back = lk.find(i)
                m = 0
                if p is not None:
                    while m < 63 and p + m < i and i + m < e_turn and seq[p + m] == seq[i + m]:
                        m += 1
                best, bk = 1.0, 1
                for k in ks:
                    got = min(m, k - 1) + 1
                    if k > 1 and got / cost(k) > best:
                        best, bk = got / cost(k), k
                got = min(m, bk - 1) + 1 if bk > 1 else 1
                tot["cost"] += cost(bk) if bk > 1 else 1.0
                tot["accepted"] += got - 1
                i += min(got, e_turn - i)
            tot["tokens"] += e_turn - s
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="snapshot dir holding tokenizer.json")
    ap.add_argument("--runs", nargs="+", required=True, help="cell dirs with sessions/<instance>/*.jsonl")
    ap.add_argument("--ngrams", default="2,3,4,8")
    ap.add_argument("--cost", nargs="*", default=[], metavar="NAME=ROWS.json[@DEPTH]",
                    help="cost tables from mlx_verify_rows_probe.py receipts, in place of the built-in ones")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids if t else []
    ns = [int(x) for x in a.ngrams.split(",")]
    sessions = []
    for run in a.runs:
        for f in sorted(glob.glob(os.path.join(run, "sessions", "*", "*.jsonl"))):
            sessions.append(load_session(f, enc))
    n_out = sum(e - s for _, spans, _ in sessions for s, e in spans)
    n_all = sum(len(seq) for seq, _, _ in sessions)
    print(f"{len(sessions)} sessions, {sum(len(sp) for _, sp, _ in sessions)} assistant turns, "
          f"{n_out} output tokens of {n_all} conversation tokens, n-grams {ns}")

    # Does the re-rendered turn have the token count the server reported for it? That is the check that
    # the replay walks the tokens the model really emitted.
    diffs = [(e - s) - u for _, spans, served in sessions for (s, e), u in zip(spans, served) if u is not None]
    exact = sum(1 for d in diffs if d == 0)
    near = sum(1 for d in diffs if abs(d) <= 2)
    print(f"re-rendered turns vs served output-token counts: {exact}/{len(diffs)} equal, {near}/{len(diffs)} within 2 tokens, "
          f"sum {sum((e - s) for _, spans, _ in sessions for s, e in spans)} vs {sum(u for _, _, sv in sessions for u in sv if u is not None)} served")

    # How long do the lookup's guesses hold? One row per position where a match exists.
    runs = collections.Counter(); pos = 0; hit = 0
    for seq, spans, _ in sessions:
        lk = Lookup(seq, ns)
        for s, e in spans:
            for i in range(s, e):
                p, back = lk.find(i); pos += 1
                if p is None:
                    continue
                hit += 1; m = 0
                while m < 32 and p + m < i and i + m < e and seq[p + m] == seq[i + m]:
                    m += 1
                runs[m] += 1
    right = hit - runs[0]
    print(f"positions with a match {hit}/{pos} = {hit / pos:.1%}; first guess right {right}/{hit} = {right / hit:.1%}; "
          f"a right first guess at {right / pos:.1%} of all positions")
    cum = {t: sum(v for m, v in runs.items() if m >= t) / pos for t in (1, 2, 3, 4, 8, 16, 32)}
    print("share of positions whose guess holds for at least n tokens: " + ", ".join(f"{t}: {v:.1%}" for t, v in cum.items()))

    policies = [
        {"name": "fixed 2 rows", "kind": "fixed", "kmax": 2, "min_back": 2},
        {"name": "fixed 2 rows, match >= 4", "kind": "fixed", "kmax": 2, "min_back": 4},
        {"name": "fixed 4 rows", "kind": "fixed", "kmax": 4, "min_back": 2},
        {"name": "fixed 4 rows, match >= 4", "kind": "fixed", "kmax": 4, "min_back": 4},
        {"name": "adaptive to 4", "kind": "adaptive", "kmax": 4, "min_back": 2},
        {"name": "adaptive to 16", "kind": "adaptive", "kmax": 16, "min_back": 2},
        {"name": "adaptive to 16, match >= 4", "kind": "adaptive", "kmax": 16, "min_back": 4},
        {"name": "adaptive to 16, match >= 8", "kind": "adaptive", "kmax": 16, "min_back": 8},
        {"name": "by match length 4/8/16", "kind": "backlen", "kmax": 16, "min_back": 2,
         "ladder": [(4, 4), (12, 8), (32, 16)]},
        {"name": "by match length, match >= 4", "kind": "backlen", "kmax": 16, "min_back": 4,
         "ladder": [(4, 3), (8, 4), (16, 8), (32, 16)]},
        {"name": "by match length, match >= 8", "kind": "backlen", "kmax": 16, "min_back": 8,
         "ladder": [(8, 3), (12, 4), (24, 8), (48, 16)]},
    ]
    out = {"sessions": len(sessions), "output_tokens": n_out, "turns_equal_to_served": exact, "turns_checked": len(diffs), "ngrams": ns, "match_positions": hit,
           "positions": pos, "first_guess_right": right, "holds_at_least": cum, "tables": {}}
    tables = dict(MEASURED)
    if a.cost:
        tables = {}
        for spec in a.cost:
            name, _, src = spec.partition("=")
            path, _, depth = src.partition("@")
            rows = json.load(open(path))["depths"]
            row = next(r for r in rows if not depth or r["depth"] == int(depth))
            tables[name] = {"step": row["one_row_pipelined_ms"],
                            "rows": {int(k): v["median_ms"] for k, v in row["rows"].items()}}
    for name, table in tables.items():
        c = cost_fn(table)
        print(f"\n{name}: cost in steps at rows 1/2/3/4/8/16 = " + " / ".join(f"{c(k):.2f}" for k in (1, 2, 3, 4, 8, 16)))
        print(f"  {'policy':34s} {'rollback':8s} {'piped':>7s} {'waits':>7s} {'forwards':>8s} {'accepted/drafted':>17s} {'first ok':>8s} {'tokens from drafts':>18s}")
        rows = []
        for pol in policies:
            for rb in ("free", "carry"):
                t = simulate(sessions, ns, pol, c, rb)
                r = {"policy": pol["name"], "rollback": rb, "speedup": t["tokens"] / t["cost"],
                     "speedup_every_forward_waits": t["tokens"] / t["cost_sync"], "forwards": t["forwards"],
                     "plain_steps": t["plain_steps"], "drafted": t["drafted"], "accepted": t["accepted"],
                     "first_ok": t["first_ok"]}
                rows.append(r)
                acc = f"{t['accepted']}/{t['drafted']}"
                print(f"  {pol['name']:34s} {rb:8s} {r['speedup']:7.3f} {r['speedup_every_forward_waits']:7.3f} {t['forwards']:8d} {acc:>17s} "
                      f"{t['first_ok'] / max(1, t['forwards']):8.1%} {t['accepted'] / t['tokens']:18.1%}")
        o = oracle(sessions, ns, c, (2, 3, 4, 8, 16))
        print(f"  {'oracle (knows the match length)':34s} {'free':8s} {o['tokens'] / o['cost']:7.3f}")
        out["tables"][name] = {"cost": {k: c(k) for k in (1, 2, 3, 4, 8, 16)}, "rows": rows,
                               "oracle_speedup": o["tokens"] / o["cost"]}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
