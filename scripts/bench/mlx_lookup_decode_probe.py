#!/usr/bin/env python3
"""Raw MLX: greedy decode with prompt-lookup drafts against the ordinary decode loop, on real agent turns.

`prompt_lookup_acceptance.py` replays archived turns against a cost table. This probe runs
the decoder itself: it rebuilds the context of chosen turns of an archived omp session,
prefills it, and decodes the turn greedily several ways from the same cache state:

  plain   the pipelined one-row loop every preset serves today
  sync    the one-row loop waiting for each token, which is what a drafting decoder pays
          whenever it has no draft
  <policy>  draft-and-verify: the last n-gram of the context is looked up earlier in the
          context, the tokens that followed it ride as extra rows of one forward, and the
          leading rows whose input matched the model's own arg-max are kept

A rejected row is undone by trimming the attention caches. A checkpoint with recurrent
layers (DeltaNet) has no trim: the recurrent state is restored from before the forward and
the kept tokens ride again in the next forward. Arms run in the order given and then
reversed, from the same cache mark. The text of a drafting arm can differ from `plain`
where two tokens are near-tied: MLX rounds a one-row and a several-row forward differently.

    SGLANG_USE_MLX=1 python scripts/bench/mlx_lookup_decode_probe.py --model <dir-or-repo> \
        --session evals/swebench/runs/<cell>/sessions/<instance>/<file>.jsonl@8,16,24 \
        --policies adapt4,fixed2 --out lookup.json
    python scripts/bench/mlx_lookup_decode_probe.py --self-test     # decoder logic, no MLX

`--fidelity` adds untimed passes per turn. Each drafting arm runs once more keeping the
log-probabilities of every row it used. The same tokens are then fed one row at a time
(the ordinary decode path) and once more as a single forward (the prefill path). The
report is KL(one-row || drafting) per token next to KL(one-row || single forward), the
distance the server's own prefill already keeps from its decode steps, and, where an
arm's text leaves `plain`, the gap between plain's two best tokens at that step.

No server may be running: the probe needs the GPU to itself.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_lookup_acceptance as pla  # noqa: E402

POLICIES = {
    "fixed2": {"kind": "fixed", "kmax": 2, "min_back": 2},
    "fixed2-m4": {"kind": "fixed", "kmax": 2, "min_back": 4},
    "fixed4": {"kind": "fixed", "kmax": 4, "min_back": 2},
    "adapt4": {"kind": "adaptive", "kmax": 4, "min_back": 2},
    "adapt4-m4": {"kind": "adaptive", "kmax": 4, "min_back": 4},
    "adapt8": {"kind": "adaptive", "kmax": 8, "min_back": 2},
    "len-m4": {"kind": "backlen", "kmax": 16, "min_back": 4, "ladder": [(4, 3), (8, 4), (16, 8), (32, 16)]},
    "len-m8": {"kind": "backlen", "kmax": 16, "min_back": 8, "ladder": [(8, 3), (12, 4), (24, 8), (48, 16)]},
}
NGRAMS = (2, 3, 4, 8)


def decode_sync(st, first, max_new, eos):
    out, y = [], first
    t0 = time.perf_counter()
    while len(out) < max_new:
        y = st.forward([y])[-1]
        out.append(y)
        if y in eos:
            break
    return out, time.perf_counter() - t0, {"plain": len(out)}


def decode_draft(st, seq, policy, max_new, eos, carry_limit=6):
    """`seq` is the context (its last token is the first input) and grows by the output.
    The stepper has consumed seq[:-1]. Returns the new tokens, seconds, counters."""
    lk = pla.Lookup(seq, NGRAMS)
    lk.find(len(seq))  # index the context before the clock starts
    stats = collections.Counter()
    out, pending, level, done = [], [seq[-1]], 0, False
    t0 = time.perf_counter()
    while not done:
        i, draft = len(seq), []
        if len(pending) <= carry_limit:
            p, back = lk.find(i)
            k = pla.rows_for(policy, back, level, (i - p) if p is not None else 0)
            if k > 1:
                draft = seq[p:p + k - 1]
        mark = st.mark() if draft and not st.pure_kv else None
        toks = st.forward(pending + draft)
        base = len(pending) - 1
        m = 0
        while m < len(draft) and toks[base + m] == draft[m]:
            m += 1
        new = draft[:m] + [toks[base + m]]
        if getattr(st, "keep", None) is not None:
            st.keep_rows(base, base + m + 1)
        stats["rows"] += len(pending) + len(draft)
        if not draft:
            stats["plain"] += 1
            pending = [new[-1]]
        else:
            stats["forwards"] += 1
            stats["drafted"] += len(draft)
            stats["accepted"] += m
            if m == len(draft):
                level += 1
                pending = [new[-1]]
            else:
                level = 0
                stats["rejects"] += 1
                if st.pure_kv:
                    st.trim(len(draft) - m)
                    pending = [new[-1]]
                else:
                    st.reset(mark)
                    pending = pending + new
                    stats["replayed"] += len(pending) - 1
        for t in new:
            out.append(t)
            seq.append(t)
            if t in eos or len(out) >= max_new:
                done = True
                break
    stats["pending_at_end"] = len(pending)
    return out, time.perf_counter() - t0, dict(stats)


class MlxStepper:
    """The loaded engine behind the three calls the decoders need."""

    def __init__(self, eng):
        import mlx.core as mx

        self.mx, self.eng = mx, eng
        self.kv = [c for c in eng.cache if hasattr(c, "trim") and hasattr(c, "offset")]
        self.rec = [c for c in eng.cache if not (hasattr(c, "trim") and hasattr(c, "offset"))]
        for c in self.rec:
            if not isinstance(getattr(c, "cache", None), list):
                raise SystemExit(f"no rollback for cache type {type(c).__name__}")
        self.pure_kv = not self.rec
        self.keep, self._last = None, None

    def forward(self, inp):
        mx = self.mx
        logits = self.eng.logits(mx.array([inp], dtype=mx.int32))[0]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        self._last = logits if self.keep is not None else None
        return tok.tolist()

    def log_probs(self, logits):
        mx = self.mx
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        return lp

    def keep_rows(self, lo, hi):
        lp = self.log_probs(self._last[lo:hi])
        self.keep.extend(lp[j] for j in range(hi - lo))

    def mark(self):
        return [int(c.offset) for c in self.kv], [list(c.cache) for c in self.rec]

    def reset(self, h):
        for c, off in zip(self.kv, h[0]):
            c.trim(int(c.offset) - off)
        for c, saved in zip(self.rec, h[1]):
            c.cache = list(saved)

    def trim(self, n):
        for c in self.kv:
            c.trim(n)


def decode_plain(eng, first, max_new, eos):
    import mlx.core as mx

    t0 = time.perf_counter()
    nxt = eng.step_full(mx.array([first], dtype=mx.int32))
    mx.async_eval(nxt)
    out = []
    while True:
        ahead = eng.step_full(nxt)
        mx.async_eval(ahead)
        tok = nxt.item()
        out.append(tok)
        if tok in eos or len(out) >= max_new:
            break
        nxt = ahead
    dt = time.perf_counter() - t0
    mx.eval(ahead)  # the step already in flight; it is not a token of this run
    return out, dt, {"plain": len(out)}


def feed(eng, ids, chunk):
    import mlx.core as mx

    for i in range(0, len(ids), chunk):
        part = mx.array(ids[i:i + chunk], dtype=mx.int32)[None]
        if eng.vlm:
            eng.lm(part, cache=eng.cache, skip_logits=True)
        elif eng.trunk_module is not None:
            eng.trunk_module(part, cache=eng.cache)
        else:
            eng.lm(part, cache=eng.cache)
        mx.eval(eng.cache_arrays())
        mx.clear_cache()


def kl_rows(mx, ref, other):
    """KL(ref || other) per position, and whether the two rows pick the same token."""
    kls, same = [], 0
    for a, b in zip(ref, other):
        kls.append(float(mx.sum(mx.exp(a) * (a - b)).item()))
        same += int(mx.argmax(a).item() == mx.argmax(b).item())
    return kls, same


def one_row_rows(st, first, ys):
    """Feed `first` then ys[:-1] one row per forward; the log-probability row of every step."""
    mx, rows, x = st.mx, [], first
    for y in ys:
        rows.append(st.log_probs(st.eng.logits(mx.array([[x]], dtype=mx.int32))[0][-1]))
        x = y
    return rows


def fidelity_pass(st, ctx, arm, out, plain_out, plain_rows, turn_mark, max_new, eos):
    mx = st.mx
    st.keep = []
    again, _, _ = decode_draft(st, list(ctx), POLICIES[arm], max_new, eos)
    draft_rows, st.keep = st.keep, None
    st.reset(turn_mark)
    n = len(out)
    one = plain_rows if out == plain_out else one_row_rows(st, ctx[-1], out)
    st.reset(turn_mark)
    whole = st.log_probs(st.eng.logits(mx.array([[ctx[-1]] + out[:-1]], dtype=mx.int32))[0])
    st.reset(turn_mark)
    kd, same_d = kl_rows(mx, one, draft_rows[:n])
    kw, same_w = kl_rows(mx, one, [whole[j] for j in range(n)])
    r = {"tokens": n, "rerun_equal": again == out, "rows_kept": len(draft_rows),
         "kl_draft_mean": sum(kd) / n, "kl_draft_max": max(kd), "argmax_same_draft": same_d,
         "kl_forward_mean": sum(kw) / n, "kl_forward_max": max(kw), "argmax_same_forward": same_w}
    d = first_diff(out, plain_out)
    if d is not None and d < len(plain_rows) and d < n:
        row = plain_rows[d]
        order = mx.argsort(row)
        best, second = int(order[-1].item()), int(order[-2].item())
        r.update({"diverges_at": d, "plain_gap": float((row[best] - row[second]).item()),
                  "arm_token_is_plain_second": out[d] == second,
                  "arm_token_gap": float((row[best] - row[out[d]]).item())})
    return r


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


class FakeStepper:
    """A toy model for the self-test: on the target path it emits the target's next token,
    off it a token derived from the history, so a rejected row leaves a wrong history."""

    def __init__(self, target, consumed, pure_kv):
        self.target, self.hist, self.pure_kv = target, list(target[:consumed]), pure_kv

    def _next(self):
        n = len(self.hist)
        if self.hist == self.target[:n] and n < len(self.target):
            return self.target[n]
        return 1000 + (sum(self.hist[-3:]) % 7)

    def forward(self, inp):
        out = []
        for t in inp:
            self.hist.append(t)
            out.append(self._next())
        return out

    def mark(self):
        return len(self.hist)

    def reset(self, h):
        del self.hist[h:]

    def trim(self, n):
        assert self.pure_kv, "trim on a checkpoint with recurrent state"
        del self.hist[len(self.hist) - n:]


def self_test():
    rng = random.Random(5)
    runs = fails = 0
    seen = collections.Counter()
    for case in range(40):
        ctx = [rng.randrange(40) for _ in range(300)]
        tail = []
        while len(tail) < 260:
            if rng.random() < 0.6:  # a stretch copied from earlier text, often cut short
                src = ctx + tail
                a = rng.randrange(len(src) - 2)
                tail.extend(src[a:a + rng.randrange(1, 24)])
            else:
                tail.extend(rng.randrange(40) for _ in range(rng.randrange(1, 6)))
        eos = {999}
        target = ctx + tail[:260] + [999] + [5] * 40
        want = target[len(ctx):len(ctx) + 261]
        for name, pol in POLICIES.items():
            for pure in (True, False):
                st = FakeStepper(target, len(ctx) - 1, pure)
                seq = list(ctx)
                out, _, stats = decode_draft(st, seq, pol, 400, eos)
                runs += 1
                seen.update({k: v for k, v in stats.items() if k != "pending_at_end"})
                # the history plus what is still pending must be the text so far
                ok = out == want and st.hist == seq[:len(seq) - stats["pending_at_end"]]
                if not ok:
                    fails += 1
                    print(f"FAIL case {case} {name} pure_kv={pure}: first diff {first_diff(out, want)} "
                          f"hist {len(st.hist)} seq {len(seq)} stats {stats}")
        st = FakeStepper(target, len(ctx) - 1, True)
        out, _, _ = decode_sync(st, ctx[-1], 400, eos)
        runs += 1
        if out != want:
            fails += 1
            print(f"FAIL case {case} sync")
    # a cap that lands inside an accepted draft still returns exactly max_new tokens
    st = FakeStepper(target, len(ctx) - 1, True)
    out, _, _ = decode_draft(st, list(ctx), POLICIES["adapt8"], 37, eos)
    runs += 1
    if out != want[:37]:
        fails += 1
        print("FAIL cap inside a draft")
    print(f"self-test: {runs - fails}/{runs} decodes equal the target text; exercised {dict(seen)}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--session", nargs="+", help="omp session files (.jsonl), each optionally <file>@<turns>")
    ap.add_argument("--turns", default="8,16,24", help="assistant turn indexes to decode in every session")
    ap.add_argument("--policies", default="adapt4,fixed2")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=24000)
    ap.add_argument("--gen-prefix", default="<think>\n\n</think>\n\n",
                    help="text after the assistant header, the thinking-off generation prompt")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--fidelity", action="store_true",
                    help="untimed passes: KL of the drafting path and of one forward against one-row steps")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    import mlx.core as mx
    import mlx_decode_census as census
    from tokenizers import Tokenizer

    mx.set_cache_limit(int(a.cache_limit_gb * 2**30))
    model_dir = census.resolve_model_dir(a.model)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids if t else []
    eos = {i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) if i is not None}
    prefix = enc(a.gen_prefix)
    eng = census.Engine(model_dir)
    arms = ["plain", "sync"] + [p for p in a.policies.split(",") if p]
    print(f"loaded {a.model} vlm={eng.vlm} layers={len(eng.layers)} active={mx.get_active_memory() / 2**30:.2f} GB "
          f"arms={arms}", flush=True)
    result = {"model": a.model, "mlx": mx.__version__, "max_new": a.max_new, "turns": []}
    pooled = {arm: [0, 0.0] for arm in arms}
    fid = {arm: [] for arm in arms[2:]}
    for spec in a.session:
        path, _, own = spec.partition("@")
        seq, spans, _ = pla.load_session(path, enc)
        eng.new_cache()
        consumed = 0
        for t in sorted(int(x) for x in (own or a.turns).split(",")):
            if t >= len(spans) or spans[t][0] + len(prefix) > a.max_context:
                print(f"skip turn {t} of {os.path.basename(os.path.dirname(path))}", flush=True)
                continue
            start = spans[t][0]
            t0 = time.perf_counter()
            feed(eng, seq[consumed:start], a.chunk)
            consumed = start
            st = MlxStepper(eng)
            base_mark = st.mark()
            ctx = seq[:start] + prefix
            feed(eng, prefix[:-1], a.chunk)
            turn_mark = st.mark()
            row = {"session": os.path.basename(os.path.dirname(path)), "turn": t, "context": len(ctx),
                   "archived_turn_tokens": spans[t][1] - spans[t][0], "pure_kv": st.pure_kv, "arms": {}}
            print(f"\n== {row['session']} turn {t}: context {len(ctx)} tokens, prefill {time.perf_counter() - t0:.1f}s, "
                  f"pure_kv={st.pure_kv}", flush=True)
            runs = {arm: [] for arm in arms}
            for arm in arms + arms[::-1]:
                if arm == "plain":
                    out, dt, stats = decode_plain(eng, ctx[-1], a.max_new, eos)
                elif arm == "sync":
                    out, dt, stats = decode_sync(st, ctx[-1], a.max_new, eos)
                else:
                    out, dt, stats = decode_draft(st, list(ctx), POLICIES[arm], a.max_new, eos)
                runs[arm].append((out, dt, stats))
                st.reset(turn_mark)
            ref = runs["plain"][0][0]
            for arm in arms:
                (o1, d1, s1), (o2, d2, _) = runs[arm]
                n, dt = len(o1) + len(o2), d1 + d2
                pooled[arm][0] += n
                pooled[arm][1] += dt
                r = {"tokens": len(o1), "ms_per_token": round(1e3 * dt / n, 3), "tok_s": round(n / dt, 2),
                     "repeat_equal": o1 == o2, "first_diff_vs_plain": first_diff(o1, ref), "stats": s1,
                     "ids": o1}
                row["arms"][arm] = r
            base = row["arms"]["plain"]["tok_s"]
            for arm in arms:
                r = row["arms"][arm]
                s = r["stats"]
                acc = f"{s.get('accepted', 0)}/{s.get('drafted', 0)}"
                print(f"  {arm:10s} {r['tokens']:4d} tok {r['ms_per_token']:8.2f} ms/tok {r['tok_s']:7.2f} tok/s "
                      f"x{r['tok_s'] / base:5.3f}  drafts {s.get('forwards', 0):3d} accepted {acc:>9s} "
                      f"replayed {s.get('replayed', 0):3d}  vs plain: "
                      f"{'same' if r['first_diff_vs_plain'] is None else 'differs at ' + str(r['first_diff_vs_plain'])}"
                      f"{'' if r['repeat_equal'] else '  REPEAT DIFFERS'}", flush=True)
            print("  text: " + json.dumps(tok.decode(ref)[:300]), flush=True)
            if a.fidelity:
                plain_rows = one_row_rows(st, ctx[-1], ref)
                st.reset(turn_mark)
                row["fidelity"] = {}
                for arm in arms[2:]:
                    f = fidelity_pass(st, ctx, arm, row["arms"][arm]["ids"], ref, plain_rows, turn_mark,
                                      a.max_new, eos)
                    row["fidelity"][arm] = f
                    fid[arm].append(f)
                    gap = ("" if "diverges_at" not in f else
                           f"  leaves plain at {f['diverges_at']}: plain's two best are {f['plain_gap']:.3f} apart, "
                           f"arm token {f['arm_token_gap']:.3f} below the best")
                    print(f"  fidelity {arm:10s} KL(one-row || drafting) mean {f['kl_draft_mean']:.5f} max "
                          f"{f['kl_draft_max']:.4f} same token {f['argmax_same_draft']}/{f['tokens']} | "
                          f"KL(one-row || one forward) mean {f['kl_forward_mean']:.5f} max {f['kl_forward_max']:.4f} "
                          f"same token {f['argmax_same_forward']}/{f['tokens']}{gap}"
                          f"{'' if f['rerun_equal'] else '  RERUN DIFFERS'}", flush=True)
                del plain_rows
                mx.clear_cache()
            result["turns"].append(row)
            st.reset(base_mark)
    print("\n== pooled over turns", flush=True)
    base = pooled["plain"][0] / pooled["plain"][1]
    result["pooled"] = {}
    for arm in arms:
        n, dt = pooled[arm]
        result["pooled"][arm] = {"tokens": n, "tok_s": round(n / dt, 2), "against_plain": round(n / dt / base, 3)}
        print(f"  {arm:10s} {n:6d} tok {n / dt:7.2f} tok/s x{n / dt / base:5.3f}", flush=True)
    if a.fidelity:
        print("\n== fidelity pooled over turns (token-weighted mean, worst token)", flush=True)
        result["fidelity_pooled"] = {}
        for arm, rows in fid.items():
            n = sum(r["tokens"] for r in rows)
            if not n:
                continue
            agg = {"tokens": n,
                   "kl_draft_mean": sum(r["kl_draft_mean"] * r["tokens"] for r in rows) / n,
                   "kl_draft_max": max(r["kl_draft_max"] for r in rows),
                   "argmax_same_draft": sum(r["argmax_same_draft"] for r in rows),
                   "kl_forward_mean": sum(r["kl_forward_mean"] * r["tokens"] for r in rows) / n,
                   "kl_forward_max": max(r["kl_forward_max"] for r in rows),
                   "argmax_same_forward": sum(r["argmax_same_forward"] for r in rows)}
            result["fidelity_pooled"][arm] = agg
            print(f"  {arm:10s} {n:5d} tok  drafting {agg['kl_draft_mean']:.5f} / {agg['kl_draft_max']:.4f} "
                  f"same token {agg['argmax_same_draft']}/{n} | one forward {agg['kl_forward_mean']:.5f} / "
                  f"{agg['kl_forward_max']:.4f} same token {agg['argmax_same_forward']}/{n}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
