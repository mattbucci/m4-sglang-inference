import json, glob, os, sys, collections
PRE = ["coder-30b","qwen3-moe","qwen3-32b","devstral","qwen36-27b","qwen35","qwen35-9b-8bit","nemotron-30b"]
def load(name):
    d = json.load(open(f"ab-{name}.json")); m = {}
    for c in d["cold"]:
        m[f"dec@{c['target']}"] = (c["cold"]["decode_tok_s"] + c["repeat"]["decode_tok_s"]) / 2
        m[f"pre@{c['target']}"] = c["prefill_tok_s"]
    for r in d["runs"]:
        m[f"run-{r['mode']}"] = r["decode_tok_s_median"]
    p = f"ab-{name}.streams.json"
    if os.path.exists(p):
        for s in json.load(open(p))["summary"]:
            m[f"agg@{s['streams']}"] = s["aggregate_tok_s"]
    w = [float(x) for x in open(f"ab-{name}.wired").read().split()]
    m["wired"] = max(w)
    return m
out = {}
for p in PRE:
    names = sorted({os.path.basename(f)[3+len(p)+1:-7] for f in glob.glob(f"ab-{p}-*-a.json")})
    names = [n for n in names if os.path.exists(f"ab-{p}-{n}-a.json") and f"ab-{p}-{n}-a.json" in os.listdir('.')]
    # guard qwen35 vs qwen35-9b-8bit prefix clash
    names = [n for n in names if not n.startswith("9b-8bit")]
    base = {o: load(f"{p}-base-{o}") for o in "ab"}
    keys = [k for k in base["a"] if k != "wired"]
    print(f"\n== {p}  base a/b: " + "  ".join(f"{k} {base['a'][k]:.1f}/{base['b'][k]:.1f}" for k in keys) + f"  wired {base['a']['wired']:.1f}/{base['b']['wired']:.1f}")
    out[p] = {}
    for n in names:
        if n == "base": continue
        arm = {o: load(f"{p}-{n}-{o}") for o in "ab"}
        row = {k: tuple(100 * (arm[o][k] / base[o][k] - 1) for o in "ab") for k in keys}
        dw = max(arm[o]["wired"] - base[o]["wired"] for o in "ab")
        dk = [k for k in keys if not k.startswith("pre")]
        win = any(min(row[k]) >= 2 for k in dk)
        lose = [k for k in keys if min(row[k]) < -1]
        verdict = "PROMOTE" if win and not lose and dw <= 2 else ("lose:" + ",".join(lose) if lose else "neutral")
        out[p][n] = {"delta_pct": row, "wired_delta_gb": dw, "verdict": verdict}
        print(f"  {n:24s} " + "  ".join(f"{k} {row[k][0]:+.1f}/{row[k][1]:+.1f}" for k in keys) + f"  dW {dw:+.1f}  -> {verdict}")
json.dump(out, open("analysis.json", "w"), indent=1)
