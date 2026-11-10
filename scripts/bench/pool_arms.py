#!/usr/bin/env python3
"""Pool decode_ab.py arms: per-depth median per arm, the spread across arms,
and groups of arms pooled over their runs with the gain over the first group.

    python scripts/bench/pool_arms.py <dir> base-1,a-1,a-2,base-2 base=base-1+base-2 a=a-1+a-2
"""
import json, statistics, sys
d, arms = sys.argv[1], sys.argv[2].split(",")
groups = dict(g.split("=") for g in sys.argv[3:] if "=" in g)
rows = {a: {c["prompt_tokens"]: c for c in json.load(open(f"{d}/{a}.json"))["contexts"]} for a in arms}
ctxs = sorted(next(iter(rows.values())))
print("arm        " + "".join(f"{c:>12}" for c in ctxs))
for a in arms:
    print(f"{a:10s} " + "".join(f"{rows[a][c]['tpot_ms_median']:12.3f}" for c in ctxs))
meds = {c: [rows[a][c]["tpot_ms_median"] for a in arms] for c in ctxs}
print("all arms: " + "  ".join(f"{c}: mean {statistics.mean(v):.3f} sd {statistics.pstdev(v):.3f} ({100*statistics.pstdev(v)/statistics.mean(v):.2f}%) range {min(v):.2f}-{max(v):.2f} ({100*(max(v)-min(v))/statistics.mean(v):.2f}%)" for c, v in meds.items()))
base = None
for g, members in groups.items():
    ms = members.split("+")
    pooled = {c: statistics.mean(r["tpot_ms"] for a in ms for r in rows[a][c]["runs"]) for c in ctxs}
    if base is None:
        base = pooled
    print(f"{g:10s} " + "".join(f"{pooled[c]:9.3f} ms {100*(base[c]/pooled[c]-1):+5.1f}%" for c in ctxs) + "   tok/s " + " ".join(f"{1000/pooled[c]:.1f}" for c in ctxs))
