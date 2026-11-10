#!/usr/bin/env python3
"""Summarise a serving-step census (serve_census/sitecustomize.py output).

usage: serve_census_report.py <serve.jsonl> [summary.json]
Rows are grouped at time gaps over 4 s (one group per benchmark context) and
printed as ms per decode step, with the nested phases subtracted out.
"""
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
# split into depth groups at time gaps > 4 s (cold prefills of the deeper contexts)
groups, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if b["t"] - a["t"] > 4.0:
        groups.append(cur); cur = []
    cur.append(b)
groups.append(cur)
keys = ["wall","ingest","launch_chained","graph_build","fwd","select","async_eval","finalize_job","finalize_result","token_wait","process_result","result_decode","stream_output"]
def ms(g, k):
    return sum(r.get(k, 0) for r in g) / sum(r["n"] for r in g) / 1e6
out = []
for g in groups:
    m = {k: ms(g, k) for k in keys}
    m["steps"] = sum(r["n"] for r in g)
    m["launch_other"] = m["launch_chained"] - m["graph_build"] - m["async_eval"]
    m["build_other"] = m["graph_build"] - m["fwd"] - m["select"]
    m["host_phase"] = m["wall"] - m["async_eval"]
    m["finalize_host"] = m["finalize_result"] - m["token_wait"]
    m["process_other"] = m["process_result"] - m["result_decode"]
    m["decode_other"] = m["result_decode"] - m["stream_output"]
    m["job_other"] = m["finalize_job"] - m["finalize_result"] - m["process_result"]
    m["loop_other"] = m["wall"] - m["ingest"] - m["launch_chained"] - m["finalize_job"]
    m["host_total"] = m["wall"] - m["token_wait"]
    out.append(m)
cols = ["steps","wall","host_phase","graph_build","fwd","select","build_other","async_eval","launch_other","token_wait","finalize_host","stream_output","decode_other","process_other","job_other","ingest","loop_other","host_total"]
print("| group | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
for i, m in enumerate(out):
    print(f"| {i} | " + " | ".join(f"{m[c]:.3f}" if c != "steps" else str(m[c]) for c in cols) + " |")
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
