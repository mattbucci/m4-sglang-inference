#!/usr/bin/env python3
"""Per-chunk prefill rate of every request in a server log.

The scheduler prints `input throughput (token/s)` with each `Prefill batch`
line; the rate printed with chunk k+1 covers chunk k. A request ends at its
first chunk shorter than the chunk size. Two arms that differ by a constant
factor on every chunk differ in the forward itself; a stretch of slow chunks
that starts and ends between requests is the host, not the arm.

usage: prefill_chunk_rates.py [--chunk 1024] [--min-chunks 3] <server log>...
"""
import argparse
import re

LINE = re.compile(
    r"Prefill batch, #new-seq: \d+, #new-token: (\d+),"
    r".*input throughput \(token/s\): ([\d.]+)"
)


def requests(path: str, chunk: int) -> list:
    out, cur = [], []
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        cur.append((int(m.group(1)), float(m.group(2))))
        if cur[-1][0] < chunk:
            out.append(cur)
            cur = []
    return [[c[i + 1][1] for i in range(len(c) - 1)] for c in out]


def picks(n: int) -> list:
    want = {0, 1, 2, 3, 4, 5, 6, n // 4, n // 2, 3 * n // 4, n - 2}
    return sorted(i for i in want if 0 <= i < n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--min-chunks", type=int, default=3)
    a = ap.parse_args()
    arms = {p: [r for r in requests(p, a.chunk) if len(r) >= a.min_chunks] for p in a.logs}
    for i in range(max(len(v) for v in arms.values())):
        for path, reqs in arms.items():
            if i >= len(reqs):
                continue
            # The last rate covers the short tail chunk; it is not comparable.
            r = reqs[i]
            cells = " ".join(f"{j}:{r[j]:.0f}" for j in picks(len(r)))
            name = path.rsplit("/", 1)[-1]
            print(f"{name:>22} request {i} ({len(r)} full chunks): {cells}")


if __name__ == "__main__":
    main()
