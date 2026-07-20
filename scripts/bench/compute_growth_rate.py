#!/usr/bin/env python3
"""Compute per-token prefill memory growth from mem_profile.sh CSV.

Usage: compute_growth_rate.py <mem_profile.csv> <prompt_tokens>

Reads the free+inactive series, reports the drop from the pre-request
plateau to the prefill-end trough in MB/token. The window is found
automatically: the largest monotonic-ish decline in free+inactive
(prefill allocates steadily; decode and idle are flat by comparison).
Prints the full series summary so the window choice is auditable.
"""

import csv
import sys


def main() -> int:
    path, prompt_tokens = sys.argv[1], int(sys.argv[2])
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            try:
                ts, free, inact = r[0], float(r[1]), float(r[2])
            except (ValueError, IndexError):
                continue
            rows.append((ts, free + inact, float(r[4]) if len(r) > 4 else 0.0))
    if len(rows) < 5:
        print("not enough samples")
        return 1

    series = [fi for _, fi, _ in rows]
    peak_i = max(range(len(series)), key=lambda i: series[i])
    # trough after the peak
    trough_i = min(range(peak_i, len(series)), key=lambda i: series[i])
    drop_gb = series[peak_i] - series[trough_i]
    mb_per_token = drop_gb * 1024 / prompt_tokens

    print(f"samples={len(rows)}  peak free+inactive={series[peak_i]:.2f} GB "
          f"(t={rows[peak_i][0]})  trough={series[trough_i]:.2f} GB "
          f"(t={rows[trough_i][0]})")
    print(f"drop={drop_gb:.2f} GB over {prompt_tokens} tokens "
          f"=> {mb_per_token:.3f} MB/token")
    print("thresholds: <0.08 clean | 0.08-0.15 inconclusive | >=0.15 regressed")
    # compact series for the receipt
    step = max(1, len(rows) // 40)
    print("series (free+inactive GB):",
          " ".join(f"{series[i]:.1f}" for i in range(0, len(rows), step)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
