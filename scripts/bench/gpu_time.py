#!/usr/bin/env python3
"""GPU time per process on Apple Silicon, from the driver's per-client counters.

The MLX server shares the GPU with the desktop: the window server, any app
that draws, the terminal the bench is driven from. This reads who got the GPU,
without elevated rights, so a timed arm can carry the server's share.

    python scripts/bench/gpu_time.py sample gpu-samples.txt --interval 5 --hours 6 &
    python scripts/bench/gpu_time.py report gpu-samples.txt arm-a=13:07:00-13:09:03 arm-b=13:09:23-13:11:18

`sample` appends one line per interval: wall time, device / renderer / tiler
utilisation %, then GPU milliseconds per wall second for every process above
1 ms/s, largest first. One sample is two `ioreg` calls, about 10 ms each.
`report` averages named windows of that file: the server (every python
process), the window server, everything else, and with `--by-process` each
other process on its own.

The counters are `accumulatedGPUTime` (ns) of each `IOGPUDeviceUserClient`,
summed per owning process. A desktop client's time grows while a long compute
kernel runs beside it, so a share is a flag that an arm had company, not the
exact cost: measure the cost with the same arm on a quiet machine
(`pmset displaysleepnow` removes the window server's part).
"""
import argparse
import collections
import re
import subprocess
import time

CREATOR = re.compile(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"')
GPU_NS = re.compile(r'"accumulatedGPUTime"=(\d+)')
STAT = re.compile(r'"(Device|Renderer|Tiler) Utilization %"=(\d+)')
ITEM = re.compile(r"(\S+?)\[(\d+)\]:(\d+)")


def ioreg(*args: str) -> str:
    return subprocess.run(
        ["/usr/sbin/ioreg", *args], capture_output=True, text=True
    ).stdout


def read_counters():
    """(time, {(pid, name): ns of GPU time}, {stat: %})."""
    totals, usage = {}, None
    for line in ioreg("-r", "-c", "IOGPUDeviceUserClient", "-l", "-w", "0").splitlines():
        if '"AppUsage"' in line:
            usage = sum(int(ns) for ns in GPU_NS.findall(line))
        match = CREATOR.search(line)
        if match and usage is not None:
            key = (int(match.group(1)), match.group(2).strip())
            totals[key] = totals.get(key, 0) + usage
            usage = None
    util = dict(STAT.findall(ioreg("-r", "-d", "1", "-c", "IOAccelerator", "-w", "0")))
    return time.time(), totals, util


def sample(out: str, interval: float, hours: float) -> None:
    end = time.time() + 3600 * hours
    prev_t, prev, _ = read_counters()
    while time.time() < end:
        time.sleep(interval)
        now, cur, util = read_counters()
        rates = []
        for key, ns in cur.items():
            ms_per_s = (ns - prev.get(key, ns)) / 1e6 / (now - prev_t)
            if ms_per_s > 1:
                rates.append((ms_per_s, key))
        rates.sort(reverse=True)
        stamp = time.strftime("%H:%M:%S", time.localtime(now))
        stats = " ".join(
            f"{short}={util.get(name, '?')}"
            for short, name in (("dev", "Device"), ("rend", "Renderer"), ("tiler", "Tiler"))
        )
        procs = " ".join(
            f"{name.replace(' ', '_')}[{pid}]:{rate:.0f}" for rate, (pid, name) in rates
        )
        with open(out, "a") as fh:
            fh.write(f"{stamp} {stats} {procs}\n")
        prev_t, prev = now, cur


def read_samples(path: str):
    rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        util = dict(part.split("=") for part in parts[1:4])
        procs = collections.Counter()
        for name, _pid, ms in ITEM.findall(line):
            procs[name] += int(ms)
        rows.append((parts[0], util, procs))
    return rows


def report(path: str, windows, by_process: bool) -> None:
    rows = read_samples(path)
    print("| window | samples | server | WindowServer | other | device % | renderer % |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    detail = []
    for spec in windows:
        label, span = spec.split("=")
        start, stop = span.split("-")
        sel = [row for row in rows if start <= row[0] <= stop]
        if not sel:
            print(f"| {label} | 0 | | | | | |")
            continue
        total = collections.Counter()
        for _, _, procs in sel:
            total.update(procs)
        n = len(sel)
        server = sum(ms for name, ms in total.items() if name.startswith("python")) / n
        window_server = total["WindowServer"] / n
        other = sum(total.values()) / n - server - window_server

        def mean(key):
            vals = [int(util[key]) for _, util, _ in sel if util[key].isdigit()]
            return sum(vals) / max(len(vals), 1)

        print(
            f"| {label} | {n} | {server:.0f} | {window_server:.0f} | {other:.0f} "
            f"| {mean('dev'):.0f} | {mean('rend'):.0f} |"
        )
        rest = {
            name: ms / n
            for name, ms in total.most_common()
            if not name.startswith("python") and name != "WindowServer"
        }
        detail.append((label, rest))
    if by_process:
        print()
        for label, rest in detail:
            print(f"{label}: " + ", ".join(f"{name} {ms:.0f}" for name, ms in rest.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="append one line per interval to a file")
    s.add_argument("out")
    s.add_argument("--interval", type=float, default=5.0, help="seconds")
    s.add_argument("--hours", type=float, default=6.0)
    r = sub.add_parser("report", help="mean GPU ms per second over named windows")
    r.add_argument("samples")
    r.add_argument("windows", nargs="+", metavar="label=HH:MM:SS-HH:MM:SS")
    r.add_argument("--by-process", action="store_true")
    a = parser.parse_args()
    if a.cmd == "sample":
        sample(a.out, a.interval, a.hours)
    else:
        report(a.samples, a.windows, a.by_process)


if __name__ == "__main__":
    main()
