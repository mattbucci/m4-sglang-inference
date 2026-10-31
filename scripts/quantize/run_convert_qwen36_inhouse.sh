#!/bin/bash
# Detached conversion with a memory watchdog: macOS has no OOM killer, so the
# converter is killed when kernel-available memory drops under GUARD_GB.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
LOG="${LOG:-$REPO/benchmarks/quality/convert_qwen36_inhouse.log}"
GUARD_GB="${GUARD_GB:-6}"; TOTAL_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
source ~/miniforge3/etc/profile.d/conda.sh && conda activate sglang-0523
nohup python scripts/quantize/convert_qwen36_inhouse.py "$@" > "$LOG" 2>&1 < /dev/null & PID=$!
echo "converter pid $PID log $LOG"
while kill -0 $PID 2>/dev/null; do
    level=$(sysctl -n kern.memorystatus_level); free=$(( level * TOTAL_GB / 100 ))
    if [ "$free" -lt "$GUARD_GB" ]; then
        echo "[$(date -u +%FT%TZ)] WATCHDOG: kernel-available ${free} GB < ${GUARD_GB} GB, killing converter $PID" | tee -a "$LOG"
        kill -9 $PID; break
    fi
    sleep 2
done
wait $PID 2>/dev/null; rc=$?; echo "CONVERT_DONE rc=$rc" | tee -a "$LOG"
