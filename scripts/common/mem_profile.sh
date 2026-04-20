#!/bin/bash
# Memory profiler — logs vm_stat free/inactive + every SGLang process RSS
# every N seconds while a bench is running. Output is a CSV that lets you
# correlate memory pressure with bench progress.
#
# Usage:
#   bash scripts/common/mem_profile.sh > /tmp/mem_profile.csv &
#   PROF_PID=$!
#   ... run bench ...
#   kill $PROF_PID
#
# Env:
#   PROF_INTERVAL — seconds between samples (default 5)
set -uo pipefail
INTERVAL="${PROF_INTERVAL:-5}"

echo "ts_iso,free_gb,inactive_gb,wired_gb,sglang_rss_gb,sglang_pid"
trap 'exit 0' TERM INT

while true; do
    TS=$(date -u "+%Y-%m-%dT%H:%M:%SZ")
    read FREE INACTIVE WIRED <<<$(vm_stat | awk '
        /Pages free/     { f=$3 }
        /Pages inactive/ { i=$3 }
        /Pages wired down/ { w=$4 }
        END              { printf "%.2f %.2f %.2f", f*16384/(1024*1024*1024), i*16384/(1024*1024*1024), w*16384/(1024*1024*1024) }
    ')
    SGPID=$(pgrep -f sglang.launch_server | head -1)
    if [ -n "$SGPID" ]; then
        SGRSS=$(ps -p "$SGPID" -o rss= 2>/dev/null | awk '{printf "%.2f", $1/(1024*1024)}')
    else
        SGRSS="0"
        SGPID=""
    fi
    echo "${TS},${FREE},${INACTIVE},${WIRED},${SGRSS},${SGPID}"
    sleep "$INTERVAL"
done
