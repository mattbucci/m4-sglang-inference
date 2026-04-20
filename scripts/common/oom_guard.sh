#!/bin/bash
# OOM guard daemon — runs in the background while a long-context bench
# is in flight, watches free+inactive memory, and kills the SGLang
# server BEFORE the OS has to start swapping/freezing.
#
# On Apple unified memory the OS doesn't dump load like a Linux OOM
# killer would; it freezes for tens of seconds → minutes when MLX
# allocates past available physical RAM. Once that happens, recovery
# usually requires a hard reboot.
#
# Usage:
#   bash scripts/common/oom_guard.sh &
#   GUARD_PID=$!
#   ... run your long-context bench ...
#   kill $GUARD_PID 2>/dev/null
#
# Tunables (env vars):
#   GUARD_KILL_GB   — kill server when free+inactive drops below this many GB (default 4)
#   GUARD_WARN_GB   — log a warning when below this (default 8)
#   GUARD_INTERVAL  — seconds between checks (default 10)
#   GUARD_LOG       — log path (default /tmp/oom_guard.log)
set -uo pipefail

KILL_GB="${GUARD_KILL_GB:-8}"
WARN_GB="${GUARD_WARN_GB:-12}"
INTERVAL="${GUARD_INTERVAL:-2}"
LOG="${GUARD_LOG:-/tmp/oom_guard.log}"

# Tight defaults: macOS starts freezing WELL before free hits 0 — the OS is
# already swapping by the time free < 4GB. The guard itself competes for
# memory and can be killed by pressure before it gets to pkill the server.
# 8GB kill + 2s interval gives margin to run + write the log + kill SGLang
# before macOS locks up.
#
# PRIOR INCIDENT (2026-04-20, Coder-Next-80B): 10s interval and 3-5GB kill
# threshold were both too loose for a 42GB MLX model load; guard itself was
# killed by memory pressure and the system OOM'd. Don't repeat.

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG" >&2; }

free_gb() {
    # Use awk to avoid printf locale issues. Pages free + inactive,
    # 16KB pages on M-series.
    vm_stat | awk '
        /Pages free/     { f=$3 }
        /Pages inactive/ { i=$3 }
        END              { printf "%.2f", (f+i)*16384/(1024*1024*1024) }
    '
}

log "oom_guard started: kill<${KILL_GB}GB warn<${WARN_GB}GB interval=${INTERVAL}s pid=$$"
trap 'log "oom_guard exiting"; exit 0' TERM INT

while true; do
    FG=$(free_gb)
    # Compare with awk (handles decimals); 0 = below threshold
    awk -v f="$FG" -v k="$KILL_GB" 'BEGIN{exit (f<k)?0:1}'
    BELOW_KILL=$?
    awk -v f="$FG" -v w="$WARN_GB" 'BEGIN{exit (f<w)?0:1}'
    BELOW_WARN=$?

    if [ "$BELOW_KILL" -eq 0 ]; then
        # Snapshot for forensics
        SGLANG_PIDS=$(pgrep -f sglang.launch_server | tr '\n' ' ')
        log "CRITICAL free=${FG}GB < ${KILL_GB}GB — killing SGLang pids: ${SGLANG_PIDS:-none}"
        ps -o pid,rss,command -p ${SGLANG_PIDS:-1} 2>/dev/null | tee -a "$LOG" >&2 || true
        if [ -n "${SGLANG_PIDS// }" ]; then
            pkill -9 -f sglang.launch_server || true
            sleep 2
            pkill -9 -f sglang.launch_server || true
        fi
        log "kill complete; guard sleeping 30s before resuming watch"
        sleep 30
    elif [ "$BELOW_WARN" -eq 0 ]; then
        log "WARN free=${FG}GB < ${WARN_GB}GB"
    fi
    sleep "$INTERVAL"
done
