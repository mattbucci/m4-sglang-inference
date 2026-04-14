#!/bin/bash
# Benchmark all models at 256K context with quantized KV cache.
#
# Strategy: try fp8 first. If the context sweep doesn't reach 128K,
# retry with turboquant (4-bit KV, ~3.5x smaller). Between each model
# run, kill ALL sglang processes and reclaim memory so the next model
# gets a full-size KV pool.
#
# Usage:
#   ./scripts/bench/bench_256k_all.sh
#   MODELS="devstral coder-30b" ./scripts/bench/bench_256k_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"

PORT=23334
BASE_URL="http://localhost:${PORT}"
BENCH_LOG="benchmarks/bench_256k_$(date +%Y%m%d_%H%M%S).log"
mkdir -p benchmarks

# Model definitions: preset|display_name|bench_key|max_running|chunked
DEVSTRAL="devstral|Devstral-24B 4-bit|devstral-24b-4bit|1|4096"
CODER30B="coder-30b|Coder-30B 4-bit|coder-30b-4bit|1|4096"
CODERNEXT="coder-next|Coder-Next 80B 4-bit|coder-next-80b-4bit|1|2048"
GEMMA4="gemma4|Gemma 4 26B 4-bit|gemma4-26b-4bit|1|4096"
GEMMA431B="gemma4-31b|Gemma 4 31B 4-bit|gemma4-31b-4bit|1|4096"
QWEN35="qwen35|Qwen3.5-27B 4-bit|qwen35-27b-4bit|1|4096"

ALL_MODELS="${MODELS:-devstral coder-30b coder-next gemma4 gemma4-31b qwen35}"
CTX=262144

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$BENCH_LOG"; }

# ── Process management ─────────────────────────────────────────────

nuke_sglang() {
    # Kill every sglang-related process: server, scheduler, detokenizer,
    # resource_tracker, bench_serving. Uses pkill patterns rather than
    # port-based lookup to catch orphaned children.
    log "Killing all sglang/bench processes..."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.bench_serving" 2>/dev/null || true
    pkill -9 -f "sglang::scheduler" 2>/dev/null || true
    pkill -9 -f "sglang::detokenizer" 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
    # Also kill by port in case patterns miss anything
    lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
    local remaining
    remaining=$(ps aux | grep -i sglang | grep -v grep | wc -l | tr -d ' ')
    if [ "$remaining" -gt 0 ]; then
        log "WARNING: $remaining sglang processes still alive, force-killing..."
        ps aux | grep -i sglang | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

reclaim_memory() {
    # Force macOS to free memory from killed processes.
    # Without this, sys_available stays low and the KV pool auto-sizer
    # picks a tiny pool.
    nuke_sglang
    log "Reclaiming memory..."

    # Clear Python/MLX cached allocations
    activate_venv
    python3 -c "
import gc; gc.collect()
try:
    import mlx.core as mx
    mx.clear_memory_cache()
except: pass
" 2>/dev/null || true

    # macOS memory purge (needs sudo, skips silently if unavailable)
    sudo purge 2>/dev/null || true

    # Give the OS time to settle
    sleep 8

    local avail
    avail=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().available / (1024**3):.1f}')" 2>/dev/null || echo "?")
    log "Available memory: ${avail} GB"
}

wait_for_server() {
    local max_wait=600
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

get_model_config() {
    case "$1" in
        devstral)   echo "$DEVSTRAL" ;;
        coder-30b)  echo "$CODER30B" ;;
        coder-next) echo "$CODERNEXT" ;;
        gemma4)     echo "$GEMMA4" ;;
        gemma4-31b) echo "$GEMMA431B" ;;
        qwen35)     echo "$QWEN35" ;;
    esac
}

# ── Launch + benchmark one model ───────────────────────────────────

launch_and_bench() {
    local preset="$1"
    local kv_mode="$2"
    local config
    config=$(get_model_config "$preset")

    local display_name bench_key max_run chunked
    IFS='|' read -r _ display_name bench_key max_run chunked <<< "$config"

    log "=============================================="
    log "  $display_name"
    log "  KV: $kv_mode  Context: $CTX"
    log "=============================================="

    # Clean slate
    reclaim_memory

    # Launch
    local server_log="/tmp/sglang_256k_${preset}.log"
    log "Launching server..."
    ./scripts/launch.sh "$preset" \
        --context-length "$CTX" \
        --max-running "$max_run" \
        --chunked-prefill "$chunked" \
        --kv-cache "$kv_mode" \
        --watchdog 1800 \
        > "$server_log" 2>&1 &
    local server_pid=$!

    log "Waiting for server (PID $server_pid)..."
    if ! wait_for_server; then
        log "ERROR: Server did not start"
        grep -E "pool_size|Error|OOM|error" "$server_log" 2>/dev/null | tail -5 | while IFS= read -r line; do log "  $line"; done
        nuke_sglang
        return 1
    fi
    log "Server ready"

    # Log pool sizing
    grep "pool_size" "$server_log" 2>/dev/null | tail -1 | while IFS= read -r line; do log "  $line"; done

    # Still alive?
    if ! kill -0 $server_pid 2>/dev/null; then
        log "ERROR: Server died after health check"
        nuke_sglang
        return 1
    fi

    # Benchmark
    log "Running benchmark..."
    local bench_output="benchmarks/${bench_key}/results.json"
    if python3 scripts/bench/bench_all_unified.py \
        --port $PORT \
        --name "$display_name" \
        --context-max "$CTX" \
        --concurrency-max 8 \
        --kv-cache "$kv_mode" \
        --output "$bench_output" \
        --skip-charts \
        2>&1 | tee -a "$BENCH_LOG"; then
        log "Results: $bench_output"
        nuke_sglang
        return 0
    else
        log "ERROR: Benchmark script failed"
        nuke_sglang
        return 1
    fi
}

max_successful_ctx() {
    # Read results.json and return the largest context that didn't error
    local bench_key="$1"
    python3 -c "
import json, sys
try:
    with open('benchmarks/${bench_key}/results.json') as f:
        d = json.load(f)
    valid = [p['context'] for p in d['context_sweep'] if 'error' not in p]
    print(max(valid) if valid else 0)
except:
    print(0)
" 2>/dev/null || echo "0"
}

# ── Main loop ──────────────────────────────────────────────────────

log "Starting 256K benchmark sweep"
log "Models: $ALL_MODELS"
log "Target context: $CTX"
log "Log: $BENCH_LOG"
echo ""

RESULTS_SUMMARY=""

for preset in $ALL_MODELS; do
    config=$(get_model_config "$preset")
    IFS='|' read -r _ display_name bench_key _ _ <<< "$config"

    # ── Try fp8 ──
    if launch_and_bench "$preset" "fp8"; then
        max_ctx=$(max_successful_ctx "$bench_key")
        log "$display_name fp8: max successful context = $max_ctx"

        if [ "$max_ctx" -ge 131072 ]; then
            RESULTS_SUMMARY+="  $display_name: fp8 (max ctx ${max_ctx})\n"
            log "PASS: $display_name fp8 reached $max_ctx"
            echo ""
            continue
        fi

        # fp8 didn't reach 128K — retry with turboquant
        log "$display_name: fp8 only reached ${max_ctx}, retrying with turboquant..."
    else
        log "$display_name: fp8 launch/bench failed, trying turboquant..."
    fi

    # ── Fallback: turboquant ──
    if launch_and_bench "$preset" "turboquant"; then
        max_ctx=$(max_successful_ctx "$bench_key")
        RESULTS_SUMMARY+="  $display_name: turboquant (max ctx ${max_ctx})\n"
        log "PASS: $display_name turboquant reached $max_ctx"
    else
        # Keep whatever partial results we have
        max_ctx=$(max_successful_ctx "$bench_key")
        RESULTS_SUMMARY+="  $display_name: PARTIAL (max ctx ${max_ctx})\n"
        log "WARN: $display_name both modes failed (partial data at $max_ctx)"
    fi

    echo ""
done

# ── Charts ──
log "Regenerating charts..."
python3 scripts/bench/generate_charts.py 2>&1 | tee -a "$BENCH_LOG"

log ""
log "=============================================="
log "RESULTS"
log "=============================================="
echo -e "$RESULTS_SUMMARY" | tee -a "$BENCH_LOG"
log "Log: $BENCH_LOG"
log "Done!"
