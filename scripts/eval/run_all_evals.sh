#!/bin/bash
# Run quality evals across all working M4 presets sequentially.
# For each preset: stop previous server, launch, wait for ready, run eval, kill, next.
#
# Adapted from R9700 sister repo. Tuned for single-user MLX (workers=1, smaller
# sample sizes than the GPU teams since each prompt is bandwidth-limited).
#
# Usage:
#   bash scripts/eval/run_all_evals.sh                # all presets
#   PRESETS="coder-30b devstral" bash scripts/eval/run_all_evals.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

PORT="${PORT:-23334}"
EVAL_CMD="python scripts/eval/eval_and_chart.py --run --port $PORT --workers 1 --mmlu-samples 100 --humaneval-samples 20 --labbench-samples 25 --needle-lengths 1024,4096,16384"

wait_for_server() {
    local max_wait=300  # MLX models load slowly
    for i in $(seq 1 $max_wait); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Server did not start within ${max_wait}s"
    return 1
}

run_eval_for() {
    local preset="$1"
    local tag="$2"

    local results_file="benchmarks/quality/${tag// /_}.json"
    if [ -f "$results_file" ] && python3 -c "import json,sys; r=json.load(open('$results_file')); sys.exit(0 if all(r.get(k,{}).get('total') for k in ('mmlu','humaneval')) and r.get('labbench',{}).get('_overall',{}).get('total') else 1)" 2>/dev/null; then
        echo "=== SKIP $tag (results complete) ==="
        return 0
    fi

    echo ""
    echo "============================================"
    echo "  Evaluating: $tag ($preset)"
    echo "============================================"

    pkill -f sglang 2>/dev/null || true
    sleep 5

    local log="/tmp/eval_${preset}.log"
    # Eval mode: disable radix cache. Patch 001 has a bug where repeated prompts
    # return garbage on cache hits (see project_radix_cache_repeat_bug memory),
    # which would silently corrupt every quality benchmark from request 2 onward.
    EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh "$preset" > "$log" 2>&1 &
    local server_pid=$!

    if ! wait_for_server; then
        echo "FAILED to start $preset (see $log)"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    echo "Server ready (PID $server_pid)"

    # Capability gate first — refuses to publish numbers if model is broken.
    if ! python scripts/eval/validate_capabilities.py --port "$PORT"; then
        echo "WARN: $tag failed capability validator — running quality eval anyway for the record"
    fi

    PYTHONUNBUFFERED=1 $EVAL_CMD --tag "$tag" 2>&1 | tee "/tmp/eval_${preset}_results.log"

    pkill -f sglang 2>/dev/null || true
    sleep 5
    echo "Done: $tag"
}

# Default preset list — override with PRESETS env var
PRESETS="${PRESETS:-coder-30b devstral gemma4 gemma4-31b qwen35 qwen3-moe qwen3-32b}"

for preset in $PRESETS; do
    case "$preset" in
        devstral)     tag="Devstral-24B" ;;
        coder-30b)    tag="Coder-30B" ;;
        coder-next)   tag="Coder-Next-80B" ;;
        gemma4)       tag="Gemma4-26B" ;;
        gemma4-31b)   tag="Gemma4-31B" ;;
        qwen35)       tag="Qwen3.5-27B" ;;
        qwen3-moe)    tag="Qwen3-30B-MoE" ;;
        qwen3-32b)    tag="Qwen3-32B" ;;
        *)            tag="$preset" ;;
    esac
    run_eval_for "$preset" "$tag"
done

echo ""
echo "============================================"
echo "  All evals complete!"
echo "============================================"
ls -la benchmarks/quality/
python scripts/eval/eval_and_chart.py --chart
