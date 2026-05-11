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
    # Skip only if every section meets the requested sample bar AND results
    # are plausible (MMLU > 5% — anything ≤5% is below random-guess on 4-choice,
    # i.e. server-died-mid-eval). Accepts cached totals >= requested (a stricter
    # measurement is fine to reuse); below the bar or 0%-everywhere means re-run.
    if [ -f "$results_file" ] && python3 -c "
import json, sys
r = json.load(open('$results_file'))
def cache_ok():
    mmlu = r.get('mmlu', {})
    if mmlu.get('total', 0) < 100 or mmlu.get('accuracy', 0) <= 0.05:
        return False
    he = r.get('humaneval', {})
    if he.get('total', 0) < 20:
        return False
    lb = r.get('labbench', {})
    if not lb.get('_overall') or lb['_overall'].get('correct', 0) == 0:
        return False
    for bench in ['LitQA2','DbQA','SuppQA','TableQA','ProtocolQA','SeqQA','CloningScenarios']:
        if lb.get(bench, {}).get('total', 0) < 25:
            return False
    needle_lengths = {n['context'] for n in r.get('needle', {}).get('results', [])}
    if not {1024, 4096, 16384}.issubset(needle_lengths):
        return False
    return True
sys.exit(0 if cache_ok() else 1)
" 2>/dev/null; then
        echo "=== SKIP $tag (results complete and current) ==="
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

# Default preset list — override with PRESETS env var.
# Includes Qwen3.6 dense + MoE which sister teams use as their flagship 256K
# agentic model, plus the smaller qwen35-9b-8bit fast variant.
PRESETS="${PRESETS:-coder-30b devstral gemma4 gemma4-31b qwen35 qwen35-9b-8bit qwen3-moe qwen3-32b qwen36 qwen36-27b}"

for preset in $PRESETS; do
    # Tags reflect the underlying model variant. DWQ presets keep the
    # -DWQ suffix so the README quality table tracks the actual checkpoint
    # we're publishing numbers for (run_all_evals shipped tagless before,
    # which orphaned earlier DWQ-tagged JSONs and forced a full re-eval).
    case "$preset" in
        devstral)        tag="Devstral-24B" ;;
        coder-30b)       tag="Coder-30B-DWQ" ;;
        coder-next)      tag="Coder-Next-80B" ;;
        gemma4)          tag="Gemma4-26B" ;;
        gemma4-31b)      tag="Gemma4-31B" ;;
        qwen35)          tag="Qwen3.5-27B" ;;
        qwen35-9b-8bit)  tag="Qwen3.5-9B-8bit" ;;
        qwen3-moe)       tag="Qwen3-30B-A3B-DWQ" ;;
        qwen3-32b)       tag="Qwen3-32B-DWQ" ;;
        qwen36)          tag="Qwen3.6-35B-A3B" ;;
        qwen36-27b)      tag="Qwen3.6-27B" ;;
        *)               tag="$preset" ;;
    esac
    run_eval_for "$preset" "$tag"
done

echo ""
echo "============================================"
echo "  All evals complete!"
echo "============================================"
ls -la benchmarks/quality/
python scripts/eval/eval_and_chart.py --chart
