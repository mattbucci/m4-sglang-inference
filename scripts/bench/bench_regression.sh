#!/bin/bash
# Performance regression test for M4 Pro MLX inference.
#
# Uses sglang.bench_serving for accurate TPOT measurement.
# Compares against stored baselines and flags regressions (>10% slower).
#
# Usage:
#   ./scripts/bench/bench_regression.sh devstral      # Run one model
#   BASELINE=save ./scripts/bench/bench_regression.sh devstral  # Save baseline
#
# Baselines stored in benchmarks/baselines.json

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$REPO_DIR/scripts/common.sh"
activate_venv
setup_mlx_env

BASELINES="$REPO_DIR/benchmarks/baselines.json"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:$PORT"
THRESHOLD="${THRESHOLD:-10}"  # % regression threshold
MODEL_FILTER="${1:?Usage: $0 <model_key> (e.g. devstral, coder-30b, gemma4, qwen35, coder-next)}"
SAVE_BASELINE="${BASELINE:-}"

# Model key → HuggingFace model ID for sglang.bench_serving
get_bench_model() {
    case "$1" in
        devstral)   echo "mistralai/Devstral-Small-2-24B-Instruct-2512" ;;
        coder-30b)  echo "Qwen/Qwen3-Coder-30B-A3B-Instruct" ;;
        gemma4)     echo "google/gemma-4-26B-A4B-it" ;;
        qwen35)     echo "Qwen/Qwen3.5-27B" ;;
        coder-next) echo "Qwen/Qwen3-Coder-Next" ;;
        *)          echo "$1" ;;
    esac
}

bench_one() {
    local model="$1" input_len="$2" output_len="$3" num_prompts="$4"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$model" \
        --dataset-name random \
        --random-input "$input_len" \
        --random-output "$output_len" \
        --num-prompts "$num_prompts" \
        --request-rate inf \
        --disable-tqdm 2>&1
}

extract_metrics() {
    local output="$1"
    local tpot throughput ttft
    tpot=$(echo "$output" | grep "Mean TPOT" | awk '{print $NF}' | sed 's/ms//')
    throughput=$(echo "$output" | grep "Output token throughput" | awk '{print $NF}')
    ttft=$(echo "$output" | grep "Mean TTFT" | awk '{print $NF}' | sed 's/ms//')
    echo "${tpot:-0} ${throughput:-0} ${ttft:-0}"
}

run_bench() {
    local key="$1"
    local model
    model=$(get_bench_model "$key")
    echo ""
    echo "=== $key ==="

    echo "  Single user (128 in, 50 out)..."
    local single_out
    single_out=$(bench_one "$model" 128 50 1)
    read -r tpot1 tp1 ttft1 <<< "$(extract_metrics "$single_out")"
    echo "    TPOT: ${tpot1}ms  Throughput: ${tp1} tok/s  TTFT: ${ttft1}ms"

    echo "  Multi user @8 (128 in, 50 out)..."
    local multi_out
    multi_out=$(bench_one "$model" 128 50 16)
    read -r tpot8 tp8 ttft8 <<< "$(extract_metrics "$multi_out")"
    echo "    TPOT: ${tpot8}ms  Throughput: ${tp8} tok/s  TTFT: ${ttft8}ms"

    python3 -c "
import json
result = {
    'single_tpot_ms': float('${tpot1}'),
    'single_throughput': float('${tp1}'),
    'single_ttft_ms': float('${ttft1}'),
    'multi8_tpot_ms': float('${tpot8}'),
    'multi8_throughput': float('${tp8}'),
    'multi8_ttft_ms': float('${ttft8}'),
}
print(json.dumps(result))
" 2>/dev/null
}

compare_baseline() {
    local key="$1" current="$2"
    if [ ! -f "$BASELINES" ]; then
        echo "  (no baseline file — run with BASELINE=save to create)"
        return 0
    fi

    python3 -c "
import json, sys

with open('$BASELINES') as f:
    baselines = json.load(f)

if '$key' not in baselines:
    print('  (no baseline for $key)')
    sys.exit(0)

base = baselines['$key']
curr = json.loads('$current')
threshold = $THRESHOLD

failed = False
for metric in ['single_tpot_ms', 'single_ttft_ms', 'multi8_tpot_ms']:
    b = base.get(metric, 0)
    c = curr.get(metric, 0)
    if b > 0 and c > 0:
        pct = ((c - b) / b) * 100
        status = 'REGRESSION' if pct > threshold else 'ok'
        if pct > threshold:
            failed = True
        print(f'  {metric}: {b:.1f} -> {c:.1f} ({pct:+.1f}%) [{status}]')

for metric in ['single_throughput', 'multi8_throughput']:
    b = base.get(metric, 0)
    c = curr.get(metric, 0)
    if b > 0 and c > 0:
        pct = ((c - b) / b) * 100
        status = 'REGRESSION' if pct < -threshold else 'ok'
        if pct < -threshold:
            failed = True
        print(f'  {metric}: {b:.1f} -> {c:.1f} ({pct:+.1f}%) [{status}]')

if failed:
    print()
    print('  *** PERFORMANCE REGRESSION DETECTED ***')
    sys.exit(1)
else:
    print('  All metrics within threshold.')
" 2>/dev/null
}

# Wait for server
echo "Waiting for server at $BASE_URL..."
for i in $(seq 1 30); do
    curl -s "$BASE_URL/health" > /dev/null 2>&1 && break
    [ "$i" -eq 30 ] && echo "ERROR: Server not ready" && exit 1
    sleep 2
done
echo "Server ready."

# Warm up MLX kernels — cold start causes massive false regression
echo "Warming up (5 requests)..."
for i in $(seq 1 5); do
    curl -s "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Warmup request $i: explain gravity briefly\"}],\"max_tokens\":50,\"temperature\":0}" > /dev/null 2>&1
done
echo "Warm."

# Run benchmarks
FAILED=0

echo ""
echo "============================================"
echo "M4 Pro MLX Performance Regression Test"
echo "Threshold: ${THRESHOLD}% deviation"
echo "============================================"

RESULT=$(run_bench "${MODEL_FILTER}")
RESULT_JSON=$(echo "$RESULT" | tail -1)

echo ""
echo "--- Comparing against baseline ---"
compare_baseline "${MODEL_FILTER}" "$RESULT_JSON" || FAILED=1

if [ -n "$SAVE_BASELINE" ]; then
    echo ""
    echo "Saving baseline..."
    python3 -c "
import json, os
path = '$BASELINES'
baselines = {}
if os.path.exists(path):
    with open(path) as f:
        baselines = json.load(f)
baselines['${MODEL_FILTER}'] = json.loads('$RESULT_JSON')
with open(path, 'w') as f:
    json.dump(baselines, f, indent=2)
print(f'Saved baseline for ${MODEL_FILTER} to {path}')
"
fi

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "RESULT: REGRESSION DETECTED"
    exit 1
else
    echo "RESULT: PASS"
fi
