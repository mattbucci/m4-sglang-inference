#!/bin/bash
# Comprehensive benchmark: concurrency sweep with proper TPOT measurement.
# Uses sglang.bench_serving for accurate TPOT/TTFT/throughput numbers.
# Works with any model served on localhost:$PORT
#
# Usage: ./scripts/bench/bench_comprehensive.sh "Model Name" [model_hf_id] [port]
# Example: ./scripts/bench/bench_comprehensive.sh "Devstral-24B 4bit" "auto" 23334

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$REPO_DIR/benchmarks"
mkdir -p "$RESULTS_DIR"

source "$REPO_DIR/scripts/common.sh"
activate_venv

LABEL="${1:-unnamed}"
MODEL="${2:-auto}"
PORT="${3:-23334}"
BASE_URL="http://localhost:${PORT}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_LABEL=$(echo "$LABEL" | tr ' /' '_-')
RESULTS_FILE="$RESULTS_DIR/${SAFE_LABEL}_${TIMESTAMP}.txt"

# Auto-detect model from server
if [ "$MODEL" = "auto" ]; then
    MODEL=$(curl -s "$BASE_URL/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
fi

# Wait for server
echo "Waiting for server at $BASE_URL..."
for i in $(seq 1 60); do
    curl -s "$BASE_URL/health" > /dev/null 2>&1 && break
    [ "$i" -eq 60 ] && echo "ERROR: Server not ready after 2min" && exit 1
    sleep 2
done

echo "=============================================" | tee "$RESULTS_FILE"
echo "BENCHMARK: $LABEL" | tee -a "$RESULTS_FILE"
echo "Model:     $MODEL" | tee -a "$RESULTS_FILE"
echo "Hardware:  Apple M4 Pro (64GB)" | tee -a "$RESULTS_FILE"
echo "Backend:   SGLang + MLX" | tee -a "$RESULTS_FILE"
echo "Date:      $(date '+%Y-%m-%d %H:%M')" | tee -a "$RESULTS_FILE"
echo "Server:    $BASE_URL" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Test basic functionality first
echo "--- Smoke test ---" | tee -a "$RESULTS_FILE"
SMOKE=$(curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in 5 words.\"}],\"max_tokens\":20,\"temperature\":0}" 2>&1)
CONTENT=$(echo "$SMOKE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  Response: $CONTENT" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Run sglang.bench_serving at each concurrency level
echo "--- Decode throughput (256 in / 256 out) ---" | tee -a "$RESULTS_FILE"
printf "%-8s  %-10s  %-12s  %-10s  %-10s\n" "Conc" "TPOT(ms)" "Throughput" "TTFT(ms)" "E2E(s)" | tee -a "$RESULTS_FILE"
echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"

for CONC in 1 2 4 8 16; do
    if [ "$CONC" -eq 1 ]; then
        NP=4; RR=1
    elif [ "$CONC" -le 4 ]; then
        NP=$((CONC * 4)); RR=inf
    else
        NP=$((CONC * 4)); RR=inf
    fi

    RESULT=$(python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input 256 \
        --random-output 256 \
        --num-prompts $NP \
        --request-rate $RR 2>&1)

    TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
    THROUGHPUT=$(echo "$RESULT" | grep "Output token throughput" | awk '{print $NF}')
    TTFT=$(echo "$RESULT" | grep "Mean TTFT" | awk '{print $NF}')
    E2E=$(echo "$RESULT" | grep "Mean E2E" | awk '{print $NF}')

    printf "%-8s  %-10s  %-12s  %-10s  %-10s\n" "$CONC" "$TPOT" "$THROUGHPUT" "$TTFT" "$E2E" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"

# Context sweep with sglang.bench_serving
echo "--- Context sweep (single user, 64 out) ---" | tee -a "$RESULTS_FILE"
printf "%-10s  %-10s  %-12s  %-10s\n" "Input" "TPOT(ms)" "Throughput" "TTFT(ms)" | tee -a "$RESULTS_FILE"
echo "----------------------------------------------------" | tee -a "$RESULTS_FILE"

for INPUT_LEN in 128 512 1024 4096 8192 16384 32768 65536 131072; do
    RESULT=$(python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input $INPUT_LEN \
        --random-output 64 \
        --num-prompts 1 \
        --request-rate 1 2>&1)

    TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
    THROUGHPUT=$(echo "$RESULT" | grep "Output token throughput" | awk '{print $NF}')
    TTFT=$(echo "$RESULT" | grep "Mean TTFT" | awk '{print $NF}')

    # Stop if server crashed or error
    if [ -z "$TPOT" ] || echo "$RESULT" | grep -qi "error\|exception\|oom"; then
        printf "%-10s  %-10s  %-12s  %-10s\n" "$INPUT_LEN" "OOM/ERR" "-" "-" | tee -a "$RESULTS_FILE"
        echo "  (stopping context sweep — server may have crashed)" | tee -a "$RESULTS_FILE"
        break
    fi

    printf "%-10s  %-10s  %-12s  %-10s\n" "$INPUT_LEN" "$TPOT" "$THROUGHPUT" "$TTFT" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"
