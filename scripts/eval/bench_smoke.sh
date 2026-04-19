#!/bin/bash
# bench_smoke.sh — minimal sanity gate across the 4 healthy M4 MLX presets.
#
# For each preset: launch with workarounds, run validate_capabilities and
# test_radix_cache_repeat, kill server, next. Exits non-zero if any preset
# fails its gates.
#
# Skips the two DeltaNet hybrids (qwen35, coder-next) which are known
# quality-broken even with patch 008 (see project_qwen35_*).
#
# Usage: bash scripts/eval/bench_smoke.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

PORT="${PORT:-23334}"
PRESETS="${PRESETS:-coder-30b devstral qwen3-moe qwen3-32b}"

wait_for_server() {
    for i in $(seq 1 300); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

failed=()
for preset in $PRESETS; do
    echo ""
    echo "============================================"
    echo "  Smoke: $preset"
    echo "============================================"
    # Match only sglang server processes, NOT launch.sh wrapper (whose
    # path contains "m4-sglang-inference" and would be matched by
    # `-f sglang`). macOS BSD pkill alternation `\|` is unreliable, so
    # call pkill once per pattern.
    pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
    pkill -9 -f 'sglang::scheduler' 2>/dev/null || true
    pkill -9 -f 'sglang::detokenizer' 2>/dev/null || true
    sleep 5

    # Always with --disable-radix-cache (patch 001 bug workaround) and --no-thinking
    # for Qwen3 family (chat-template loop workaround).
    EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh "$preset" > "/tmp/smoke_${preset}.log" 2>&1 &
    if ! wait_for_server; then
        echo "FAIL: $preset did not start (see /tmp/smoke_${preset}.log)"
        failed+=("$preset:start")
        continue
    fi
    echo "  server up"

    if [[ "$preset" == qwen* ]]; then
        FLAGS="--no-thinking"
    else
        FLAGS=""
    fi

    if ! python3 scripts/eval/validate_capabilities.py --port "$PORT" $FLAGS; then
        echo "FAIL: $preset validate_capabilities"
        failed+=("$preset:validate")
    fi
    if ! python3 scripts/eval/test_radix_cache_repeat.py --port "$PORT" --iters 3; then
        echo "FAIL: $preset test_radix_cache_repeat"
        failed+=("$preset:radix-repeat")
    fi
done

pkill -9 -f sglang 2>/dev/null || true

echo ""
echo "============================================"
n_presets=$(echo "$PRESETS" | wc -w | tr -d ' ')
if [ ${#failed[@]} -eq 0 ]; then
    echo "  Smoke: ALL $n_presets presets PASS"
    exit 0
else
    echo "  Smoke: FAILED checks: ${failed[*]}"
    exit 1
fi
