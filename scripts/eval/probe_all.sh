#!/bin/bash
# Run the probe trio (thinking + vision + codegen) against every M4 preset.
# Ported in concept from the 3090 sister repo's test_capabilities_all.sh — but
# uses the deeper content-aware probes (probe_thinking / probe_vision /
# probe_codegen) instead of the validator's loose keyword grep.
#
# For each preset:
#   stop server -> launch -> wait /health=200 -> run applicable probes -> kill -> next
#
# Auto-selects probes per preset (skipping irrelevant ones to save time):
#   - coder-30b / coder-next:               codegen only (non-thinking, text-only)
#   - devstral:                             codegen + vision (Dense+VLM)
#   - qwen3-moe / qwen3-32b:                codegen only (no-thinking path used in evals)
#   - qwen35 / qwen35-9b-8bit:              vision (DeltaNet+VL); thinking skipped (greedy loop)
#   - qwen36 / qwen36-27b:                  codegen + vision (DeltaNet+MoE+VL); thinking too
#   - gemma4 / gemma4-31b:                  codegen + thinking (vision blocked, no preprocessor)
#   - nemotron-30b:                         codegen (thinking-mode; reasoning parser not wired yet)
#
# Output: JSON-per-preset under benchmarks/quality/probe-trio/<preset>.json with
# {thinking, vision, codegen} verdicts and per-probe rc.
#
# Usage:
#   bash scripts/eval/probe_all.sh                       # default preset list
#   bash scripts/eval/probe_all.sh coder-30b devstral    # specific presets
#   PRESETS="qwen36 qwen36-27b" bash scripts/eval/probe_all.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/common.sh"

PORT="${PORT:-23334}"
LOG_DIR="${LOG_DIR:-/tmp/probe-trio-logs}"
RESULTS_DIR="$REPO_DIR/benchmarks/quality/probe-trio"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Per-preset probe selection: which of {thinking, vision, video, codegen} to run.
# Vision/video probes need --enable-multimodal and a working VLM bridge; thinking
# probe will loop on Qwen3 family (greedy MLX) so we skip those. Codegen is
# the most universally applicable — it works on any text-completion model.
# Video probe sends multi-image frames client-side (bypasses SGLang's video
# processor which pulls in torchcodec/decord); arch must support multi-image
# input (Qwen-VL, Gemma 4, Devstral) — same patch 013 plumbing as vision.
probes_for() {
    case "$1" in
        coder-30b)         echo "codegen" ;;
        coder-next)        echo "codegen" ;;
        devstral)          echo "codegen vision" ;;  # Devstral arch is image-only, no video
        gemma4)            echo "codegen thinking vision video" ;;
        gemma4-31b)        echo "codegen thinking vision video" ;;
        qwen3-moe)         echo "codegen" ;;   # --no-thinking in evals
        qwen3-32b)         echo "codegen" ;;   # --no-thinking in evals
        qwen35)            echo "codegen vision video" ;;  # thinking skipped — known greedy loop
        qwen35-9b-8bit)    echo "codegen vision video" ;;
        qwen36)            echo "codegen vision video thinking" ;;
        qwen36-27b)        echo "codegen vision video thinking" ;;
        nemotron-30b)      echo "codegen" ;;
        nemotron-omni)     echo "codegen vision thinking" ;;  # Omni has image, no video
        *)                 echo "codegen" ;;
    esac
}

wait_ready() {
    local max_wait=900   # MLX model load + chunked prefill warmup
    local start; start=$(date +%s)
    while true; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "  server ready in $(( $(date +%s) - start ))s"
            return 0
        fi
        if [ $(( $(date +%s) - start )) -gt "$max_wait" ]; then
            echo "  TIMEOUT after ${max_wait}s"
            return 1
        fi
        sleep 4
    done
}

stop_server() {
    # SGLang spawns several worker processes: the parent (launch_server),
    # one or more sglang::scheduler workers, sglang::detokenizer workers,
    # and multiprocessing.resource_tracker shims. The first two pkill
    # patterns only catch the parents; if we don't also kill the worker
    # processes by their renamed setproctitle names, they orphan, hold
    # the model weights resident, and accumulate memory pressure across
    # successive model swaps. The 2026-05-16 audit found 4 zombie
    # schedulers from prior sweeps holding ~14 GB RSS.
    pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
    pkill -KILL -f "scripts/launch.sh" 2>/dev/null || true
    pkill -KILL -f "sglang::scheduler" 2>/dev/null || true
    pkill -KILL -f "sglang::detokenizer" 2>/dev/null || true
    pkill -KILL -f "multiprocessing.resource_tracker" 2>/dev/null || true
    sleep 4
    for _ in $(seq 1 20); do
        if ! curl -sf -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    sleep 3
}

run_one() {
    local preset="$1"
    local logfile="$LOG_DIR/${preset}.log"
    local result_json="$RESULTS_DIR/${preset}.json"
    local probes
    probes=$(probes_for "$preset")

    echo
    echo "=================================================="
    echo "Probe trio: $preset"
    echo "  probes: $probes"
    echo "=================================================="

    stop_server

    # Detached launch so a single failing model doesn't break the orchestrator.
    # macOS lacks setsid — use nohup + disown for the same effect (PPID=1 after
    # the calling shell exits).
    nohup bash "$REPO_DIR/scripts/launch.sh" "$preset" \
        > "$logfile" 2>&1 < /dev/null &
    local launch_pid=$!
    disown
    echo "  launched (PID=$launch_pid, log: $logfile)"

    if ! wait_ready; then
        echo "FAIL: server didn't become ready. Last 30 log lines:"
        tail -30 "$logfile"
        stop_server
        # Still write a result JSON so the sweep is auditable
        python3 -c "
import json, time
json.dump({
    'preset': '$preset',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'server': 'failed_to_boot',
}, open('$result_json', 'w'), indent=2)
"
        return 1
    fi

    # Run each applicable probe and capture verdict text from the log
    for probe in $probes; do
        echo
        echo "  >>> probe_${probe} <<<"
        local probe_log="$LOG_DIR/${preset}-${probe}.log"
        set +e
        python "$REPO_DIR/scripts/eval/probe_${probe}.py" --port "$PORT" > "$probe_log" 2>&1
        local probe_rc=$?
        set -e
        local verdict
        verdict=$(grep -E "^VERDICT:|^THINKING (VERIFIED|DEGRADED)" "$probe_log" | tail -1 || echo "")
        echo "$verdict (rc=$probe_rc)  log=$probe_log"
    done

    # Persist the cell result as JSON — re-greps the per-probe log files
    LOG_DIR_SHELL="$LOG_DIR" RESULT_JSON="$result_json" \
    python3 - "$preset" "$probes" <<'PY'
import json, os, sys, time
preset = sys.argv[1]
probes = sys.argv[2].split()
log_dir = os.environ["LOG_DIR_SHELL"]
out_path = os.environ["RESULT_JSON"]
results = {"preset": preset, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "probes": {}}
for probe in probes:
    log = f"{log_dir}/{preset}-{probe}.log"
    verdict = None
    try:
        with open(log) as fh:
            for line in fh:
                if line.startswith("VERDICT:"):
                    verdict = line.split(":", 1)[1].strip()
                elif line.startswith("THINKING VERIFIED") or line.startswith("THINKING DEGRADED"):
                    verdict = line.strip()
    except FileNotFoundError:
        pass
    results["probes"][probe] = {"verdict": verdict, "log": log}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"  wrote {out_path}")
PY

    stop_server
}

PRESETS_ARG="$*"
if [ -n "${PRESETS:-}" ]; then
    PRESETS_LIST="$PRESETS"
elif [ -n "$PRESETS_ARG" ]; then
    PRESETS_LIST="$PRESETS_ARG"
else
    PRESETS_LIST="coder-30b devstral gemma4 gemma4-31b qwen3-moe qwen3-32b qwen35-9b-8bit qwen35 qwen36-27b qwen36"
fi

for preset in $PRESETS_LIST; do
    run_one "$preset"
done

stop_server

echo
echo "=================================================="
echo "Probe trio sweep complete."
echo "Per-preset JSONs: $RESULTS_DIR/"
echo "=================================================="
ls -la "$RESULTS_DIR/" || true
