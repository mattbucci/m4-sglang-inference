#!/bin/bash
# Depth-verified throughput regression tripwire for the M4 Pro MLX rig.
#
# Instrument: scripts/bench/bench_all_unified.py --context-list (fleet
# invariants: single-user, --random-range-ratio 1, actual_input_tokens
# recorded per point, degenerate points rejected — dead-server partial
# streams and zero-output rejections never become rows).
# Three depths per preset: 1024, 8192, 32768 (the M4 ceiling depths).
#
# Baselines: benchmarks/baselines.json, fleet-standard SCHEMA v2 —
#   { "_meta": {schema:2, instrument, stack, hardware, output_tokens, saved},
#     "<preset>": { "1024": {tok_per_sec,tpot_ms,ttft_ms,actual_input_tokens},
#                   "8192": {...}, "32768": {...} } }
# Gate: tok_per_sec drop >THRESHOLD% (default 10) at any depth => REGRESSION,
# exit 1. ttft_ms drift is reported WARN-only. Points with depth_shortfall /
# error / actual <95% of label are never saved and never compared — listed
# NOT-COMPARED so a silently missing deep point can't masquerade as a PASS.
#
# Usage:
#   scripts/bench/bench_regression.sh <preset>            # bench live server
#                                                         #   (must be serving
#                                                         #   <preset>), compare
#   BASELINE=save scripts/bench/bench_regression.sh <preset>   # save instead
#   scripts/bench/bench_regression.sh arm [preset...]     # serve each preset
#                                                         #   itself (tripwire
#                                                         #   recipe), bench,
#                                                         #   stop, next
#   scripts/bench/bench_regression.sh check <preset> <run.json>  # compare/save
#                                                         #   from an existing
#                                                         #   results.json (no
#                                                         #   serving)
#
# Ops rules:
#   - The 32768-labeled point does not fit the as-shipped presets (CTX 32768
#     rejects 32768+64, and radix-on serving OOMs the genuine 32K prefill —
#     benchmarks/coder-30b-4bit/results.json receipts). Arm/bench serve with
#     the documented long-context tripwire recipe: CTX=36000 MEM_FRAC=0.45
#     SGLANG_MLX_CACHE_LIMIT_GB=2 --disable-radix-cache --kv-cache turboquant.
#   - oom_guard.sh must be running for any arm/bench (32K point). Refuses to
#     start without it.
#   - BASELINE=save is a DELIBERATE act (after a receipted WIN or verified
#     flip). Save merges per-preset AND per-depth: a flagged rerun can't drop
#     a previously-armed point.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYBENCH="$SCRIPT_DIR/bench_all_unified.py"
source "$REPO_DIR/scripts/common.sh"
activate_venv
setup_mlx_env
[ -f "$PYBENCH" ] || { echo "FATAL: instrument missing at $PYBENCH"; exit 2; }

BASELINES="${BASELINES:-$REPO_DIR/benchmarks/baselines.json}"
RUNS_DIR="$REPO_DIR/benchmarks/regression"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:$PORT"
THRESHOLD="${THRESHOLD:-10}"
SAVE_BASELINE="${BASELINE:-}"
DEPTHS="1024,8192,32768"
STACK_TAG="${STACK_TAG:-sglang-v0.5.15.post1}"

# One preset per distinct M4 arch path: MoE (coder-30b), MoE-DWQ (qwen3-moe),
# dense Mistral3+VL (devstral), MoE+DeltaNet hybrid+VL (qwen36).
TRIPWIRE_PRESETS=(coder-30b qwen3-moe devstral qwen36)

mkdir -p "$RUNS_DIR"

require_oom_guard() {
    pgrep -f "oom_guard.sh" > /dev/null 2>&1 || {
        echo "FATAL: oom_guard.sh not running (mandatory for the 32K point)."
        echo "Start it first: bash scripts/common/oom_guard.sh &"
        exit 2
    }
}

wait_health() {
    for _ in $(seq 1 90); do
        curl -s -m 3 "$BASE_URL/health" > /dev/null 2>&1 && return 0
        sleep 10
    done
    return 1
}

run_instrument() {
    local preset="$1" out_json="$2"
    SGLANG_USE_MLX=1 python3 "$PYBENCH" --port "$PORT" --name "$preset" \
        --context-list "$DEPTHS" --skip-concurrency --skip-charts \
        --output "$out_json"
}

# compare/save one preset from a run JSON. Args: preset run_json save_flag
check_json() {
    python3 - "$1" "$2" "${3:-}" <<'PYEOF'
import json, os, sys, time

preset, run_path, save = sys.argv[1], sys.argv[2], sys.argv[3]
BASELINES = os.environ["BASELINES_PATH"]
THRESHOLD = float(os.environ.get("THRESHOLD", "10"))
DEPTHS = ["1024", "8192", "32768"]

with open(run_path) as f:
    run = json.load(f)

points = {}
for pt in run.get("context_sweep", []):
    label = str(pt.get("context"))
    if label not in DEPTHS:
        continue
    actual = pt.get("actual_input_tokens")
    flagged = ("error" in pt or pt.get("depth_shortfall")
               or actual is None or actual < 0.95 * pt["context"])
    points[label] = None if flagged else {
        "tok_per_sec": pt["tok_per_sec"], "tpot_ms": pt["tpot_ms"],
        "ttft_ms": pt["ttft_ms"], "actual_input_tokens": actual,
    }

baselines = {}
if os.path.exists(BASELINES):
    with open(BASELINES) as f:
        baselines = json.load(f)

if save:
    base = baselines.setdefault(preset, {})
    saved = 0
    for label in DEPTHS:
        if points.get(label):          # merge per-depth; flagged never saved
            base[label] = points[label]
            saved += 1
        else:
            print(f"  {label}: NOT-SAVED (missing or flagged)")
    baselines["_meta"] = {
        "schema": 2,
        "instrument": "scripts/bench/bench_all_unified.py --context-list",
        "stack": run.get("sglang_version", os.environ.get("STACK_TAG", "")),
        "hardware": "Apple M4 Pro 64GB",
        "output_tokens": 64,
        "serving_recipe": ("CTX=36000 MEM_FRAC=0.45 SGLANG_MLX_CACHE_LIMIT_GB=2 "
                           "--disable-radix-cache --kv-cache turboquant"),
        "saved": time.strftime("%Y-%m-%d"),
    }
    with open(BASELINES, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
    print(f"  saved {saved}/{len(DEPTHS)} depths for {preset} -> {BASELINES}")
    sys.exit(0 if saved else 1)

base = baselines.get(preset)
if not isinstance(base, dict):
    print(f"  no baseline for {preset} — run with BASELINE=save to arm")
    sys.exit(2)

failed = False
for label in DEPTHS:
    b, c = base.get(label), points.get(label)
    if not b:
        print(f"  {label}: no armed baseline (NOT-COMPARED)")
        continue
    if not c:
        print(f"  {label}: current point missing/flagged (NOT-COMPARED)")
        failed = True     # an armed depth that can't be measured is a failure
        continue
    pct = (c["tok_per_sec"] - b["tok_per_sec"]) / b["tok_per_sec"] * 100
    verdict = "REGRESSION" if pct < -THRESHOLD else "ok"
    if pct < -THRESHOLD:
        failed = True
    print(f"  {label}: tok/s {b['tok_per_sec']:.1f} -> {c['tok_per_sec']:.1f} "
          f"({pct:+.1f}%) [{verdict}]")
    tdrift = (c["ttft_ms"] - b["ttft_ms"]) / b["ttft_ms"] * 100 if b["ttft_ms"] else 0
    if abs(tdrift) > THRESHOLD:
        print(f"    WARN ttft_ms {b['ttft_ms']:.0f} -> {c['ttft_ms']:.0f} "
              f"({tdrift:+.1f}%) — warn-only")
sys.exit(1 if failed else 0)
PYEOF
}
export BASELINES_PATH="$BASELINES" THRESHOLD STACK_TAG

server_served_name() {
    curl -s -m 5 "$BASE_URL/v1/models" | python3 -c \
        "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null
}

MODE="${1:?Usage: $0 <preset> | arm [preset...] | check <preset> <run.json>}"

if [ "$MODE" = "check" ]; then
    PRESET="${2:?check needs <preset> <run.json>}"
    RUN_JSON="${3:?check needs <preset> <run.json>}"
    echo "=== $PRESET (offline check: $RUN_JSON) ==="
    check_json "$PRESET" "$RUN_JSON" "$SAVE_BASELINE"
    exit $?
fi

if [ "$MODE" = "arm" ]; then
    shift
    if [ $# -gt 0 ]; then PRESETS=("$@"); else PRESETS=("${TRIPWIRE_PRESETS[@]}"); fi
    require_oom_guard
    FAILED=0
    for preset in "${PRESETS[@]}"; do
        echo ""
        echo "=== arming $preset ==="
        pkill -f "sglang.launch_server" 2>/dev/null; sleep 5
        CTX=36000 MEM_FRAC=0.45 SGLANG_MLX_CACHE_LIMIT_GB=2 \
            EXTRA_ARGS="--disable-radix-cache" \
            bash "$REPO_DIR/scripts/launch.sh" "$preset" --kv-cache turboquant \
            > "$RUNS_DIR/${preset}-serve.log" 2>&1 &
        if ! wait_health; then
            echo "  FAILED: $preset never became healthy (NOT-ARMED)"
            FAILED=1
            continue
        fi
        RUN_JSON="$RUNS_DIR/${preset}-$(date +%Y%m%dT%H%M%S).json"
        run_instrument "$preset" "$RUN_JSON"
        BASELINE=save check_json "$preset" "$RUN_JSON" save || FAILED=1
        pkill -f "sglang.launch_server" 2>/dev/null; sleep 5
    done
    exit $FAILED
fi

# default: bench the live server for <preset>, compare (or save)
PRESET="$MODE"
require_oom_guard
SERVED="$(server_served_name)"
if [ -z "$SERVED" ]; then
    echo "FATAL: no server responding on $BASE_URL"
    exit 2
fi
if [ "$SERVED" != "$PRESET" ]; then
    echo "FATAL: server is serving '$SERVED', not '$PRESET'"
    exit 2
fi
RUN_JSON="$RUNS_DIR/${PRESET}-$(date +%Y%m%dT%H%M%S).json"
RECALL_FAIL=0
echo "=== $PRESET (live bench @ $DEPTHS) ==="
run_instrument "$PRESET" "$RUN_JSON"

# Standing 32K recall tripwire (quality field, schema v2): when the baseline
# stores recall_32k for this preset, re-run the seeded multi-needle probe at
# 32768 and require score >= stored score - 1. Depth is server-verified by
# the probe itself; same seed => byte-identical prompt (controlled A/B).
RECALL_BASE=$(python3 -c "
import json, sys
b = json.load(open('$BASELINES')) if __import__('os').path.exists('$BASELINES') else {}
q = b.get('$PRESET', {}).get('recall_32k')
print(f\"{q['score']} {q['seed']} {q['label']}\" if q else '')" 2>/dev/null)
if [ -n "$RECALL_BASE" ] && [ -z "$SAVE_BASELINE" ]; then
    read -r R_SCORE R_SEED R_LABEL <<< "$RECALL_BASE"
    echo "--- recall_32k tripwire (baseline ${R_SCORE}/6, seed $R_SEED, label $R_LABEL) ---"
    R_OUT=$(SGLANG_USE_MLX=1 python3 "$REPO_DIR/scripts/eval/probe_depth_recall.py" \
        --port "$PORT" --labels "$R_LABEL" --seed "$R_SEED" --tag tripwire 2>&1)
    echo "$R_OUT" | tail -2
    R_NOW=$(echo "$R_OUT" | grep -oE "score=[0-9]+" | head -1 | cut -d= -f2)
    if [ -z "$R_NOW" ] || [ "$R_NOW" -lt $((R_SCORE - 1)) ]; then
        echo "  recall_32k: ${R_NOW:-none} < $((R_SCORE - 1)) [REGRESSION]"
        RECALL_FAIL=1
    else
        echo "  recall_32k: ${R_NOW}/6 [ok]"
    fi
fi
echo "--- $([ -n "$SAVE_BASELINE" ] && echo saving baseline || echo comparing) ---"
check_json "$PRESET" "$RUN_JSON" "$SAVE_BASELINE"
RC=$?
if { [ $RC -eq 1 ] || [ "$RECALL_FAIL" = 1 ]; } && [ -z "$SAVE_BASELINE" ]; then
    echo ""
    echo "RESULT: REGRESSION DETECTED"
    exit 1
fi
[ -z "$SAVE_BASELINE" ] && echo "" && echo "RESULT: PASS"
exit $RC
