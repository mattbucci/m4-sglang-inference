#!/bin/bash
# Sequential SWE-bench Lite bake-off across the 4 README "Recommended picks"
# on M4, all routed through no_thinking_proxy. One instance per model
# (astropy__astropy-12907 — the default first instance), then a comparison
# table.
#
# Usage:
#   bash evals/swebench/bakeoff.sh                    # all 4 picks
#   PRESETS="qwen35 gemma4-31b" bash evals/swebench/bakeoff.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

PRESETS="${PRESETS:-coder-30b qwen36 qwen35 gemma4-31b}"
INSTANCES="${INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-600}"
BAKEOFF_OUT="${BAKEOFF_OUT:-/tmp/swebench-bakeoff-$(date +%Y%m%d-%H%M)}"
mkdir -p "$BAKEOFF_OUT"

echo "============================================================"
echo "SWE-bench Lite bake-off via no_thinking_proxy"
echo "  presets:   $PRESETS"
echo "  instances: $INSTANCES per preset"
echo "  out:       $BAKEOFF_OUT"
echo "============================================================"

for preset in $PRESETS; do
    out_dir="$BAKEOFF_OUT/$preset"
    mkdir -p "$out_dir"
    echo
    echo "──────────── $preset ────────────"
    # Per-preset launch knobs. Qwen3.x family needs --tool-call-parser
    # qwen3_coder; gemma4 family is text-only on M4 (no tool_call=true in
    # opencode.json yet — opencode might not surface tool calls at all on
    # the gemma4 preset). Run anyway and measure.
    case "$preset" in
        coder-30b)
            ctx=131072
            launch="--kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5"
            ;;
        qwen36)
            ctx=32768
            launch="--disable-radix-cache --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5 --enable-multimodal --tool-call-parser qwen3_coder"
            ;;
        qwen35|qwen35-9b-8bit)
            ctx=32768
            launch="--disable-radix-cache --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5 --enable-multimodal --tool-call-parser qwen3_coder"
            ;;
        gemma4-31b)
            ctx=16384
            launch="--disable-radix-cache --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5"
            ;;
        *)
            ctx=32768
            launch="--disable-radix-cache --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5"
            ;;
    esac

    OUT="$out_dir" PRESET="$preset" MODEL_KEY="$preset" CTX="$ctx" \
        EXTRA_LAUNCH="$launch" INSTANCES="$INSTANCES" TIMEOUT="$TIMEOUT" \
        NO_THINKING_PROXY=1 \
        bash "$SCRIPT_DIR/smoke.sh"
    rc=$?
    echo "  $preset rc=$rc"
done

# --- Comparison table ---
echo
echo "============================================================"
echo "BAKE-OFF RESULTS"
echo "============================================================"
python3 - <<PY
import json, os
out_dir = "$BAKEOFF_OUT"
print(f"{'preset':<14} {'rc':<3} {'wall_s':<8} {'tool_calls':<12} {'edit':<6} {'diff_B':<8}")
print("-" * 60)
for preset in os.listdir(out_dir):
    pj = os.path.join(out_dir, preset, "predictions.jsonl")
    log_dir = os.path.join(out_dir, preset, "logs")
    if not os.path.exists(pj):
        print(f"{preset:<14} NO_PREDICTIONS")
        continue
    with open(pj) as fh:
        for line in fh:
            r = json.loads(line)
            inst = r.get("instance_id", "?")
            rc = r.get("rollout_returncode", "?")
            wall = f"{r.get('rollout_seconds', 0):.1f}"
            patch = r.get("model_patch", "")
            diff_B = len(patch)
            log_path = os.path.join(log_dir, f"{inst}.log")
            tool_count = 0
            edit_count = 0
            if os.path.exists(log_path):
                try:
                    with open(log_path) as lf:
                        for line in lf:
                            if '"type":"tool_use"' in line:
                                tool_count += 1
                                if '"tool":"edit"' in line:
                                    edit_count += 1
                except Exception:
                    pass
            print(f"{preset:<14} {rc!s:<3} {wall:<8} {tool_count:<12} {edit_count:<6} {diff_B:<8}")
PY

echo
echo "Per-preset artifacts: $BAKEOFF_OUT/<preset>/{predictions.jsonl, predictions/<inst>.diff, logs/<inst>.log}"
