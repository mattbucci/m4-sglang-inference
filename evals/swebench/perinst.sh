#!/bin/bash
# Per-instance SWE-bench Lite sweep — fresh server boot per instance.
#
# This is the M4 sweep workflow. At CTX=131K, multi-instance single-server
# sweeps hit recurring macOS jetsam (the SGLang scheduler gets reaped
# silently after ~10 minutes of accumulated state). Per-instance restart
# fully sidesteps that floor: each instance gets a clean ~55GB-free server,
# state doesn't accumulate across instances.
#
# Trade-off: adds ~30s/instance for server boot. For N=5 sweep:
#   single-server: 5 instances at 10min wall = 50min, but jetsam likely
#   per-instance: 5 instances at 11min wall = 55min, jetsam-immune
#
# Usage:
#   bash evals/swebench/perinst.sh INSTANCE1 INSTANCE2 ...
#   bash evals/swebench/perinst.sh \
#       pallets__flask-4045 \
#       psf__requests-1963 \
#       pydata__xarray-3364
#
# Env overrides:
#   PRESET       launch.sh preset (default: qwen36)
#   TIMEOUT      per-instance opencode timeout (default: 900)
#   OUT          combined output dir (default: /tmp/qwen36-perinst-<date>)
#   SLEEP_SECS   pause between instances for OS memory reclaim (default: 15)
#   SCORE        score after rollouts? (default: 1; 0 = skip scoring)
#
# Output layout:
#   $OUT/<eco>/                  per-instance smoke.sh OUT dir
#   $OUT/predictions.jsonl       merged across all instances
#   $OUT/scores.jsonl            if SCORE=1
#
# Example complete run on the 4 missing M4-scorable ecosystems:
#   PRESET=qwen36 TIMEOUT=900 bash evals/swebench/perinst.sh \
#       pallets__flask-4045 \
#       psf__requests-1963 \
#       pydata__xarray-3364 \
#       pytest-dev__pytest-11143
#
# This workflow unlocked the cross-ecosystem coverage gap on 2026-05-18:
# the previous multi-instance sweep had hit jetsam at instance 2/5,
# contaminating 4 of 5 results with fake "model failures" (server was
# already dead when those instances ran). Per-instance pattern: 3/4
# real patches captured, real-ecosystem confirmation.

set -euo pipefail
SWE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SWE_SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"
activate_venv

if [ $# -eq 0 ]; then
    echo "Usage: $0 INSTANCE1 INSTANCE2 ..."
    echo "       $0 pallets__flask-4045 psf__requests-1963 pydata__xarray-3364"
    exit 1
fi

PRESET="${PRESET:-qwen36}"
TIMEOUT="${TIMEOUT:-900}"
SLEEP_SECS="${SLEEP_SECS:-15}"
SCORE="${SCORE:-1}"
OUT="${OUT:-/tmp/${PRESET}-perinst-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT"
echo "============================================================"
echo "perinst sweep: $PRESET, $# instances, OUT=$OUT"
echo "  TIMEOUT=$TIMEOUT  SLEEP_SECS=$SLEEP_SECS  SCORE=$SCORE"
echo "============================================================"

for inst in "$@"; do
    short=$(echo "$inst" | sed 's/__.*//')
    inst_out="$OUT/$short"
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] $inst -> $inst_out"
    echo "============================================================"
    rm -rf "$inst_out"
    PRESET="$PRESET" INSTANCE_IDS="$inst" TIMEOUT="$TIMEOUT" OUT="$inst_out" \
        bash "$SWE_SCRIPT_DIR/smoke.sh" > "$inst_out.run.log" 2>&1 || true

    # Report result
    if [ -f "$inst_out/predictions.jsonl" ]; then
        python3 -c "
import json
r = json.loads(open('$inst_out/predictions.jsonl').readline())
b = len(r['model_patch']); w = r.get('rollout_seconds',0); rc = r.get('rollout_returncode','?')
dead = r.get('server_dead', False)
print(f'  result: {b}B  {w:.0f}s  rc={rc}{\"  [DEAD]\" if dead else \"\"}')"
    else
        echo "  result: NO PREDICTIONS WRITTEN"
    fi

    # OS memory reclaim
    sleep "$SLEEP_SECS"
done

echo "============================================================"
echo "[$(date +%H:%M:%S)] ALL ROLLOUTS DONE"
echo "============================================================"

# Merge predictions
cat "$OUT"/*/predictions.jsonl > "$OUT/predictions.jsonl" 2>/dev/null || true
N_PREDS=$(wc -l < "$OUT/predictions.jsonl" 2>/dev/null || echo 0)
echo "merged $N_PREDS predictions -> $OUT/predictions.jsonl"

# Optional: score
if [ "$SCORE" = "1" ] && [ "$N_PREDS" -gt 0 ]; then
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] SCORING"
    echo "============================================================"
    python3 "$SWE_SCRIPT_DIR/score_local.py" \
        --predictions "$OUT/predictions.jsonl" \
        --workdir "$OUT/.score-work" \
        --venvdir "$OUT/.score-venvs" \
        --out "$OUT/scores.jsonl" \
        --timeout 600 2>&1 | tail -15
fi

echo "============================================================"
echo "[$(date +%H:%M:%S)] DONE: $OUT"
echo "============================================================"
