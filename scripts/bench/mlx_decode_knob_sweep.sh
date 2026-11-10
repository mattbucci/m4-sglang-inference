#!/bin/bash
# Interleaved raw-MLX knob sweep: every pass runs every arm once in a fresh
# process, and even passes run the arms in reverse order.
# usage: mlx_decode_knob_sweep.sh <out jsonl> <depths csv> <passes> "TAG|ENV=VAL ENV=VAL" ...
#   e.g. mlx_decode_knob_sweep.sh sweep.jsonl 24,8192 2 "base|X=1" "m400|MLX_MAX_MB_PER_BUFFER=400"
set -u
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
OUT=$1; DEPTHS=$2; PASSES=$3; shift 3
ARMS=("$@"); N=${#ARMS[@]}
for p in $(seq 1 "$PASSES"); do
  for k in $(seq 0 $((N-1))); do
    if [ $((p % 2)) = 0 ]; then i=$((N-1-k)); else i=$k; fi
    arm="${ARMS[$i]}"; tag="${arm%%|*}"; envs="${arm#*|}"
    env SGLANG_USE_MLX=1 $envs "$PY" scripts/bench/mlx_decode_knobs.py "$tag" "$DEPTHS" "$OUT" 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-260
  done
done
echo KNOB_SWEEP_DONE
