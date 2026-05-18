#!/bin/bash
# Single-instance SWE-bench Lite smoke against coder-30b on M4 SGLang+MLX.
#
# Mirrors the 3090 sister repo's evals/swebench/bake_off.sh pattern but
# scoped to one model (coder-30b — our recommended primary for agentic
# coding) and one instance (the first in SWE-bench Lite, deterministic).
#
# Phase 1 only (rollout). Phase 2 (Docker scoring) is mirrored from the
# 3090 score_local.py / score_docker.py; we don't run it here because
# the official SWE-bench harness needs Docker which we don't ship on M4.
# Use ../score_local.py if you have a local Python env that mimics the
# instance's runtime, or push predictions to the 3090 stack for scoring.
#
# Usage:
#   bash evals/swebench/smoke.sh                          # 1 instance, coder-30b
#   PRESET=qwen36 INSTANCES=3 bash evals/swebench/smoke.sh
#   MODEL_KEY=qwen36 INSTANCES=3 bash evals/swebench/smoke.sh
#
# Env overrides:
#   PRESET           launch.sh preset name (default: coder-30b)
#   MODEL_KEY        opencode model key in opencode.json (default: $PRESET)
#   INSTANCES        how many SWE-bench Lite instances to run (default: 1)
#   TIMEOUT          per-instance opencode timeout in seconds (default: 600)
#   CTX              context length passed to launch.sh (default: 131072)
#   EXTRA_LAUNCH     extra args to launch.sh (default: long-context recipe)
#   OUT              output directory for predictions/logs (default: /tmp/swebench-smoke)

set -euo pipefail
# Save our own dir before sourcing common.sh — it redefines SCRIPT_DIR.
SWE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SWE_SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"
# common.sh defines activate_venv but doesn't call it. Without an explicit
# call, `python3` inside this subshell resolves to system Python (which
# doesn't have the `swebench` HF dataset module — run_rollouts.py would
# crash with `ModuleNotFoundError`). launch.sh handles its own activation
# on line 360 of that script, so this is only needed for run_rollouts.py.
activate_venv

PRESET="${PRESET:-coder-30b}"
MODEL_KEY="${MODEL_KEY:-$PRESET}"
INSTANCES="${INSTANCES:-1}"
# Optional space-separated list of specific instance IDs (overrides INSTANCES).
INSTANCE_IDS="${INSTANCE_IDS:-}"
TIMEOUT="${TIMEOUT:-600}"
CTX="${CTX:-131072}"
PORT=23334
PROXY_PORT="${PROXY_PORT:-23335}"
OUT="${OUT:-/tmp/swebench-smoke}"
EXTRA_LAUNCH="${EXTRA_LAUNCH:---kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5}"
# NO_THINKING_PROXY=1 (default): start no_thinking_proxy.py on $PROXY_PORT
# and route opencode through it. The proxy injects
# chat_template_kwargs={"enable_thinking": false} on every chat-completion
# POST, which stops Qwen3-family models from emitting <think> blocks that
# either go to reasoning_content (invisible to opencode) or leak </think>
# markers into the content stream (breaks the agent loop). Verified
# 2026-05-18: this turned qwen36 from "9 tool calls, 0-byte diff" into
# "6 tool calls including 1 edit, 506-byte plausible patch" on
# astropy__astropy-12907. Set NO_THINKING_PROXY=0 to disable.
NO_THINKING_PROXY="${NO_THINKING_PROXY:-1}"

mkdir -p "$OUT"
LOG="$OUT/launch.log"

echo "============================================================"
echo "SWE-bench Lite smoke: $PRESET → opencode model sglang/$MODEL_KEY"
echo "  instances:   $INSTANCES"
echo "  timeout:     ${TIMEOUT}s/instance"
echo "  context:     $CTX"
echo "  extra args:  $EXTRA_LAUNCH"
echo "  out:         $OUT"
echo "============================================================"

# --- Stop any prior server cleanly ---
pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
pkill -KILL -f "scripts/launch.sh" 2>/dev/null || true
pkill -KILL -f "sglang::scheduler" 2>/dev/null || true
pkill -KILL -f "sglang::detokenizer" 2>/dev/null || true
sleep 3

# --- Launch model server ---
echo "[$(date +%H:%M:%S)] Launching $PRESET at CTX=$CTX..."
CTX="$CTX" EXTRA_ARGS="$EXTRA_LAUNCH" \
    bash scripts/launch.sh "$PRESET" > "$LOG" 2>&1 &
LAUNCH_PID=$!
disown

# Wait for /health with generous timeout (turboquant pool sizing + RoPE
# scaling at 131K can take 60+ seconds).
echo "[$(date +%H:%M:%S)] Waiting for server (PID=$LAUNCH_PID, log=$LOG)..."
WAIT=0
while [ $WAIT -lt 600 ]; do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Server ready after ${WAIT}s"
        break
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] FAIL: server died — last 30 log lines:"
        tail -30 "$LOG"
        exit 1
    fi
    sleep 5
    WAIT=$((WAIT + 5))
done

if [ $WAIT -ge 600 ]; then
    echo "[$(date +%H:%M:%S)] TIMEOUT waiting for server"
    pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
    exit 1
fi

# --- Verify served-model-name matches what opencode expects ---
SERVED=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[$(date +%H:%M:%S)] Server reports served-model-name='$SERVED'"
if [ "$SERVED" != "$PRESET" ]; then
    echo "WARN: opencode key '$MODEL_KEY' will not match served name '$SERVED' — set SERVED_NAME=$MODEL_KEY when launching."
fi

# --- Optionally start the no-thinking proxy + repoint opencode config ---
PROXY_PID=""
OPENCODE_CFG="$HOME/.config/opencode/opencode.jsonc"
OPENCODE_CFG_BACKUP="$HOME/.config/opencode/opencode.jsonc.smoke-backup"
if [ "$NO_THINKING_PROXY" = "1" ]; then
    echo "[$(date +%H:%M:%S)] Starting no_thinking_proxy on :$PROXY_PORT..."
    # Bash env-prefix scoping bites here — $PORT inside UPSTREAM resolves
    # incorrectly, so build the URL in a plain var first.
    upstream_url="http://127.0.0.1:$PORT"
    PORT="$PROXY_PORT" UPSTREAM="$upstream_url" \
        nohup python3 "$SWE_SCRIPT_DIR/no_thinking_proxy.py" > "$OUT/proxy.log" 2>&1 &
    PROXY_PID=$!
    disown
    # Wait for proxy to accept connections
    for _ in $(seq 1 20); do
        if curl -sf "http://127.0.0.1:$PROXY_PORT/health" > /dev/null 2>&1; then break; fi
        sleep 0.5
    done
    # Repoint opencode at the proxy
    cp "$OPENCODE_CFG" "$OPENCODE_CFG_BACKUP"
    python3 -c "
import json, sys
src = open('$OPENCODE_CFG').read()
open('$OPENCODE_CFG', 'w').write(src.replace('http://127.0.0.1:$PORT/v1', 'http://127.0.0.1:$PROXY_PORT/v1'))
"
    echo "[$(date +%H:%M:%S)] Proxy ready (PID=$PROXY_PID), opencode config repointed to :$PROXY_PORT"
fi

# --- Run opencode rollout (1 instance by default) ---
echo "[$(date +%H:%M:%S)] Running SWE-bench Lite rollout..."

# opencode reads ./opencode.json from the cwd OR --config-file. We have
# evals/swebench/opencode.json; the rollout driver doesn't take a config
# path so we run with cwd=evals/swebench.
cd "$SWE_SCRIPT_DIR"
set +e
if [ -n "$INSTANCE_IDS" ]; then
    python3 run_rollouts.py \
        --model "sglang/$MODEL_KEY" \
        --instance-ids $INSTANCE_IDS \
        --timeout "$TIMEOUT" \
        --out "$OUT" \
        --server-url "http://127.0.0.1:$PORT" \
        --served-name "$PRESET" 2>&1 | tee "$OUT/rollout.log"
else
    python3 run_rollouts.py \
        --model "sglang/$MODEL_KEY" \
        --instances "$INSTANCES" \
        --timeout "$TIMEOUT" \
        --out "$OUT" \
        --server-url "http://127.0.0.1:$PORT" \
        --served-name "$PRESET" 2>&1 | tee "$OUT/rollout.log"
fi
RC=$?
set -e
cd "$REPO_DIR"

# --- Tear down proxy (if started) + restore opencode config ---
if [ -n "$PROXY_PID" ]; then
    echo "[$(date +%H:%M:%S)] Tearing down proxy..."
    kill "$PROXY_PID" 2>/dev/null || true
    pkill -KILL -f "no_thinking_proxy" 2>/dev/null || true
    if [ -f "$OPENCODE_CFG_BACKUP" ]; then
        mv "$OPENCODE_CFG_BACKUP" "$OPENCODE_CFG"
    fi
fi

# --- Tear down server ---
echo "[$(date +%H:%M:%S)] Tearing down server..."
pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
pkill -KILL -f "sglang::scheduler" 2>/dev/null || true
pkill -KILL -f "sglang::detokenizer" 2>/dev/null || true
sleep 3

echo "[$(date +%H:%M:%S)] Smoke done (rc=$RC). Output: $OUT"
echo "  Predictions:  $OUT/predictions.jsonl"
echo "  Per-instance: $OUT/predictions/<instance_id>.diff"
echo "  Logs:         $OUT/logs/<instance_id>.log"

exit "$RC"
