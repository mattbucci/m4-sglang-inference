#!/bin/bash
# SWE-bench Lite rollout orchestrator on M4 SGLang+MLX: boots the preset,
# runs the scaffold (omp by default) over N instances, tears down.
#
# Mirrors the 3090 sister repo's evals/swebench/bake_off.sh pattern but
# scoped to one preset per invocation.
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
#   SCAFFOLD         coding-agent scaffold: omp (default; evals/swebench/bin/omp,
#                    installed by vendor_omp.sh) or opencode (the archived cells)
#   MODEL_KEY        model id recorded in predictions (sglang/$MODEL_KEY); for
#                    opencode it must be a key in opencode.json (default: $PRESET)
#   OUTPUT_BUDGET    per-completion output cap handed to omp (default: 8192,
#                    the opencode.json limit.output the archived cells ran with)
#   OMP_THINKING     omp --thinking level (default: off; the proxy forces
#                    enable_thinking=false anyway when NO_THINKING_PROXY=1)
#   INSTANCES        how many SWE-bench Lite instances to run (default: 1)
#   TIMEOUT          per-instance opencode timeout in seconds (default: 900;
#                    bumped from 600 on 2026-05-21 — today's qwen36 sessions
#                    average ~30 tool calls per instance vs ~6-10 in May 18,
#                    making 600s frequently insufficient; see
#                    evals/swebench/runs/qwen36-django10914-N3-STABLE-RESOLVED-2026-05-21/perf-investigation.md)
#   CTX              context length passed to launch.sh (default: 131072)
#   EXTRA_LAUNCH     extra args to launch.sh (default: long-context recipe)
#   OUT              output directory for predictions/logs (default: /tmp/swebench-smoke)
#   PROMPT_EXTRA     text appended to every task prompt (passed to run_rollouts.py
#                    --prompt-extra); use for model-specific tool-call rules
#   BACKFILL_BASH_DESCRIPTION=1
#                    make the proxy add opencode's required bash `description`
#                    argument to tool calls that omit it (Flint; see the
#                    proxy's docstring)
#   BACKFILL_INTENT=1
#                    the omp form of the same repair: add omp's required `i`
#                    (intent) argument to tool calls that omit it (Flint)
#   REMAP_INTENT=1   with BACKFILL_INTENT: move a payload sent as `i` to the
#                    tool's own argument and clamp `i` to omp's 200 chars
#   INTENT_LAST=1    move `i` and the other metadata keys last in every omp
#                    tool schema on the request side (default 1 for
#                    PRESET=flint, 0 otherwise); omp runs calls without `i`,
#                    so this replaces both response-side repairs for Flint
#   PROXY_UPSTREAM   where the harness proxy forwards (default: the server);
#                    point it at a recording proxy to capture every request
#   DISABLE_TOOLS    comma-separated scaffold tools to switch off for the run
#                    (task and the web tools are always off), e.g.
#                    "edit,write,read,grep,glob,eval" to make the agent
#                    bash-only like Flint's native harnesses
#   PROMPT_STYLE     task prompt variant passed to run_rollouts.py --prompt-style
#                    (default | bash); use bash with a bash-only DISABLE_TOOLS set
#   SWEBENCH_SANDBOX=0
#                    lift the localhost-only network seal around opencode
#                    (run_rollouts.py wraps it in sandbox-exec by default)

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
SCAFFOLD="${SCAFFOLD:-omp}"
MODEL_KEY="${MODEL_KEY:-$PRESET}"
OUTPUT_BUDGET="${OUTPUT_BUDGET:-8192}"
OMP_THINKING="${OMP_THINKING:-off}"
INSTANCES="${INSTANCES:-1}"
# Optional space-separated list of specific instance IDs (overrides INSTANCES).
INSTANCE_IDS="${INSTANCE_IDS:-}"
TIMEOUT="${TIMEOUT:-900}"
CTX="${CTX:-131072}"
PORT=23334
PROXY_PORT="${PROXY_PORT:-23335}"
OUT="${OUT:-/tmp/swebench-smoke}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac   # the rollout runs from evals/swebench
PROMPT_EXTRA="${PROMPT_EXTRA:-}"
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
BACKFILL_BASH_DESCRIPTION="${BACKFILL_BASH_DESCRIPTION:-0}"
BACKFILL_INTENT="${BACKFILL_INTENT:-0}"
REMAP_INTENT="${REMAP_INTENT:-0}"
if [ "$PRESET" = flint ]; then INTENT_LAST="${INTENT_LAST:-1}"; else INTENT_LAST="${INTENT_LAST:-0}"; fi
DISABLE_TOOLS="${DISABLE_TOOLS:-}"
PROMPT_STYLE="${PROMPT_STYLE:-default}"

mkdir -p "$OUT"
LOG="$OUT/launch.log"

echo "============================================================"
echo "SWE-bench Lite smoke: $PRESET → $SCAFFOLD model sglang/$MODEL_KEY"
echo "  instances:   $INSTANCES"
echo "  timeout:     ${TIMEOUT}s/instance"
echo "  context:     $CTX"
echo "  extra args:  $EXTRA_LAUNCH"
echo "  out:         $OUT"
echo "============================================================"

if [ "$SCAFFOLD" = "omp" ] && [ ! -x "$SWE_SCRIPT_DIR/bin/omp" ]; then
    echo "[$(date +%H:%M:%S)] Installing the pinned omp binary..."
    bash "$SWE_SCRIPT_DIR/vendor_omp.sh"
fi

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
if [ "$SERVED" != "$PRESET" ] && [ "$SCAFFOLD" = "opencode" ]; then
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
    upstream_url="${PROXY_UPSTREAM:-http://127.0.0.1:$PORT}"
    PORT="$PROXY_PORT" UPSTREAM="$upstream_url" \
        BACKFILL_BASH_DESCRIPTION="$BACKFILL_BASH_DESCRIPTION" BACKFILL_INTENT="$BACKFILL_INTENT" \
        REMAP_INTENT="$REMAP_INTENT" INTENT_LAST="$INTENT_LAST" \
        nohup python3 "$SWE_SCRIPT_DIR/no_thinking_proxy.py" > "$OUT/proxy.log" 2>&1 &
    PROXY_PID=$!
    disown
    # Wait for proxy to accept connections
    for _ in $(seq 1 20); do
        if curl -sf "http://127.0.0.1:$PROXY_PORT/health" > /dev/null 2>&1; then break; fi
        sleep 0.5
    done
    MODEL_URL="http://127.0.0.1:$PROXY_PORT"
fi
if [ "$NO_THINKING_PROXY" = "1" ] && [ "$SCAFFOLD" = "opencode" ]; then
    # Repoint opencode at the proxy + disable the `task` sub-agent tool and
    # the web tools (`webfetch`/`websearch`): a session on astropy-12907
    # fetched the post-fix separable.py from GitHub through webfetch. The
    # process-level seal is run_rollouts.py's sandbox (SWEBENCH_SANDBOX).
    # 2026-05-21 finding: qwen36 on django-11001 invokes the `task` tool to
    # spawn a sub-agent that runs many chat completions but never produces
    # an edit within reasonable TIMEOUT. Disabling `task` for SWE-bench
    # rollouts forces direct read/grep/edit work. Verified: with task
    # disabled, django-11001 finishes in 1204s with RESOLVED (F2P 2/2,
    # P2P 118/118); with task enabled it hits even TIMEOUT=1800 with 0B
    # patch. Backup/restore around the rollout keeps the user's
    # interactive opencode config untouched.
    cp "$OPENCODE_CFG" "$OPENCODE_CFG_BACKUP"
    DISABLE_TOOLS="$DISABLE_TOOLS" python3 -c "
import json, os, re, sys
src = open('$OPENCODE_CFG').read()
src = src.replace('http://127.0.0.1:$PORT/v1', 'http://127.0.0.1:$PROXY_PORT/v1')
# Inject a \"tools\" block (task off, plus DISABLE_TOOLS) after the \$schema line if not already present.
off = ['task', 'webfetch', 'websearch'] + [t.strip() for t in os.environ.get('DISABLE_TOOLS', '').split(',') if t.strip()]
block = '\n  \"tools\": {\n' + ',\n'.join('    \"%s\": false' % t for t in off) + '\n  },'
if '\"webfetch\": false' not in src and '\"task\": false' in src:
    # A tools block from an earlier run survived (restore did not happen): extend it.
    src = src.replace('\"task\": false', '\"task\": false,\n    \"webfetch\": false,\n    \"websearch\": false', 1)
if '\"webfetch\": false' not in src:
    src = re.sub(
        r'(\"\\\$schema\":\\s*\"[^\"]+\",)',
        lambda m: m.group(1) + block,
        src,
        count=1,
    )
open('$OPENCODE_CFG', 'w').write(src)
"
    echo "[$(date +%H:%M:%S)] Proxy ready (PID=$PROXY_PID), opencode config repointed to :$PROXY_PORT"
fi
MODEL_URL="${MODEL_URL:-http://127.0.0.1:$PORT}"

# --- Run the scaffold rollout (1 instance by default) ---
echo "[$(date +%H:%M:%S)] Running SWE-bench Lite rollout ($SCAFFOLD → $MODEL_URL)..."

# opencode resolves its project config from --dir (the task worktree), not
# from the cwd, so the rollout uses the global ~/.config/opencode/opencode.jsonc
# (the copy of evals/swebench/opencode.json installed by setup). cwd is only
# for run_rollouts.py's sibling imports.
cd "$SWE_SCRIPT_DIR"
set +e
# Optional: skip the tool-call-round-trip canary (some chat templates,
# notably Mistral/Devstral, reject the canary's strict user→assistant
# (tool_calls)→tool→user sequence even when the model itself can do
# agentic flows fine).
preflight_arg=""
if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
    preflight_arg="--skip-preflight"
fi
# ${arr[@]+"${arr[@]}"}: bash 3.2 (macOS /bin/bash) treats an empty array as
# unbound under set -u.
extra_args=(--scaffold "$SCAFFOLD")
if [ "$SCAFFOLD" = "omp" ]; then
    # omp reads its provider from the per-run agent dir run_rollouts.py writes
    # (the proxy URL when the proxy is on); contextWindow must match CTX.
    extra_args+=(--model-url "$MODEL_URL" --context-window "$CTX"
                 --output-budget "$OUTPUT_BUDGET" --thinking-level "$OMP_THINKING")
    if [ -n "$DISABLE_TOOLS" ]; then
        extra_args+=(--disable-tools "$DISABLE_TOOLS")
    fi
fi
if [ -n "$PROMPT_EXTRA" ]; then
    extra_args+=(--prompt-extra "$PROMPT_EXTRA")
fi
if [ "$PROMPT_STYLE" != "default" ]; then
    extra_args+=(--prompt-style "$PROMPT_STYLE")
fi
if [ -n "$INSTANCE_IDS" ]; then
    python3 run_rollouts.py \
        --model "sglang/$MODEL_KEY" \
        --instance-ids $INSTANCE_IDS \
        --timeout "$TIMEOUT" \
        --out "$OUT" \
        --server-url "http://127.0.0.1:$PORT" \
        --served-name "$PRESET" \
        $preflight_arg ${extra_args[@]+"${extra_args[@]}"} 2>&1 | tee "$OUT/rollout.log"
else
    python3 run_rollouts.py \
        --model "sglang/$MODEL_KEY" \
        --instances "$INSTANCES" \
        --timeout "$TIMEOUT" \
        --out "$OUT" \
        --server-url "http://127.0.0.1:$PORT" \
        --served-name "$PRESET" \
        $preflight_arg ${extra_args[@]+"${extra_args[@]}"} 2>&1 | tee "$OUT/rollout.log"
fi
RC=$?
set -e
cd "$REPO_DIR"

# --- Tear down proxy (if started) + restore opencode config ---
if [ -n "$PROXY_PID" ]; then
    echo "[$(date +%H:%M:%S)] Tearing down proxy..."
    kill "$PROXY_PID" 2>/dev/null || true
    pkill -KILL -f "no_thinking_proxy" 2>/dev/null || true
    if [ "$SCAFFOLD" = "opencode" ] && [ -f "$OPENCODE_CFG_BACKUP" ]; then
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
if [ "$SCAFFOLD" = "omp" ]; then
    echo "  Sessions:     $OUT/sessions/<instance_id>/*.jsonl"
fi

exit "$RC"
