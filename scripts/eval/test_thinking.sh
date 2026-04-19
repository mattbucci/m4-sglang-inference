#!/bin/bash
# Quick thinking-format sanity check across our thinking-capable presets.
# For each preset: launch server, send 3 prompts, look for <think> tags + clean
# answer extraction. Catches the Qwen3 family infinite-think-loop on greedy MLX.
#
# Adapted from R9700 sister repo.
#
# Usage: bash scripts/eval/test_thinking.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

PORT="${PORT:-23334}"

wait_for_server() {
    for i in $(seq 1 300); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

test_thinking() {
    local preset="$1"
    local tag="$2"

    echo ""
    echo "=== $tag ($preset) ==="

    pkill -9 -f sglang 2>/dev/null || true
    sleep 5

    # Same as run_all_evals.sh — disable radix cache to avoid the patch-001
    # repeated-prompt corruption bug masking real thinking-mode regressions.
    EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh "$preset" > "/tmp/think_test_${preset}.log" 2>&1 &
    local pid=$!

    if ! wait_for_server; then
        echo "  FAILED to start (see /tmp/think_test_${preset}.log)"
        kill $pid 2>/dev/null || true
        return
    fi

    PORT=$PORT python3 - <<'PY'
import os, json, re, urllib.request

port = int(os.environ["PORT"])
url = f"http://localhost:{port}/v1/chat/completions"

prompts = [
    ("simple",    "What is 2+2? A. 3 B. 4 C. 5 D. 6 -- Answer with just the letter:"),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? A. Yes B. No C. Cannot determine D. Only some -- Answer with just the letter:"),
    ("knowledge", "What is the capital of Japan? A. Beijing B. Seoul C. Tokyo D. Bangkok -- Answer with just the letter:"),
]

def post(payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

think_count = clean_count = total_tokens = 0
for name, prompt in prompts:
    try:
        r = post({"model": "default", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1024, "temperature": 0})
        content = r["choices"][0]["message"]["content"] or ""
        tokens = r["usage"]["completion_tokens"]
        finish = r["choices"][0]["finish_reason"]
        has_tags = "<think>" in content and "</think>" in content
        after = content.split("</think>")[-1].strip() if "</think>" in content else content
        after = re.sub(r"<think>.*", "", after, flags=re.DOTALL).strip()
        letters = re.findall(r"\b[ABCD]\b", after)
        clean = len(letters) > 0
        if has_tags: think_count += 1
        if clean: clean_count += 1
        total_tokens += tokens
        status = []
        if has_tags: status.append("THINK")
        if clean: status.append(f"answer={letters[-1] if letters else '?'}")
        if finish == "length": status.append("TRUNCATED")
        print(f"  {name:12s}: {' '.join(status):30s} ({tokens} tok, {finish})")
    except Exception as e:
        print(f"  {name:12s}: ERROR {e}")

n = len(prompts)
print(f"  ---")
print(f"  Think tags: {think_count}/{n}  Clean answers: {clean_count}/{n}  Avg tokens: {total_tokens//n}")
PY

    pkill -f sglang 2>/dev/null || true
    sleep 3
}

echo "=== M4 THINKING FORMAT TEST ==="
echo "Catches Qwen3 family infinite-think-loop under greedy MLX decode."
echo ""

PRESETS="${PRESETS:-coder-30b devstral gemma4 qwen35 qwen3-moe qwen3-32b}"
for preset in $PRESETS; do
    case "$preset" in
        devstral)   tag="Devstral-24B" ;;
        coder-30b)  tag="Coder-30B" ;;
        gemma4)     tag="Gemma4-26B" ;;
        gemma4-31b) tag="Gemma4-31B" ;;
        qwen35)     tag="Qwen3.5-27B" ;;
        qwen3-moe)  tag="Qwen3-30B-MoE" ;;
        qwen3-32b)  tag="Qwen3-32B" ;;
        *)          tag="$preset" ;;
    esac
    test_thinking "$preset" "$tag"
done

echo ""
echo "=== DONE ==="
