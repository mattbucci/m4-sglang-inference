"""Content-aware code-generation probe — ported from the 3090 sister repo.

Complements probe_thinking (channel-structure) and probe_vision (image
recognition). validate_capabilities.py:check_basic only asks "capital of
France" — passes on any model that emits "paris" somewhere. For coder
models (coder-30b, devstral) we need to actually verify the model can
synthesize working code.

This probe:
  - Sends two algorithmic prompts (paren-balance + interval-merge).
  - Extracts the Python code block from the response.
  - Executes it and runs hand-rolled unit tests.
  - Classifies STRONG (8/8) / PARTIAL (some) / FAIL (none).

M4 notes:
  - Greedy decode only on MLX backend (temperature=0 to stay on the supported
    sampler path; 3090's reference uses 0.7/0.95/20).
  - Thinking is disabled by default (--enable-thinking opts back in). On
    unified memory a 600-tok think trace + 400-tok code response on the
    Qwen3 family pushes the workload past macOS jetsam during decode. The
    codegen probe is testing code synthesis, not reasoning narration.
  - Default max_tokens=400 (down from 600). Tighter activation scratch budget.

Usage:
  python scripts/eval/probe_codegen.py [--port PORT] [--model MODEL]
                                       [--max-tokens N] [--enable-thinking]
"""
import argparse
import json
import textwrap
from urllib.request import Request, urlopen


PROMPTS = [
    {
        "name": "is_balanced (parens)",
        "fn_name": "is_balanced",
        "prompt": textwrap.dedent("""\
            Write a complete Python function `is_balanced(s)` that takes a
            string of parentheses (just '(' and ')') and returns True if the
            parens are balanced, False otherwise. Just the function — no
            imports, no examples. Use markdown code fences."""),
        "tests": [
            ("()", True),
            ("(())", True),
            ("(()", False),
            (")(", False),
            ("", True),
        ],
    },
    {
        "name": "merge_intervals",
        "fn_name": "merge_intervals",
        "prompt": textwrap.dedent("""\
            Write a complete Python function `merge_intervals(intervals)` that
            takes a list of intervals like [[1,3], [2,6], [8,10], [15,18]]
            and returns the merged non-overlapping intervals as a list. Just
            the function — no imports, no examples. Use markdown code
            fences."""),
        "tests": [
            ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
            ([[1, 4], [4, 5]], [[1, 5]]),
            ([[1, 4]], [[1, 4]]),
        ],
    },
]


def _call(
    host_port: str,
    model: str,
    prompt: str,
    max_tokens: int = 400,
    enable_thinking: bool = False,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # Skip the <think> channel: codegen probes test whether the model
        # can write working code, not whether it can narrate its reasoning
        # first. On M4 unified memory a 600-token thinking trace + 400-token
        # code response on Qwen3-family triggers macOS jetsam mid-decode
        # (see feedback_mem_frac_unified_memory.md). enable_thinking=False
        # is a no-op for non-thinking models (Devstral, Coder-30B).
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = Request(
        f"http://{host_port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urlopen(req, timeout=300).read())
    choice = r["choices"][0]
    msg = choice["message"]
    # If thinking was forcibly disabled, content is the canonical channel.
    # Keep the reasoning_content fallback for the --enable-thinking opt-in path.
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return choice.get("finish_reason"), text, r.get("usage")


_SMART_PUNCT_MAP = str.maketrans({
    "—": "-",   # em-dash  → ASCII hyphen
    "–": "-",   # en-dash  → ASCII hyphen
    "−": "-",   # minus sign → ASCII hyphen
    "‘": "'",   # left single quote → ASCII apostrophe
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "…": "...", # horizontal ellipsis
})


def _extract_code(text: str) -> str:
    """Pull the first ```python ... ``` block (or any ``` block as fallback).

    Devstral and other models occasionally emit smart punctuation (U+2014
    em-dash etc.) inside comments. Python's tokenizer rejects those with
    SyntaxError before the function body even runs. Translate the common
    offenders back to ASCII equivalents so the probe measures the model's
    code correctness, not its UTF-8 hygiene.
    """
    if "```python" in text:
        code = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        code = text.split("```", 1)[1].split("```", 1)[0]
    else:
        code = text
    return code.translate(_SMART_PUNCT_MAP)


def _run_one(
    host_port: str,
    model: str,
    spec: dict,
    max_tokens: int,
    enable_thinking: bool,
) -> tuple[int, int, str]:
    name = spec["name"]
    fn_name = spec["fn_name"]
    print("=" * 60)
    print(f"PROBE: {name}")
    finish, content, usage = _call(
        host_port, model, spec["prompt"],
        max_tokens=max_tokens, enable_thinking=enable_thinking,
    )
    print(f"finish={finish}  usage={usage}")
    print()
    print("--- response (first 400 chars) ---")
    print(content[:400])
    print()
    code = _extract_code(content).strip()
    if not code:
        print("FAIL: no code extracted")
        return 0, len(spec["tests"]), f"{name}: no code"
    try:
        ns = {}
        exec(code, ns)  # noqa: S102 — controlled local probe
    except Exception as e:
        print(f"FAIL: code did not exec: {e!r}")
        return 0, len(spec["tests"]), f"{name}: exec failed ({type(e).__name__})"
    fn = ns.get(fn_name)
    if fn is None:
        print(f"FAIL: {fn_name} not defined")
        return 0, len(spec["tests"]), f"{name}: fn not defined"
    n_pass = 0
    for inp, want in spec["tests"]:
        try:
            got = fn(*inp) if isinstance(inp, tuple) else fn(inp)
            ok = got == want
            print(f"  {fn_name}({inp!r}) = {got!r}  expected {want!r}  {'OK' if ok else 'FAIL'}")
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  {fn_name}({inp!r}) raised {e!r}")
    return n_pass, len(spec["tests"]), f"{name}: {n_pass}/{len(spec['tests'])}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--model", default="default")
    p.add_argument(
        "--max-tokens", type=int, default=400,
        help="Per-request max_tokens cap. Lower bounds activation scratch on M4.",
    )
    p.add_argument(
        "--enable-thinking", action="store_true",
        help="Opt back into the <think> channel. Default off — codegen probe "
             "tests code synthesis, not reasoning narration, and thinking "
             "traces compete with the code budget under tight memory.",
    )
    args = p.parse_args()

    host_port = f"localhost:{args.port}"
    summaries = []
    total_pass = 0
    total_count = 0
    for spec in PROMPTS:
        n_pass, n_total, summary = _run_one(
            host_port, args.model, spec,
            max_tokens=args.max_tokens, enable_thinking=args.enable_thinking,
        )
        summaries.append(summary)
        total_pass += n_pass
        total_count += n_total
        print()

    print("=" * 60)
    for s in summaries:
        print(f"  {s}")
    print(f"OVERALL: {total_pass}/{total_count}")
    print()
    if total_pass == total_count:
        verdict = "STRONG"
        rc = 0
    elif total_pass > 0:
        verdict = "PARTIAL"
        rc = 1
    else:
        verdict = "FAIL"
        rc = 2
    print(f"VERDICT: {verdict}")
    print()
    print("Notes:")
    print("- STRONG = all 8 algorithmic test cases pass.")
    print("- PARTIAL = some tests fail (algorithm mostly right, edge cases broken).")
    print("- FAIL = no working code returned.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
