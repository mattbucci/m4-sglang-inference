#!/usr/bin/env python3
"""Run quality evals against an MLX SGLang server and generate comparison charts.

Adapted from the 3090/R9700 sister repos so M4 numbers can be lined up against
their published quality table:

    Benchmarks: MMLU, HumanEval, LAB-Bench (7 science benchmarks), Needle-in-Haystack
    Output:     benchmarks/quality/<tag>.json + benchmarks/quality/quality_comparison.png

Designed for single-user MLX servers — `--workers 1` is the right default; the MLX
backend processes batches serially anyway, so concurrent client requests just add
queueing latency.

Usage:
    # Run a single model (server already up on PORT):
    python scripts/eval/eval_and_chart.py --run --port 23334 --tag "Coder-30B"

    # Re-render charts from saved results:
    python scripts/eval/eval_and_chart.py --chart

    # Smaller sample sizes for a quick check:
    python scripts/eval/eval_and_chart.py --run --port 23334 --tag "Devstral-24B" \
        --mmlu-samples 50 --humaneval-samples 10 --labbench-samples 10
"""
import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS_DIR = Path("benchmarks/quality")
LAB_BENCH_BENCHMARKS = ["LitQA2", "DbQA", "SuppQA", "TableQA", "ProtocolQA", "SeqQA", "CloningScenarios"]

# Module-level: extra chat_template_kwargs added to every chat request.
# Set via --no-thinking on the CLI (Qwen3 family enters infinite <think>
# loop under greedy MLX otherwise).
_chat_template_kwargs = None

# Server-death detection — set to True once mmlu/humaneval/labbench/needle
# sees a connection-class exception (server reaped by macOS jetsam mid-eval,
# scheduler crashed, etc). Once True, remaining samples short-circuit
# instead of accumulating false misses + the eval result is tagged
# `server_dead: True` so consumers can distinguish "server died" from
# "model genuinely got 0%". Patched 2026-05-18 across all four functions
# after the 2026-05-17 Qwen Needle "regression" root cause showed jetsam
# can silently zero out an entire eval section.
import http.client as _http_client
import socket as _socket
import urllib.error as _urllib_error
_SERVER_DEAD_EXCEPTIONS = (
    _urllib_error.URLError,
    _http_client.RemoteDisconnected,
    ConnectionRefusedError,
    _socket.timeout,
)
_server_dead_flag = {"dead": False}


def _is_server_dead_exc(exc: BaseException) -> bool:
    """True if `exc` indicates the SGLang server is unreachable, not that
    the model produced bad output. URLError wraps many transport-level
    failures; tighten the check by looking at the wrapped reason.
    """
    if isinstance(exc, _SERVER_DEAD_EXCEPTIONS):
        return True
    # Some library wraps RemoteDisconnected inside URLError; the str
    # representation usually contains "Connection refused" /
    # "Remote end closed connection" / "Can't assign requested address".
    msg = repr(exc).lower()
    for token in (
        "connection refused",
        "remote end closed",
        "can't assign requested",
        "broken pipe",
    ):
        if token in msg:
            return True
    return False


def _server_dead_check_and_set(exc: BaseException) -> bool:
    """Set the module-level dead flag on any matching exception and return
    True. Returns False if `exc` is something else (model output bug, etc).
    """
    if _is_server_dead_exc(exc):
        _server_dead_flag["dead"] = True
        return True
    return False


def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    if _chat_template_kwargs is not None and "chat/completions" in url:
        payload = dict(payload)
        payload["chat_template_kwargs"] = _chat_template_kwargs
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8").strip()
        return json.loads(body) if body else {}


def get_max_tokens(base_url: str, default: int = 4096) -> int:
    try:
        r = _get(f"{base_url}/v1/models")
        return r["data"][0].get("max_model_len", default)
    except Exception:
        return default


def mmlu_eval(chat_url, n_samples=200, max_workers=1, max_tokens=1024):
    """MMLU multiple-choice reasoning."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    subjects = list(set(ds["subject"]))
    # MMLU has 57 subjects. ``n_samples // len(subjects)`` floors to 1 when
    # n_samples=100, capping the eval at 57 samples and producing a total
    # that never matches the requested 100. Use ceil-division so the bucket
    # is always large enough; the final ``samples[:n_samples]`` truncates
    # back down. With n_samples=100 → per_subject=2 → 114 candidates → 100.
    per_subject = max(1, -(-n_samples // len(subjects)))
    samples = []
    for subj in subjects:
        samples.extend([x for x in ds if x["subject"] == subj][:per_subject])
    samples = samples[:n_samples]
    choices_map = ["A", "B", "C", "D"]
    correct = total = 0

    def eval_one(item):
        if _server_dead_flag["dead"]:
            return None  # server gone — short-circuit, don't count
        q, choices, answer_idx = item["question"], item["choices"], item["answer"]
        prompt = f"Question: {q}\n"
        for i, c in enumerate(choices):
            prompt += f"{choices_map[i]}. {c}\n"
        prompt += "\nAnswer with just the letter (A, B, C, or D):"
        try:
            r = _post(chat_url, {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0,
            })
            content = r["choices"][0]["message"]["content"] or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            matches = re.findall(r"\b[ABCD]\b", content)
            return (matches[-1] if matches else "") == choices_map[answer_idx]
        except Exception as e:
            if _server_dead_check_and_set(e):
                return None
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(eval_one, s) for s in samples]):
            result = f.result()
            if result is None:
                # server died; don't count this sample
                continue
            total += 1; correct += int(result)
    out = {"correct": correct, "total": total, "accuracy": correct / total if total else 0}
    if _server_dead_flag["dead"]:
        out["server_dead"] = True
    return out


_HUMANEVAL_CHAT_INSTRUCTION = (
    "Complete the following Python function. Return ONLY the function body "
    "(no `def` line, no class, no explanation, no markdown). Indent with 4 spaces."
)


def _strip_chat_code(text: str) -> str:
    """Pull a function body out of a chat reply.

    Handles three common shapes IT models produce:
    1. Plain indented body (best case — return as-is).
    2. ```python ... ``` fenced block — extract the contents.
    3. Full `def name(...)` redefinition — pull out the body only.

    Also removes any `<think>…</think>` traces and leading/trailing prose.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    fenced = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # If the model re-emitted the `def` signature, drop it and keep only the body.
    def_match = re.search(r"^\s*def\s+\w+\s*\([^)]*\)[^:]*:\s*\n(.*)", text, flags=re.DOTALL)
    if def_match:
        text = def_match.group(1)
    return text.rstrip()


def humaneval_eval(chat_url, n_samples=30, max_workers=1, max_tokens=4096, mode="completions"):
    """HumanEval code generation pass@1.

    Two modes:
    - ``completions`` (default, sister-team compatible): POST to /v1/completions
      with the function-signature prefix and let the base model continue. Works
      for coder-tuned models (Coder-30B 95%) but fails for instruction-tuned
      models that prepend chat tokens to every request (Gemma4 → 0%).
    - ``chat``: POST to /v1/chat/completions with an explicit "complete this
      function" instruction. Strips markdown fences and any redefined `def`
      from the reply before exec'ing. Use for IT-only checkpoints that the
      base-completions path can't reach.
    """
    from datasets import load_dataset
    ds = list(load_dataset("openai/openai_humaneval", split="test"))[:n_samples]
    completions_url = chat_url.replace("/chat/completions", "/completions")
    passed = total = 0

    def eval_one(item):
        if _server_dead_flag["dead"]:
            return None
        try:
            if mode == "chat":
                r = _post(chat_url, {
                    "model": "default",
                    "messages": [
                        {"role": "user",
                         "content": f"{_HUMANEVAL_CHAT_INSTRUCTION}\n\n```python\n{item['prompt']}```"},
                    ],
                    "max_tokens": min(max_tokens, 4096),
                    "temperature": 0,
                }, timeout=120)
                raw = r["choices"][0]["message"]["content"] or ""
                comp = _strip_chat_code(raw)
                # If chat model emitted just the body without indent, indent it.
                if comp and not comp.lstrip().startswith(("    ", "\t")) \
                        and "\n    " not in comp[:80]:
                    comp = "\n".join("    " + ln if ln.strip() else ln
                                     for ln in comp.splitlines())
            else:
                r = _post(completions_url, {
                    "prompt": item["prompt"],
                    "max_tokens": min(max_tokens, 4096),
                    "temperature": 0,
                    "stop": ["\ndef ", "\nclass ", "\n#", "\nif __name__"],
                }, timeout=120)
                comp = re.sub(r"<think>.*?</think>", "", r["choices"][0]["text"], flags=re.DOTALL)
                comp = re.sub(r"<think>.*", "", comp, flags=re.DOTALL)
            g = {}
            exec(item["prompt"] + comp + "\n" + item["test"], g)
            g["check"](g[item["entry_point"]])
            return True
        except Exception as e:
            if _server_dead_check_and_set(e):
                return None
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(eval_one, s) for s in ds]):
            result = f.result()
            if result is None:
                continue
            total += 1; passed += int(result)
    out = {"passed": passed, "total": total, "pass_rate": passed / total if total else 0,
           "mode": mode}
    if _server_dead_flag["dead"]:
        out["server_dead"] = True
    return out


def labbench_eval(chat_url, bench_name, n_samples=50, max_workers=1, max_tokens=1024):
    """Run a single LAB-Bench text-only benchmark."""
    from datasets import load_dataset
    import random, string
    ds = list(load_dataset("futurehouse/lab-bench", name=bench_name, split="train"))
    if n_samples and n_samples < len(ds):
        random.seed(42)
        ds = random.sample(ds, n_samples)
    correct = total = 0

    # Sentinel to disambiguate "no distractors, skip" (=None) from
    # "server dead, abort" (=_SERVER_DEAD).
    _SERVER_DEAD = object()

    def eval_one(item):
        if _server_dead_flag["dead"]:
            return _SERVER_DEAD
        q = item["question"]
        ideal = item["ideal"]
        distractors = item.get("distractors") or []
        if not distractors:
            return None
        options = [ideal] + distractors
        random.seed(hash(q))
        random.shuffle(options)
        choices_map = list(string.ascii_uppercase[:len(options)])
        correct_letter = choices_map[options.index(ideal)]

        prompt = f"Question: {q}\n\nOptions:\n"
        for i, opt in enumerate(options):
            prompt += f"{choices_map[i]}. {opt}\n"
        prompt += "\nAnswer with just the letter:"
        try:
            r = _post(chat_url, {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0,
            })
            content = r["choices"][0]["message"]["content"] or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            matches = re.findall(r"\b[A-Z]\b", content)
            return (matches[-1] if matches else "") == correct_letter
        except Exception as e:
            if _server_dead_check_and_set(e):
                return _SERVER_DEAD
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(eval_one, s) for s in ds]):
            result = f.result()
            if result is None or result is _SERVER_DEAD:
                continue
            total += 1; correct += int(result)
    out = {"correct": correct, "total": total, "accuracy": correct / total if total else 0}
    if _server_dead_flag["dead"]:
        out["server_dead"] = True
    return out


def labbench_suite(chat_url, n_samples=50, max_workers=1, max_tokens=1024):
    results = {}
    for bench in LAB_BENCH_BENCHMARKS:
        print(f"    {bench}...", end=" ", flush=True)
        r = labbench_eval(chat_url, bench, n_samples=n_samples, max_workers=max_workers, max_tokens=max_tokens)
        results[bench] = r
        print(f"{r['accuracy']:.1%} ({r['correct']}/{r['total']})")
    total_correct = sum(r["correct"] for r in results.values())
    total_n = sum(r["total"] for r in results.values())
    results["_overall"] = {
        "correct": total_correct, "total": total_n,
        "accuracy": total_correct / total_n if total_n else 0,
    }
    return results


def needle_eval(chat_url, lengths=[1024, 4096, 16384, 65536], max_tokens=512):
    """Needle-in-a-haystack at various context lengths.

    Distinguishes a real model miss (server alive, model didn't include the
    needle in its answer) from server death (connection refused / remote
    disconnect / network errors — typically macOS jetsam reaping the
    sglang scheduler after a memory-pressure spike during a prior eval
    phase). The 2026-05-17 root cause of the apparent "Qwen3.5 Needle
    100%->0% regression" was the latter: a long CloningScenarios prompt
    in the preceding LAB-Bench phase pushed memory past the jetsam
    threshold; all subsequent eval requests (including every Needle
    length) recorded `found=False` purely because the connection died.
    Now those rows are tagged with `server_dead=True` and reported as
    such instead of being silently scored as misses.
    """
    import urllib.error, http.client, socket
    filler = "The quick brown fox jumps over the lazy dog. " * 100
    needle = "The secret password is: BANANA42."
    needle_budget = min(max_tokens, 512)
    results = []
    for ctx in lengths:
        mid = ctx * 2
        haystack = filler[:mid] + "\n" + needle + "\n" + filler[:mid]
        prompt = haystack[:ctx * 4] + "\n\nWhat is the secret password? Answer with just the password."
        entry = {"context": ctx, "found": False}
        try:
            r = _post(chat_url, {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": needle_budget, "temperature": 0,
            }, timeout=900)  # long-context prefill on M4 takes a while
            content = r["choices"][0]["message"]["content"] or ""
            entry["found"] = "BANANA42" in content
        except (urllib.error.URLError, http.client.RemoteDisconnected,
                ConnectionRefusedError, socket.timeout) as e:
            entry["server_dead"] = True
            entry["error"] = repr(e)
        except Exception as e:
            entry["error"] = repr(e)
        results.append(entry)
    server_dead_any = any(r.get("server_dead") for r in results)
    return {
        "results": results,
        "score": sum(r["found"] for r in results) / len(results),
        "server_dead": server_dead_any,
    }


def _mmlu_cache_ok(d, expected_n):
    """Cached MMLU is reusable when sample size meets the bar and accuracy is plausible.

    A cache with MORE samples than asked is fine — it's a stricter measurement.
    A cache with fewer samples should be re-run since it's less robust. Below
    5% accuracy is below 4-choice random guessing (25%) and indicates a
    server-crashed-mid-eval state.
    """
    if not d or d.get("total", 0) < expected_n:
        return False
    return d.get("accuracy", 0) > 0.05


def _humaneval_cache_ok(d, expected_n, mode="completions"):
    """Cached HumanEval is reusable when sample size meets the bar and the
    evaluation mode matches. A chat-mode cache must not be reused for a
    completions-mode request and vice versa — the methodology differs.
    """
    if not d or d.get("total", 0) < expected_n:
        return False
    cached_mode = d.get("mode", "completions")  # legacy JSONs default to completions
    if cached_mode != mode:
        return False
    # 0% pass@1 is technically possible for a broken model, but combined with
    # 0% on MMLU it indicates a crash. Caller checks MMLU first; if that's
    # invalid, this also gets re-run via the corrupt-run detector.
    return True


def _labbench_cache_ok(d, expected_n):
    """Cached LAB-Bench is reusable when each benchmark meets the sample bar."""
    if not d or not d.get("_overall"):
        return False
    for bench in LAB_BENCH_BENCHMARKS:
        if bench not in d:
            return False
        if d[bench].get("total", 0) < expected_n:
            return False
    # Detect "every benchmark scored 0/total" — that's a crash, not a
    # legitimate result.
    if d["_overall"].get("correct", 0) == 0:
        return False
    return True


def _needle_cache_ok(d, expected_lengths):
    """Cached Needle is reusable when every requested length is present.

    Extra cached lengths are fine; missing any means re-run.
    """
    if not d or not d.get("results"):
        return False
    cached_lengths = {r["context"] for r in d["results"]}
    return all(L in cached_lengths for L in expected_lengths)


def run_eval(port, tag, mmlu_n=200, he_n=30, labbench_n=50,
             needle_lengths=[1024, 4096, 16384, 65536], workers=1,
             humaneval_mode="completions"):
    # Reset the jetsam flag — a previous run in the same Python process
    # shouldn't leave us tagged "dead" for a fresh server.
    _server_dead_flag["dead"] = False
    base_url = f"http://localhost:{port}"
    chat_url = f"{base_url}/v1/chat/completions"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    max_ctx = get_max_tokens(base_url)
    mc_budget = min(1024, max_ctx - 512)
    code_budget = min(4096, max_ctx - 512)
    print(f"{'=' * 50}")
    print(f"Quality Eval: {tag} (workers={workers}, max_context={max_ctx})")
    print(f"  MC budget: {mc_budget} tokens, Code budget: {code_budget} tokens")
    print(f"{'=' * 50}")

    outfile = RESULTS_DIR / f"{tag.replace(' ', '_')}.json"
    if outfile.exists():
        results = json.load(open(outfile))
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
    else:
        results = {"tag": tag, "timestamp": time.strftime("%Y-%m-%d %H:%M"), "max_context": max_ctx}

    # Whole-run corruption detector: if MMLU shows ≤5% (below random guessing),
    # the server probably died mid-eval and the rest of the sections are also
    # bogus zeroes. Invalidate everything so we get a clean re-run.
    if "mmlu" in results and results["mmlu"].get("total", 0) > 0 \
            and results["mmlu"].get("accuracy", 0) <= 0.05:
        print(f"\n[cache] {tag}: MMLU={results['mmlu']['accuracy']:.1%} ≤ 5% — "
              f"prior run looks corrupt, invalidating all cached sections.")
        for k in ("mmlu", "humaneval", "labbench", "needle"):
            results.pop(k, None)

    def save():
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)

    if not _mmlu_cache_ok(results.get("mmlu"), mmlu_n):
        print(f"\nMMLU ({mmlu_n} samples)...")
        results["mmlu"] = mmlu_eval(chat_url, mmlu_n, max_workers=workers, max_tokens=mc_budget)
        print(f"  {results['mmlu']['accuracy']:.1%}")
        save()
    else:
        print(f"\nMMLU: {results['mmlu']['accuracy']:.1%} (cached, {results['mmlu']['total']} samples)")

    if not _humaneval_cache_ok(results.get("humaneval"), he_n, humaneval_mode):
        print(f"\nHumanEval ({he_n} samples, mode={humaneval_mode})...")
        results["humaneval"] = humaneval_eval(chat_url, he_n, max_workers=workers,
                                              max_tokens=code_budget, mode=humaneval_mode)
        print(f"  {results['humaneval']['pass_rate']:.1%}")
        save()
    else:
        print(f"\nHumanEval: {results['humaneval']['pass_rate']:.1%} "
              f"(cached, {results['humaneval']['total']} samples, mode={humaneval_mode})")

    if labbench_n <= 0:
        print(f"\nLAB-Bench: skipped (labbench_n={labbench_n})")
    elif not _labbench_cache_ok(results.get("labbench"), labbench_n):
        print(f"\nLAB-Bench ({labbench_n} samples per benchmark, {len(LAB_BENCH_BENCHMARKS)} benchmarks)...")
        results["labbench"] = labbench_suite(chat_url, n_samples=labbench_n, max_workers=workers, max_tokens=mc_budget)
        print(f"  Overall: {results['labbench']['_overall']['accuracy']:.1%}")
        save()
    else:
        lb = results["labbench"]
        print(f"\nLAB-Bench: {lb['_overall']['accuracy']:.1%} (cached, {labbench_n} per bench)")
        for bench in LAB_BENCH_BENCHMARKS:
            if bench in lb:
                print(f"    {bench}: {lb[bench]['accuracy']:.1%}")

    if not _needle_cache_ok(results.get("needle"), needle_lengths):
        print(f"\nNeedle ({needle_lengths})...")
        results["needle"] = needle_eval(chat_url, needle_lengths, max_tokens=mc_budget)
        for r in results["needle"]["results"]:
            print(f"  {r['context']:>6d}: {'OK' if r['found'] else 'MISS'}")
        save()
    else:
        print(f"\nNeedle: {results['needle']['score']:.1%} (cached, {len(needle_lengths)} lengths)")

    print(f"\nSaved to {outfile}")
    return results


def generate_charts():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib for charts")
        return

    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        results.append(json.load(open(f)))
    if not results:
        print("No results found. Run evals first.")
        return

    tags = [r["tag"] for r in results]

    def get_score(r, path, key):
        d = r
        for p in path:
            d = d.get(p, {})
        if not d:
            return None
        if isinstance(d, dict) and d.get("total", 0) == 0 and key in ("accuracy", "pass_rate"):
            return None
        return d.get(key)

    mmlu = [get_score(r, ["mmlu"], "accuracy") for r in results]
    he = [get_score(r, ["humaneval"], "pass_rate") for r in results]
    labbench = [get_score(r, ["labbench", "_overall"], "accuracy") for r in results]
    needle = [get_score(r, ["needle"], "score") for r in results]

    benchmarks = [
        ("MMLU (%)", mmlu),
        ("HumanEval pass@1 (%)", he),
        ("LAB-Bench (%)", labbench),
        ("Needle-in-Haystack (%)", needle),
    ]
    benchmarks = [(t, d) for t, d in benchmarks if any(v is not None for v in d)]
    n_bench = len(benchmarks)

    fig, axes = plt.subplots(1, n_bench, figsize=(4.5 * n_bench, 5))
    if n_bench == 1:
        axes = [axes]
    fig.suptitle("Quality Comparison (4-bit MLX, M4 Pro)", fontsize=14, fontweight="bold")

    base_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
                   "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39"]
    # Repeat colors if we have more tags than the palette
    colors = [base_colors[i % len(base_colors)] for i in range(len(tags))]
    for ax, (title, data) in zip(axes, benchmarks):
        vals = [v * 100 if v is not None else 0 for v in data]
        tested = [v is not None for v in data]
        bar_colors = [colors[i] if tested[i] else "#E0E0E0" for i in range(len(tags))]
        bars = ax.bar(range(len(tags)), vals, color=bar_colors)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if tested[i]:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 2,
                        "N/A", ha="center", va="bottom", fontsize=9, color="#999")

    plt.tight_layout()
    outpath = RESULTS_DIR / "quality_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {outpath}")

    print(f"\n{'Model':<20} {'MMLU':>8} {'HumanEval':>10} {'LAB-Bench':>10} {'Needle':>8}")
    print("-" * 60)
    for i, _ in enumerate(results):
        m = f"{mmlu[i]*100:.1f}%" if mmlu[i] is not None else "N/A"
        h = f"{he[i]*100:.1f}%" if he[i] is not None else "N/A"
        l = f"{labbench[i]*100:.1f}%" if labbench[i] is not None else "N/A"
        n = f"{needle[i]*100:.1f}%" if needle[i] is not None else "N/A"
        print(f"{tags[i]:<20} {m:>8} {h:>10} {l:>10} {n:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run evals")
    parser.add_argument("--chart", action="store_true", help="Generate charts")
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--mmlu-samples", type=int, default=200)
    parser.add_argument("--humaneval-samples", type=int, default=30)
    parser.add_argument("--labbench-samples", type=int, default=50)
    parser.add_argument("--needle-lengths", type=str, default="1024,4096,16384,65536")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent requests (1 for MLX)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass chat_template_kwargs={'enable_thinking': false} to every "
                             "request. Required for Qwen3 family on greedy MLX to avoid "
                             "infinite <think> loops.")
    parser.add_argument("--invalidate", action="store_true",
                        help="Drop any existing JSON for this tag before running, "
                             "forcing a full re-eval.")
    parser.add_argument("--humaneval-mode", choices=["completions", "chat"],
                        default="completions",
                        help="HumanEval eval style. `completions` (default) uses "
                             "/v1/completions with the function-signature prefix — "
                             "compatible with sister-team 3090/R9700 numbers. "
                             "`chat` uses /v1/chat/completions with an explicit "
                             "completion instruction — required for IT models "
                             "like Gemma4 that don't respond to raw base completion.")
    args = parser.parse_args()

    if args.no_thinking:
        _chat_template_kwargs = {"enable_thinking": False}
        # Re-bind module global so _post() picks it up.
        import sys as _sys
        _sys.modules[__name__]._chat_template_kwargs = _chat_template_kwargs

    if args.invalidate:
        outfile = RESULTS_DIR / f"{args.tag.replace(' ', '_')}.json"
        if outfile.exists():
            print(f"--invalidate: removing {outfile}")
            outfile.unlink()

    if args.run:
        lengths = [int(x) for x in args.needle_lengths.split(",")]
        run_eval(args.port, args.tag, args.mmlu_samples, args.humaneval_samples,
                 args.labbench_samples, lengths, args.workers,
                 humaneval_mode=args.humaneval_mode)
    if args.chart:
        generate_charts()
    if not args.run and not args.chart:
        parser.print_help()
