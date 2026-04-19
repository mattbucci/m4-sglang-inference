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
    per_subject = max(1, n_samples // len(subjects))
    samples = []
    for subj in subjects:
        samples.extend([x for x in ds if x["subject"] == subj][:per_subject])
    samples = samples[:n_samples]
    choices_map = ["A", "B", "C", "D"]
    correct = total = 0

    def eval_one(item):
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
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(eval_one, s) for s in samples]):
            total += 1; correct += f.result()
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0}


def humaneval_eval(chat_url, n_samples=30, max_workers=1, max_tokens=4096):
    """HumanEval code generation pass@1."""
    from datasets import load_dataset
    ds = list(load_dataset("openai/openai_humaneval", split="test"))[:n_samples]
    completions_url = chat_url.replace("/chat/completions", "/completions")
    passed = total = 0

    def eval_one(item):
        try:
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
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(eval_one, s) for s in ds]):
            total += 1; passed += f.result()
    return {"passed": passed, "total": total, "pass_rate": passed / total if total else 0}


def labbench_eval(chat_url, bench_name, n_samples=50, max_workers=1, max_tokens=1024):
    """Run a single LAB-Bench text-only benchmark."""
    from datasets import load_dataset
    import random, string
    ds = list(load_dataset("futurehouse/lab-bench", name=bench_name, split="train"))
    if n_samples and n_samples < len(ds):
        random.seed(42)
        ds = random.sample(ds, n_samples)
    correct = total = 0

    def eval_one(item):
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
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(eval_one, s) for s in ds]):
            result = f.result()
            if result is not None:
                total += 1; correct += result
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0}


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
    """Needle-in-a-haystack at various context lengths."""
    filler = "The quick brown fox jumps over the lazy dog. " * 100
    needle = "The secret password is: BANANA42."
    needle_budget = min(max_tokens, 512)
    results = []
    for ctx in lengths:
        mid = ctx * 2
        haystack = filler[:mid] + "\n" + needle + "\n" + filler[:mid]
        prompt = haystack[:ctx * 4] + "\n\nWhat is the secret password? Answer with just the password."
        try:
            r = _post(chat_url, {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": needle_budget, "temperature": 0,
            }, timeout=900)  # long-context prefill on M4 takes a while
            content = r["choices"][0]["message"]["content"] or ""
            found = "BANANA42" in content
        except Exception:
            found = False
        results.append({"context": ctx, "found": found})
    return {"results": results, "score": sum(r["found"] for r in results) / len(results)}


def run_eval(port, tag, mmlu_n=200, he_n=30, labbench_n=50,
             needle_lengths=[1024, 4096, 16384, 65536], workers=1):
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

    def save():
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)

    if "mmlu" not in results or results["mmlu"].get("total", 0) == 0:
        print(f"\nMMLU ({mmlu_n} samples)...")
        results["mmlu"] = mmlu_eval(chat_url, mmlu_n, max_workers=workers, max_tokens=mc_budget)
        print(f"  {results['mmlu']['accuracy']:.1%}")
        save()
    else:
        print(f"\nMMLU: {results['mmlu']['accuracy']:.1%} (cached)")

    if "humaneval" not in results or results["humaneval"].get("total", 0) == 0:
        print(f"\nHumanEval ({he_n} samples)...")
        results["humaneval"] = humaneval_eval(chat_url, he_n, max_workers=workers, max_tokens=code_budget)
        print(f"  {results['humaneval']['pass_rate']:.1%}")
        save()
    else:
        print(f"\nHumanEval: {results['humaneval']['pass_rate']:.1%} (cached)")

    if labbench_n <= 0:
        print(f"\nLAB-Bench: skipped (labbench_n={labbench_n})")
    elif "labbench" not in results or not results["labbench"].get("_overall"):
        print(f"\nLAB-Bench ({labbench_n} samples per benchmark, {len(LAB_BENCH_BENCHMARKS)} benchmarks)...")
        results["labbench"] = labbench_suite(chat_url, n_samples=labbench_n, max_workers=workers, max_tokens=mc_budget)
        print(f"  Overall: {results['labbench']['_overall']['accuracy']:.1%}")
        save()
    else:
        lb = results["labbench"]
        print(f"\nLAB-Bench: {lb['_overall']['accuracy']:.1%} (cached)")
        for bench in LAB_BENCH_BENCHMARKS:
            if bench in lb:
                print(f"    {bench}: {lb[bench]['accuracy']:.1%}")

    if "needle" not in results or not results["needle"].get("results"):
        print(f"\nNeedle ({needle_lengths})...")
        results["needle"] = needle_eval(chat_url, needle_lengths, max_tokens=mc_budget)
        for r in results["needle"]["results"]:
            print(f"  {r['context']:>6d}: {'OK' if r['found'] else 'MISS'}")
        save()
    else:
        print(f"\nNeedle: {results['needle']['score']:.1%} (cached)")

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

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"][:len(tags)]
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
    args = parser.parse_args()

    if args.no_thinking:
        _chat_template_kwargs = {"enable_thinking": False}
        # Re-bind module global so _post() picks it up.
        import sys as _sys
        _sys.modules[__name__]._chat_template_kwargs = _chat_template_kwargs

    if args.run:
        lengths = [int(x) for x in args.needle_lengths.split(",")]
        run_eval(args.port, args.tag, args.mmlu_samples, args.humaneval_samples,
                 args.labbench_samples, lengths, args.workers)
    if args.chart:
        generate_charts()
    if not args.run and not args.chart:
        parser.print_help()
