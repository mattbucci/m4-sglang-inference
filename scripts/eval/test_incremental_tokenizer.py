#!/usr/bin/env python
"""Exactness + speed of the incremental tokenizer (patch 027, from the 3090's patch 072), CPU only.

For each preset's tokenizer: a growing multi-turn conversation (prose as user / assistant turns) rendered
through the model's chat template; at each user turn the incremental encode must equal
apply_chat_template(tokenize=True) (= encode(rendered, add_special_tokens=False)), the /generate mode must
equal tokenizer(text), an edited last turn must re-encode exactly, and a random append/cut fuzz over texts
stuffed with special tokens must be exact.

    python scripts/eval/test_incremental_tokenizer.py
    python scripts/eval/test_incremental_tokenizer.py --presets qwen36,flint --target-tokens 60000 --out <json>

The prose is this repo's own Markdown (deterministic, punctuation- and code-heavy), repeated to 3 MB.
"""
import argparse, glob, json, os, random, sys, time
os.environ["SGLANG_INCREMENTAL_TOKENIZE"] = "1"
from transformers import AutoTokenizer
from sglang.srt.managers.incremental_tokenizer import IncrementalTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRESETS = {
    "qwen36": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "qwen38": "mlx-community/Qwen3.8-27B-4bit",
    "qwen36-27b": "mlx-community/Qwen3.6-27B-4bit",
    "qwen35": "mlx-community/Qwen3.5-27B-4bit",
    "qwen35-9b-8bit": "mlx-community/Qwen3.5-9B-MLX-8bit",
    "flint": "mlx-community/flint-oQ8e",
    "devstral": "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
    "nemotron-30b": "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
    "coder-30b": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
    "qwen3-moe": "mlx-community/Qwen3-30B-A3B-4bit-DWQ",
    "qwen3-32b": "mlx-community/Qwen3-32B-4bit-DWQ",
}


def snapshot(repo_id: str) -> str:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + repo_id.replace("/", "--"), "snapshots", "*")))
    return snaps[-1] if snaps else repo_id


def prose() -> str:
    parts = []
    for f in sorted(glob.glob(os.path.join(REPO, "*.md")) + glob.glob(os.path.join(REPO, "*", "README.md"))):
        parts.append(open(f, encoding="utf-8", errors="replace").read())
    hay = "\n\n".join(parts) or "lorem ipsum "
    while len(hay) < 3_000_000:
        hay += hay
    return hay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--target-tokens", type=int, default=250000)
    ap.add_argument("--chunk-chars", type=int, default=6000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hay = prose()
    results = []; fails = 0
    for name in a.presets.split(","):
        path = snapshot(PRESETS[name])
        try:
            tok = AutoTokenizer.from_pretrained(path)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: tokenizer load failed: {str(e)[:120]}"); fails += 1; continue
        inc = IncrementalTokenizer(tok, verify=False)
        row = {"preset": name, "enabled": inc.enabled, "special_prefix_ok": inc.special_prefix_ok, "turns": []}
        if not inc.enabled:
            print(f"{name}: incremental tokenizer DISABLED by the exactness check (safe fallback, not a failure)")
            results.append(row); continue
        msgs = [{"role": "system", "content": "You are a careful assistant. " + hay[:2000]}]
        pos = 2000; turn = 0; n_tok = 0; mism = 0; checkpoints = []
        while n_tok < a.target_tokens:
            role = "user" if turn % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": hay[pos:pos + a.chunk_chars]}); pos += a.chunk_chars; turn += 1
            if role == "assistant":
                continue
            try:
                t0 = time.perf_counter(); rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); t_render = time.perf_counter() - t0
                t0 = time.perf_counter(); full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False); t_full = time.perf_counter() - t0
            except Exception as e:  # noqa: BLE001
                print(f"{name}: chat template failed at turn {turn}: {str(e)[:120]}"); break
            t0 = time.perf_counter(); got = inc.encode(rendered, add_special_tokens=False); t_inc = time.perf_counter() - t0
            n_tok = len(full)
            if got != full:
                mism += 1
                i = next((k for k in range(min(len(got), len(full))) if got[k] != full[k]), min(len(got), len(full)))
                print(f"{name} turn {turn}: MISMATCH at token {i} of {len(full)} (inc {len(got)})")
            if turn in (1, 3, 7, 15, 31) or n_tok >= a.target_tokens or turn % 20 == 1:
                checkpoints.append({"turn": turn, "tokens": n_tok, "render_ms": round(t_render * 1e3, 1), "full_ms": round(t_full * 1e3, 1), "inc_ms": round(t_inc * 1e3, 1)})
                print(f"{name:14} turn {turn:3} {n_tok:>7} tok  render {t_render*1e3:6.1f} ms  full encode {t_full*1e3:7.1f} ms  incremental {t_inc*1e3:6.1f} ms  {'ok' if got == full else 'MISMATCH'}", flush=True)
        msgs[-1] = {"role": "user", "content": "Actually, ignore that. " + hay[pos:pos + 3000]}
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
        t0 = time.perf_counter(); got = inc.encode(rendered, add_special_tokens=False); t_edit = time.perf_counter() - t0
        edit_ok = got == full; mism += 0 if edit_ok else 1
        print(f"{name:14} edited last turn: {'ok' if edit_ok else 'MISMATCH'} in {t_edit*1e3:.1f} ms (hits {inc.stats['hits']}/{inc.stats['calls']})")
        gen_ok = True
        if inc.special_prefix_ok:
            base = hay[:400_000]
            for cut in (100_000, 250_000, 400_000):
                txt = base[:cut]; got = inc.encode(txt, add_special_tokens=True); full = tok(txt)["input_ids"]
                gen_ok = gen_ok and got == full
            print(f"{name:14} /generate mode (add_special_tokens=True): {'ok' if gen_ok else 'MISMATCH'}")
        mism += 0 if gen_ok else 1
        specials = [str(t) for t in tok.added_tokens_decoder.values()][:40]; words = hay.split()[:5000]; rng = random.Random(0); fuzz_bad = 0
        inc2 = IncrementalTokenizer(tok, verify=False)
        prev = ""
        for _ in range(300):
            if prev and rng.random() < 0.7:
                cut = rng.randint(max(0, len(prev) - 2000), len(prev)); txt = prev[:cut]
            else:
                txt = ""
            txt += " ".join(rng.choice(specials) if rng.random() < 0.05 else rng.choice(words) for _ in range(rng.randint(2000, 4000)))
            if rng.random() < 0.3:
                txt += rng.choice(specials)
            got = inc2.encode(txt, add_special_tokens=False); full = tok(txt, add_special_tokens=False)["input_ids"]
            fuzz_bad += 0 if got == full else 1; prev = txt
        print(f"{name:14} fuzz 300 random append/cut texts with specials: {300 - fuzz_bad}/300 exact (hits {inc2.stats['hits']})")
        mism += fuzz_bad
        row.update({"turns": checkpoints, "mismatches": mism, "edit_ok": edit_ok, "generate_mode_ok": gen_ok, "fuzz_exact": 300 - fuzz_bad})
        results.append(row); fails += 1 if mism else 0
    if a.out:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "results": results}, open(a.out, "w"), indent=1); print("wrote", a.out)
    print("RESULT:", "FAIL" if fails else "PASS"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
