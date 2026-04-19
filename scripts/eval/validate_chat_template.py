#!/usr/bin/env python3
"""Chat template validator — run before trusting any quantized MLX model.

Adapted from the 3090 sister repo. Catches static (no-server-needed) issues:

  1. tokenizer.chat_template is None — chat completions will 400.
  2. Template renders a leading BOS — SGLang adds BOS again, doubled BOS yields
     <unk> at decode time (the Devstral AWQ failure mode).
  3. Thinking templates: enable_thinking kwarg actually toggles the marker
     (catches Qwen3.5/Coder where the template embeds <think> regardless).
  4. Vision content placeholder renders without raising.

Optionally also pings a live server with a tiny prompt to catch unicode-replacement
chars or empty content.

Usage:
    # Static check on a local model
    python scripts/eval/validate_chat_template.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

    # Plus live roundtrip
    python scripts/eval/validate_chat_template.py --model <path> --port 23334
"""
import argparse
import sys


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def check_has_template(tok):
    tmpl = getattr(tok, "chat_template", None)
    if not tmpl:
        return False, "tokenizer.chat_template is None — model will error on /v1/chat/completions"
    return True, f"template length: {len(tmpl)} chars"


def check_bos_handling(tok):
    msgs = [{"role": "user", "content": "Hello"}]
    try:
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return False, f"render failed: {e}"
    bos = getattr(tok, "bos_token", None)
    if bos and rendered.startswith(bos):
        return False, f"template starts with BOS token '{bos}' — SGLang auto-adds BOS, doubled BOS produces <unk>"
    return True, "no leading BOS"


def check_thinking(tok):
    msgs = [{"role": "user", "content": "What is 1+1?"}]
    markers = ("<think>", "<|channel>", "<start_working_out>")
    findings = []
    for enable in (True, False):
        try:
            out = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": enable},
            )
        except TypeError:
            try:
                out = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable,
                )
            except Exception:
                return None, "template does not accept enable_thinking kwarg — OK for non-thinking models"
        except Exception as e:
            return False, f"render failed with enable_thinking={enable}: {e}"
        findings.append((enable, out))

    on, off = findings[0][1], findings[1][1]
    markers_on = [t for t in markers if t in on]
    markers_off = [t for t in markers if t in off]
    if on == off and not markers_on and not markers_off:
        return None, "template ignores enable_thinking and has no thinking markers — not a thinking model"
    if on == off:
        return False, f"enable_thinking had no effect but template embeds {markers_off} — model emits thinking regardless"
    if not markers_on:
        return False, "enable_thinking=True did not introduce any known thinking marker"
    return True, f"thinking markers toggle correctly (on={markers_on})"


def check_vision_placeholder(tok):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "Describe this image."},
        ],
    }]
    try:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return None, f"template does not support image content (text-only model OK): {e}"
    placeholder = "<image>" in out or "<|image|>" in out or "<image_pad>" in out or "<start_of_image>" in out
    return True, f"vision content rendered (placeholder present: {placeholder})"


def check_live(port, model, tok):
    import urllib.request, json
    msgs = [{"role": "user", "content": "Say 'pong'."}]
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=json.dumps({
                "model": model, "messages": msgs, "max_tokens": 8, "temperature": 0,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"].get("content", "")
        if not text.strip():
            return False, "server returned empty content — possible <unk> or template bug"
        if "<unk>" in text.lower() or "\ufffd" in text:
            return False, f"output contains <unk> / unknown chars: {text!r}"
        return True, f"server responded: {text!r}"
    except Exception as e:
        return False, f"live check failed: {e}"


CHECKS = [
    ("has chat_template", check_has_template),
    ("no doubled BOS", check_bos_handling),
    ("thinking toggle", check_thinking),
    ("vision content", check_vision_placeholder),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=None, help="Also run live roundtrip against server")
    args = ap.parse_args()

    print(f"Validating chat template: {args.model}")
    tok = load_tokenizer(args.model)

    results = []
    for name, fn in CHECKS:
        try:
            status, detail = fn(tok)
        except Exception as e:
            status, detail = False, f"check crashed: {e}"
        tag = "OK" if status is True else ("SKIP" if status is None else "FAIL")
        print(f"  [{tag:4}] {name}: {detail}")
        results.append((name, status, detail))

    live_status = None
    if args.port:
        print(f"\nLive server check on :{args.port}")
        live_status, live_detail = check_live(args.port, args.model, tok)
        tag = "OK" if live_status is True else "FAIL"
        print(f"  [{tag:4}] live /v1/chat/completions roundtrip: {live_detail}")

    failures = [r for r in results if r[1] is False]
    if args.port and live_status is False:
        failures.append(("live", False, "server check failed"))

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    else:
        print("PASS: chat template is ready for launch")


if __name__ == "__main__":
    main()
