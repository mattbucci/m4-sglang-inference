#!/usr/bin/env python3
"""Served check: a request keeps its own positions while another one is live.

mlx-vlm's Qwen language models number the tokens of a request with history
from the cache offset plus an mRoPE delta, and keep that delta in one field
on the model: zero after a text prompt, negative after an image (an image
takes more sequence slots than positions). With two live requests the field
holds whatever the last prompt left, so a text stream that is decoding when
an image arrives would be renumbered by the image's delta, and the other way
round. Patch 034 keeps the delta per request.

The probe decodes a text request and an image request alone, then twice more
with the second request submitted while the first is mid-stream:

    text-first    text stream running, image request arrives
    image-first   image stream running, text request arrives

Greedy, thinking off. Launch with ``MAX_RUNNING`` of 2 or more and
``EXTRA_ARGS=--disable-radix-cache`` so every run is a cold prefill and the
solo runs are a like-for-like reference.

On the serial decode path each request's forwards are the same graphs as
when it runs alone, so ``--expect exact`` (the default) requires every
interleaved response to equal its solo reference to the character. With
batched mRoPE decode (``SGLANG_MLX_BATCH_MROPE_DECODE``, patch 035) a
batched step rounds differently from a solo one and greedy text may part
ways at a near-tie, so ``--expect content`` gates on the image answer only
and reports the common prefix; teacher-forced fidelity of that path is
``batched_decode_fidelity.py``.

    python scripts/eval/probe_concurrent_positions.py --port 23334 --out <json>
"""
import argparse
import base64
import io
import json
import sys
import threading
from urllib.request import Request, urlopen

TEXT_PROMPT = (
    "Explain step by step how a binary search tree keeps its keys ordered, "
    "how insertion and deletion work, and what goes wrong when the tree "
    "becomes unbalanced. Write at least 400 words of plain prose."
)
IMAGE_PROMPT = (
    "Describe this image precisely: name the shape and its colour first. "
    "Then write a 300 word story in which that object matters."
)


def make_image(side):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (side, side), color="white")
    q = side // 4
    ImageDraw.Draw(img).ellipse([(q, q), (3 * q, 3 * q)], fill="red", outline="black", width=max(3, side // 85))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stream_chat(base, model, messages, max_tokens, on_piece=None, timeout=900):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    pieces = []
    with urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                pieces.append(piece)
                if on_piece is not None:
                    on_piece(len(pieces))
    return "".join(pieces)


def interleaved(base, model, first, second, after, max_tokens):
    """Run ``first``; submit ``second`` once ``first`` has streamed ``after`` pieces."""
    out, errors = {}, []
    fired = threading.Event()

    def run_second():
        try:
            out["second"] = stream_chat(base, model, second, max_tokens)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    th = threading.Thread(target=run_second)

    def on_piece(n):
        if n >= after and not fired.is_set():
            fired.set()
            th.start()

    out["first"] = stream_chat(base, model, first, max_tokens, on_piece)
    if not fired.is_set():
        raise SystemExit("the first stream ended before the second request was submitted")
    th.join()
    if errors:
        raise SystemExit("second request failed: " + errors[0])
    return out["first"], out["second"]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def sees_circle(text):
    low = text.lower()
    return "red" in low and any(w in low for w in ("circle", "round", "disk", "dot", "ball", "sphere"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--after", type=int, default=24, help="streamed pieces of the first request before the second is submitted")
    ap.add_argument("--image-side", type=int, default=896, help="pixels; a larger image leaves a larger delta")
    ap.add_argument("--expect", choices=("exact", "content"), default="exact")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    model = json.loads(urlopen(base + "/v1/models", timeout=30).read())["data"][0]["id"]
    text = [{"role": "user", "content": TEXT_PROMPT}]
    image = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": make_image(a.image_side)}},
                {"type": "text", "text": IMAGE_PROMPT},
            ],
        }
    ]

    solo_text = stream_chat(base, model, text, a.max_tokens)
    solo_image = stream_chat(base, model, image, a.max_tokens)
    # A second solo pass: the reference has to be reproducible before a
    # difference under concurrency means anything.
    again_text = stream_chat(base, model, text, a.max_tokens)
    again_image = stream_chat(base, model, image, a.max_tokens)
    tf_text, tf_image = interleaved(base, model, text, image, a.after, a.max_tokens)
    if_image, if_text = interleaved(base, model, image, text, a.after, a.max_tokens)

    rows = [
        ("solo repeat", "text", solo_text, again_text),
        ("solo repeat", "image", solo_image, again_image),
        ("text-first", "text", solo_text, tf_text),
        ("text-first", "image", solo_image, tf_image),
        ("image-first", "image", solo_image, if_image),
        ("image-first", "text", solo_text, if_text),
    ]
    result = {"tag": a.tag, "model": model, "expect": a.expect, "max_tokens": a.max_tokens, "image_side": a.image_side, "rows": []}
    print(f"model {model}  tag {a.tag or '-'}  expect {a.expect}")
    print("run           request  chars  equal prefix  same as solo  sees the red circle")
    ok = True
    for run, kind, ref, got in rows:
        same = ref == got
        prefix = common_prefix(ref, got)
        circle = sees_circle(got) if kind == "image" else None
        print(
            f"{run:<13} {kind:<7} {len(got):>6}  {prefix:>12}  {'yes' if same else 'NO':<12}  "
            f"{'-' if circle is None else ('yes' if circle else 'NO')}"
        )
        result["rows"].append(
            {"run": run, "request": kind, "chars": len(got), "equal_prefix": prefix, "same": same, "sees_circle": circle, "text": got}
        )
        if circle is False:
            ok = False
        if not same and (a.expect == "exact" or run == "solo repeat"):
            ok = False
    result["solo_text"], result["solo_image"] = solo_text, solo_image
    result["pass"] = ok
    print("PASS" if ok else "FAIL")
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
