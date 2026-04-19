#!/usr/bin/env python3
"""Synthetic video probe — moving red circle, ask the model where it moves.

Bypasses SGLang's video processor (which needs torchcodec/decord and
pulls in extra torch deps that have OOM'd Qwen3.5-27B on M4). Instead
we generate frames client-side and send them as N `image_url` items.
Qwen VL's vision tower handles image sequences natively — same code
path the working Devstral red-circle test used.

Usage:
    python scripts/eval/test_video.py --port 23334
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request


def make_frames(n_frames: int = 6, w: int = 224, h: int = 224):
    """Return [bytes] PNG frames of a red circle moving left→right."""
    from PIL import Image, ImageDraw

    radius = 22
    out = []
    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        cx = int(radius + (w - 2 * radius) * (i / max(1, n_frames - 1)))
        cy = h // 2
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(220, 30, 30),
            outline=(0, 0, 0),
            width=2,
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def post(url: str, payload: dict, timeout: int = 240) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--prompt", type=str,
                    default="These frames are sampled from a short video in chronological order. "
                            "In which direction does the red circle move? "
                            "Reply with one of: left, right, up, down.")
    args = ap.parse_args()

    frames = make_frames(args.frames)
    content = []
    for fb in frames:
        b64 = base64.b64encode(fb).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": args.prompt})

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    url = f"http://localhost:{args.port}/v1/chat/completions"
    try:
        r = post(url, payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code}: {body[:600]}")
        return 1

    if "choices" not in r:
        print(f"  ERROR: {json.dumps(r)[:400]}")
        return 1
    msg = r["choices"][0]["message"].get("content") or ""
    fin = r["choices"][0].get("finish_reason")
    print(f"  frames={args.frames}  finish={fin}  content={msg!r}")

    msg_l = msg.lower()
    motion_terms = ["right", "moving", "moves", "moved", "translates", "across", "rightward"]
    hit = next((t for t in motion_terms if t in msg_l), None)
    if hit:
        print(f"  PASS — saw motion word {hit!r}")
        return 0
    print("  FAIL — no motion word in response")
    return 1


if __name__ == "__main__":
    sys.exit(main())
