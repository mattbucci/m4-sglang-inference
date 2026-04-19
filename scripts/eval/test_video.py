#!/usr/bin/env python3
"""Synthetic video probe — moving red circle, ask the model where it moves.

Usage:
    python scripts/eval/test_video.py --port 23334
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tempfile
import urllib.request
from pathlib import Path


def make_video(path: Path, n_frames: int = 12, w: int = 224, h: int = 224, fps: int = 4) -> None:
    """Render a red circle moving left→right across n_frames; encode H.264 MP4."""
    import av
    from PIL import Image, ImageDraw

    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    radius = 22
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
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


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
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--keep-video", type=str, default="",
                    help="Optional path to write the MP4 to (for inspection).")
    ap.add_argument("--prompt", type=str,
                    default="In this video, in which direction does the red circle move? "
                            "Reply with one of: left, right, up, down.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "moving_circle.mp4"
        make_video(mp4, n_frames=args.frames)
        if args.keep_video:
            Path(args.keep_video).write_bytes(mp4.read_bytes())
            print(f"  saved video to {args.keep_video} ({mp4.stat().st_size} bytes)")

        b64 = base64.b64encode(mp4.read_bytes()).decode("ascii")
        data_url = f"data:video/mp4;base64,{b64}"

        payload = {
            "model": "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        url = f"http://localhost:{args.port}/v1/chat/completions"
        try:
            r = post(url, payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  HTTP {e.code}: {body[:400]}")
            return 1

        if "choices" not in r:
            print(f"  ERROR: {json.dumps(r)[:400]}")
            return 1
        msg = r["choices"][0]["message"].get("content") or ""
        fin = r["choices"][0].get("finish_reason")
        print(f"  finish={fin}  content={msg!r}")

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
