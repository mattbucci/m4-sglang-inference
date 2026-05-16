"""Content-aware video probe — moving red circle, ask for direction.

Same STRONG / PARTIAL / FAIL classification as probe_vision and probe_codegen.
Two test cases (horizontal + vertical) so a model that hallucinates a single
direction can't pass.

Bypasses SGLang's video processor (which needs torchcodec/decord and pulls
in extra torch deps that have OOM'd Qwen3.5-27B on M4). Instead we generate
N frames client-side and send them as `image_url` content items. Qwen VL /
Gemma 4 / Devstral all handle multi-image input natively — same code path
patch 013 verified for single-image vision probes.

Verdict ladder:
  STRONG  = both directions identified correctly (horizontal=right, vertical=down)
  PARTIAL = one direction correct, the other mentions motion but wrong
  DEGRADED = both mention motion, both wrong direction
  FAIL    = at least one response makes no motion claim at all

M4 notes:
  - Greedy decode only on MLX backend (temperature=0).
  - enable_thinking=False to keep the 200-token response budget for the answer.
  - max_tokens=200 — direction is a one-word answer, no need for verbose
    reasoning that compounds memory pressure across the 2 test cases.

Usage:
  python scripts/eval/probe_video.py [--port PORT] [--model MODEL]
                                     [--frames N] [--max-tokens N]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from urllib.request import Request, urlopen


def make_frames(direction: str, n_frames: int = 6, w: int = 224, h: int = 224) -> list[bytes]:
    """Synthetic frames of a red circle moving in the given direction."""
    from PIL import Image, ImageDraw

    radius = 22
    out = []
    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        t = i / max(1, n_frames - 1)
        if direction == "right":
            cx = int(radius + (w - 2 * radius) * t)
            cy = h // 2
        elif direction == "down":
            cx = w // 2
            cy = int(radius + (h - 2 * radius) * t)
        else:
            raise ValueError(f"unknown direction {direction!r}")
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


def _call(host_port: str, model: str, frames: list[bytes], prompt: str, max_tokens: int):
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(f).decode('ascii')}"}}
        for f in frames
    ]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(
        f"http://{host_port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urlopen(req, timeout=240).read())
    choice = r["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return choice.get("finish_reason"), text, r.get("usage")


def _classify(text: str, want_direction: str) -> str:
    """Return STRONG / PARTIAL / FAIL given the response text + the true direction."""
    t = text.lower()
    # Direction-word vocabulary
    direction_words = {
        "right": ["right", "rightward", "left-to-right", "left to right"],
        "down": ["down", "downward", "top-to-bottom", "top to bottom", "downwards"],
        "left": ["left", "leftward", "right-to-left"],
        "up": ["up", "upward", "bottom-to-top", "upwards"],
    }
    motion_terms = ["moving", "moves", "moved", "translates", "across",
                    "traveling", "travels", "shift", "sliding", "displacement",
                    "motion", "trajectory"]

    correct_hit = any(w in t for w in direction_words.get(want_direction, []))
    wrong_hit = any(
        any(w in t for w in words)
        for d, words in direction_words.items()
        if d != want_direction
    )
    motion_hit = any(w in t for w in motion_terms)

    if correct_hit and not wrong_hit:
        return "CORRECT"
    if correct_hit and wrong_hit:
        # Model hedged ("possibly right, possibly down") — partial credit
        return "AMBIGUOUS"
    if wrong_hit or motion_hit:
        return "WRONG_DIRECTION"
    return "NO_MOTION"


def _run_one(host_port: str, model: str, direction: str, n_frames: int, max_tokens: int) -> tuple[str, str]:
    print("=" * 60)
    print(f"PROBE: red circle moving {direction}")
    frames = make_frames(direction, n_frames=n_frames)
    prompt = (
        f"These {n_frames} frames are sampled in chronological order from a short video. "
        "In which direction does the red circle move? Answer with one word: left, right, up, or down."
    )
    finish, content, usage = _call(host_port, model, frames, prompt, max_tokens)
    print(f"finish={finish}  usage={usage}")
    print(f"--- response (first 400 chars) ---")
    print(content[:400])
    print()
    cls = _classify(content, direction)
    print(f"classification: {cls}")
    return cls, content


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--model", default="default")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    host_port = f"localhost:{args.port}"
    h_cls, _ = _run_one(host_port, args.model, "right", args.frames, args.max_tokens)
    print()
    v_cls, _ = _run_one(host_port, args.model, "down", args.frames, args.max_tokens)
    print()

    # Verdict
    corrects = sum(1 for c in (h_cls, v_cls) if c == "CORRECT")
    motion_hits = sum(1 for c in (h_cls, v_cls) if c != "NO_MOTION")

    print("=" * 60)
    print(f"  horizontal: {h_cls}")
    print(f"  vertical:   {v_cls}")
    print()
    if corrects == 2:
        verdict, rc = "STRONG", 0
    elif corrects == 1 and motion_hits == 2:
        verdict, rc = "PARTIAL", 1
    elif motion_hits == 2:
        verdict, rc = "DEGRADED", 1
    else:
        verdict, rc = "FAIL", 2
    print(f"VERDICT: {verdict}")
    print()
    print("Notes:")
    print("- STRONG = both direction probes (right + down) correctly identified.")
    print("- PARTIAL = one direction correct, the other mentions motion.")
    print("- DEGRADED = both mention motion but wrong direction (hallucination from priors).")
    print("- FAIL = at least one probe lacks any motion language (image-token misroute).")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
