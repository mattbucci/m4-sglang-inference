#!/usr/bin/env python3
"""Audio dictation probe — synthesize a known phrase, ask the model to transcribe.

Uses macOS `say` to render a phrase to WAV (no external downloads), sends it
via ``audio_url`` data URL to ``/v1/chat/completions``, and checks that the
model's transcription contains the expected keywords.

Blocked on an audio-enabled checkpoint reaching mlx-community today:
Gemma 4 supports audio architecturally but ``mlx-community/gemma-4-31b-it-mxfp4``
ships without ``preprocessor_config.json``, so SGLang can't load the audio
processor. The harness below runs end-to-end against any future checkpoint
that does ship it — no changes required.

Usage (server already up):
    python scripts/eval/test_audio.py --port 23334
    python scripts/eval/test_audio.py --port 23334 --phrase "testing one two three"
    python scripts/eval/test_audio.py --port 23334 --wav /path/to/input.wav --expected "the quick brown fox"
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_PHRASE = "the quick brown fox jumps over the lazy dog"


def synth_wav(phrase: str, out_path: Path) -> None:
    """Render a spoken phrase to 16 kHz mono WAV via macOS `say`."""
    subprocess.run(
        [
            "say",
            "-o", str(out_path),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            phrase,
        ],
        check=True,
    )


def post(url: str, payload: dict, timeout: int = 120) -> dict:
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
    ap.add_argument("--phrase", default=DEFAULT_PHRASE,
                    help="Phrase to synthesize via macOS `say` when --wav isn't given.")
    ap.add_argument("--wav", default="",
                    help="Path to an input WAV to use instead of synth.")
    ap.add_argument("--expected", default="",
                    help="Keywords we expect in the transcript (comma-separated). "
                         "Defaults to the input phrase's lowercase words.")
    ap.add_argument("--keep-wav", default="",
                    help="Save the input WAV to this path for inspection.")
    ap.add_argument("--prompt", default="Transcribe this audio verbatim. Output the text only.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.wav:
            wav = Path(args.wav)
            if not wav.exists():
                print(f"  ERROR: --wav not found: {wav}")
                return 1
            phrase_for_keywords = args.phrase
        else:
            wav = Path(tmp) / "dictation.wav"
            synth_wav(args.phrase, wav)
            phrase_for_keywords = args.phrase

        if args.keep_wav:
            Path(args.keep_wav).write_bytes(wav.read_bytes())
            print(f"  saved WAV to {args.keep_wav} ({wav.stat().st_size} bytes)")

        expected = (args.expected or phrase_for_keywords).lower()
        keywords = [w.strip() for w in expected.split(",")] if "," in expected \
                   else [w for w in expected.split() if len(w) > 2]

        b64 = base64.b64encode(wav.read_bytes()).decode("ascii")
        data_url = f"data:audio/wav;base64,{b64}"

        payload = {
            "model": "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }],
            "max_tokens": 128,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        url = f"http://localhost:{args.port}/v1/chat/completions"
        try:
            r = post(url, payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  HTTP {e.code}: {body[:500]}")
            return 1

        if "choices" not in r:
            print(f"  ERROR: {json.dumps(r)[:500]}")
            return 1
        msg = r["choices"][0]["message"].get("content") or ""
        fin = r["choices"][0].get("finish_reason")
        print(f"  phrase  : {phrase_for_keywords!r}")
        print(f"  finish  : {fin}")
        print(f"  content : {msg!r}")

        msg_l = msg.lower()
        hits = [k for k in keywords if k in msg_l]
        miss = [k for k in keywords if k not in msg_l]
        if not miss:
            print(f"  PASS — all {len(keywords)} keywords found")
            return 0
        print(f"  PARTIAL — hits={hits} miss={miss}")
        return 1 if not hits else 2  # 0=pass, 1=fail, 2=partial


if __name__ == "__main__":
    sys.exit(main())
