#!/usr/bin/env python3
"""Forwards HTTP requests to the M4 SGLang server (port 23334) while injecting
`chat_template_kwargs={"enable_thinking": false}` into every /v1/chat/completions
body.

Why this exists: opencode's `@ai-sdk/openai-compatible` provider doesn't expose
a `chat_template_kwargs` body extension. SGLang's Qwen3-family chat template
emits `<think>...</think>` blocks unless `enable_thinking=false` is set. Those
blocks either (a) go to `reasoning_content` (with `--reasoning-parser qwen3`)
where opencode's ai-sdk client doesn't see them, or (b) leak as raw `</think>`
markers into the content stream (without the parser). Both modes break the
agent loop on SWE-bench rollouts.

Putting this proxy in front of the server makes the model behave as if
opencode passed `enable_thinking=false` itself.

BACKFILL_BASH_DESCRIPTION=1 adds a second, model-specific repair: opencode's
`bash` tool requires a `description` argument (5-10 words) next to `command`,
and Recursive Flint -- trained on single-argument shell tools -- omits it on
nearly every call under opencode's real prompt (0-3 of ~12 calls across every
rule placement tried), so opencode rejects the call with a SchemaError and
the agent loops. The proxy buffers each streamed `bash` tool call's arguments
and, when the call closes without `description`, inserts one derived from the
command before forwarding the arguments as a single delta. Other tools and
calls that already carry `description` pass through untouched.

Usage:
    python evals/swebench/no_thinking_proxy.py           # listen on 23335, upstream 23334
    PORT=23336 UPSTREAM=http://127.0.0.1:23334 python evals/swebench/no_thinking_proxy.py

Then point opencode at http://127.0.0.1:23335/v1.
"""
import asyncio
import json
import os
import sys

import httpx
from aiohttp import web

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:23334")
PORT = int(os.environ.get("PORT", "23335"))
BACKFILL_BASH_DESCRIPTION = os.environ.get("BACKFILL_BASH_DESCRIPTION", "0") == "1"
BACKFILL_TOOLS = {"bash"}


def _derive_description(arguments: dict) -> str:
    command = str(arguments.get("command", "")).strip().split("\n", 1)[0]
    words = command.split()
    return ("Run: " + " ".join(words[:8]))[:80] if words else "Run a shell command"


def _backfill_arguments(name: str, arguments: str) -> str:
    """Return *arguments* (a JSON string) with `description` added for a
    buffered tool call that lacks it; unparsable arguments pass through."""
    if name not in BACKFILL_TOOLS:
        return arguments
    try:
        parsed = json.loads(arguments)
    except Exception:
        return arguments
    if not isinstance(parsed, dict) or parsed.get("description"):
        return arguments
    parsed["description"] = _derive_description(parsed)
    return json.dumps(parsed, ensure_ascii=False)


class _ToolCallBackfill:
    """Rewrites an OpenAI chat-completion SSE stream so each `bash` tool call's
    arguments arrive as one delta that includes `description`.

    Argument deltas of a buffered call are held back; they are released
    (rewritten if needed) when the call closes: a delta for another index,
    a finish_reason, or the end of the stream."""

    def __init__(self):
        self.calls: dict[tuple[int, int], dict] = {}  # (choice, index) -> {name, id, type, args}
        self.open: tuple[int, int] | None = None
        self.pending = b""

    def _release(self, key) -> list[bytes]:
        call = self.calls.pop(key, None)
        if call is None or call["released"]:
            return []
        call["released"] = True
        if not call["args"]:
            return []
        args = _backfill_arguments(call["name"] or "", call["args"])
        chunk = dict(call["template"])
        chunk["choices"] = [{
            "index": key[0],
            "delta": {"tool_calls": [{"index": key[1], "function": {"arguments": args}}]},
            "finish_reason": None,
        }]
        return [b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"]

    def feed(self, data: bytes) -> bytes:
        self.pending += data
        out: list[bytes] = []
        while b"\n\n" in self.pending:
            event, self.pending = self.pending.split(b"\n\n", 1)
            out.extend(self._event(event))
        return b"".join(out)

    def finish(self) -> bytes:
        out: list[bytes] = []
        if self.pending.strip():
            out.extend(self._event(self.pending))
            self.pending = b""
        if self.open is not None:
            out.extend(self._release(self.open))
            self.open = None
        return b"".join(out)

    def _event(self, event: bytes) -> list[bytes]:
        raw = event + b"\n\n"
        line = event.strip()
        if not line.startswith(b"data:"):
            return [raw]
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return self._flush_open() + [raw]
        try:
            chunk = json.loads(payload)
        except Exception:
            return [raw]
        out: list[bytes] = []
        choices = chunk.get("choices") or []
        if not choices:
            return [raw]
        passthrough = True
        for choice in choices:
            cidx = choice.get("index", 0)
            delta = choice.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                key = (cidx, tc.get("index", 0))
                fn = tc.get("function") or {}
                if key not in self.calls:
                    if self.open is not None and self.open != key:
                        out.extend(self._release(self.open))
                    self.calls[key] = {"name": fn.get("name"), "args": "", "released": False,
                                       "template": {k: v for k, v in chunk.items() if k != "choices"}}
                    self.open = key
                    if self.calls[key]["name"] in BACKFILL_TOOLS:
                        # Forward the head (id/type/name) now, hold the arguments.
                        head = dict(tc)
                        head["function"] = {k: v for k, v in fn.items() if k != "arguments"}
                        head["function"]["arguments"] = ""
                        hchunk = dict(chunk)
                        hchunk["choices"] = [{"index": cidx, "delta": {"tool_calls": [head]}, "finish_reason": None}]
                        out.append(b"data: " + json.dumps(hchunk, ensure_ascii=False).encode() + b"\n\n")
                        self.calls[key]["args"] += fn.get("arguments") or ""
                        passthrough = False
                    continue
                call = self.calls[key]
                if fn.get("name"):
                    call["name"] = fn["name"]
                if call["name"] in BACKFILL_TOOLS and not call["released"]:
                    call["args"] += fn.get("arguments") or ""
                    passthrough = False
            if choice.get("finish_reason"):
                out.extend(self._flush_open())
        if passthrough:
            out.append(raw)
        elif any(c.get("finish_reason") for c in choices) or any(
            k for k in (choices[0].get("delta") or {}) if k not in ("tool_calls", "role")
        ):
            # Keep finish/usage/content information when the chunk also carried held arguments.
            stripped = dict(chunk)
            stripped["choices"] = [
                {**c, "delta": {k: v for k, v in (c.get("delta") or {}).items() if k != "tool_calls"}}
                for c in choices
            ]
            out.append(b"data: " + json.dumps(stripped, ensure_ascii=False).encode() + b"\n\n")
        return out

    def _flush_open(self) -> list[bytes]:
        if self.open is None:
            return []
        out = self._release(self.open)
        self.open = None
        return out


def _backfill_json_body(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
        for choice in payload.get("choices") or []:
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                fn["arguments"] = _backfill_arguments(fn.get("name", ""), fn.get("arguments") or "")
        return json.dumps(payload, ensure_ascii=False).encode()
    except Exception:
        return body


def _inject_no_thinking(body_bytes: bytes) -> bytes:
    """Add chat_template_kwargs.enable_thinking=false to a JSON body."""
    try:
        payload = json.loads(body_bytes)
    except Exception:
        return body_bytes
    if not isinstance(payload, dict):
        return body_bytes
    ctk = payload.setdefault("chat_template_kwargs", {})
    if not isinstance(ctk, dict):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        ctk["enable_thinking"] = False
    return json.dumps(payload).encode()


async def forward(request: web.Request) -> web.StreamResponse:
    upstream_url = f"{UPSTREAM}{request.path_qs}"
    method = request.method
    body = await request.read()
    inject = method == "POST" and "/chat/completions" in request.path
    if inject:
        body = _inject_no_thinking(body)

    # Strip hop-by-hop and host headers; httpx sets its own.
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")
    }
    if inject:
        headers["Content-Length"] = str(len(body))

    timeout = httpx.Timeout(connect=10.0, read=900.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream_req = client.build_request(method, upstream_url, content=body, headers=headers)
        try:
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.RequestError as e:
            return web.Response(status=502, text=f"upstream error: {e!r}")

        resp = web.StreamResponse(
            status=upstream_resp.status_code,
            reason=upstream_resp.reason_phrase or "",
        )
        # Forward response headers, stripping hop-by-hop.
        for k, v in upstream_resp.headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            resp.headers[k] = v
        backfill = (
            BACKFILL_BASH_DESCRIPTION
            and inject
            and upstream_resp.status_code == 200
        )
        streaming = "text/event-stream" in (upstream_resp.headers.get("content-type") or "")
        if backfill and not streaming:
            # Non-streaming: rewrite the whole JSON body.
            body_bytes = await upstream_resp.aread()
            await upstream_resp.aclose()
            fixed = _backfill_json_body(body_bytes)
            resp.headers.pop("Content-Length", None)
            await resp.prepare(request)
            await resp.write(fixed)
            await resp.write_eof()
            return resp
        await resp.prepare(request)

        rewriter = _ToolCallBackfill() if (backfill and streaming) else None
        try:
            async for chunk in upstream_resp.aiter_raw():
                await resp.write(rewriter.feed(chunk) if rewriter else chunk)
            if rewriter:
                tail = rewriter.finish()
                if tail:
                    await resp.write(tail)
        finally:
            await upstream_resp.aclose()

        await resp.write_eof()
        return resp


def main() -> int:
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", forward)
    print(f"no_thinking_proxy: listening on http://127.0.0.1:{PORT} -> {UPSTREAM}", file=sys.stderr)
    print(f"  injects chat_template_kwargs={{\"enable_thinking\": false}} on POST /v1/chat/completions", file=sys.stderr)
    if BACKFILL_BASH_DESCRIPTION:
        print("  backfills the bash tool's required `description` argument on tool calls that omit it", file=sys.stderr)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
