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

BACKFILL_INTENT=1 is the omp-scaffold form of the same repair: omp's tools
all require an `i` intent string, derived here from the call's payload when
the model leaves it out. BACKFILL_BASH_DESCRIPTION=1 is the opencode form: opencode's
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
# BACKFILL_INTENT=1: omp's tools all require an `i` ("concise intent") string
# next to their payload argument (bash.command, read.path, edit.input, ...);
# a model trained on single-argument tools omits it the way Flint omits
# opencode's bash `description`. Every tool call is buffered and `i` is
# derived from the payload when missing.
BACKFILL_INTENT = os.environ.get("BACKFILL_INTENT", "0") == "1"
# REMAP_INTENT=1 (needs BACKFILL_INTENT): when a call carries `i` but not the
# tool's payload argument, the payload is in `i` -- Flint sends omp's hashline
# edit as `{"i": "[file#TAG]\nPUT ..."}` on 115 of 118 edit calls -- so `i`
# moves to the payload key and a fresh intent is derived. Also clamps `i` to
# omp's 200-character limit, which rejects the call otherwise.
REMAP_INTENT = BACKFILL_INTENT and os.environ.get("REMAP_INTENT", "0") == "1"
# INTENT_LAST=1: rewrite the request's tool schemas so omp's `i` (and other
# metadata keys) come last in `properties` and `required`. omp declares `i`
# first; Flint writes its natural first argument into the first declared key,
# so with `i` first the command lands in `i`. The 3090 fixed the same swap in
# its strict tool grammar by moving the metadata keys last; here the schema
# order is what the model sees in the prompt. A request-side fix, so no
# response rewrite for the model to copy into later calls.
INTENT_LAST = os.environ.get("INTENT_LAST", "0") == "1"
TRAILING_KEYS = ("i", "description", "intent", "explanation", "reason", "summary")
OMP_PAYLOAD_KEY = {"edit": "input", "bash": "command", "eval": "code", "grep": "pattern", "read": "path"}
INTENT_MAX = 200
BACKFILL_TOOLS = {"bash"}
INTENT_SOURCES = ("command", "path", "pattern", "input", "code", "op", "content")


def _needs_buffer(name: str | None) -> bool:
    return BACKFILL_INTENT or name in BACKFILL_TOOLS


def _derive_description(arguments: dict) -> str:
    command = str(arguments.get("command", "")).strip().split("\n", 1)[0]
    words = command.split()
    return ("Run: " + " ".join(words[:8]))[:80] if words else "Run a shell command"


def _derive_intent(name: str, arguments: dict) -> str:
    for key in INTENT_SOURCES:
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            head = val.strip().split("\n", 1)[0]
            return f"{name}: {' '.join(head.split()[:8])}"[:80]
    return f"{name} call"


def _backfill_arguments(name: str, arguments: str) -> str:
    """Return *arguments* (a JSON string) with the required metadata argument
    added for a buffered tool call that lacks it (`description` on opencode's
    bash, `i` on every omp tool); unparsable arguments pass through."""
    if not _needs_buffer(name):
        return arguments
    try:
        parsed = json.loads(arguments)
    except Exception:
        return arguments
    if not isinstance(parsed, dict):
        return arguments
    changed = False
    if BACKFILL_BASH_DESCRIPTION and name in BACKFILL_TOOLS and not parsed.get("description"):
        parsed["description"] = _derive_description(parsed)
        changed = True
    if REMAP_INTENT:
        payload_key = OMP_PAYLOAD_KEY.get(name)
        intent = parsed.get("i")
        if payload_key and not parsed.get(payload_key) and isinstance(intent, str) and intent.strip():
            parsed[payload_key] = parsed.pop("i")
            changed = True
        intent = parsed.get("i")
        if isinstance(intent, str) and len(intent) > INTENT_MAX:
            parsed["i"] = intent[: INTENT_MAX - 1] + "…"
            changed = True
    if BACKFILL_INTENT and not parsed.get("i"):
        parsed["i"] = _derive_intent(name, parsed)
        changed = True
    return json.dumps(parsed, ensure_ascii=False) if changed else arguments


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
                    if _needs_buffer(self.calls[key]["name"]):
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
                if _needs_buffer(call["name"]) and not call["released"]:
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


def _intent_last(params):
    """Move TRAILING_KEYS to the end of a JSON-schema object's `properties`
    and `required`, keeping every other key in place; returns True if changed."""
    if not isinstance(params, dict):
        return False
    props = params.get("properties")
    if not isinstance(props, dict):
        return False
    trailing = [k for k in props if k in TRAILING_KEYS]
    if not trailing or list(props)[-len(trailing):] == trailing:
        changed = False
    else:
        params["properties"] = {**{k: v for k, v in props.items() if k not in TRAILING_KEYS},
                                **{k: props[k] for k in trailing}}
        changed = True
    req = params.get("required")
    if isinstance(req, list):
        new_req = [k for k in req if k not in TRAILING_KEYS] + [k for k in req if k in TRAILING_KEYS]
        if new_req != req:
            params["required"] = new_req
            changed = True
    return changed


def _inject_no_thinking(body_bytes: bytes) -> bytes:
    """Add chat_template_kwargs.enable_thinking=false to a JSON body; with
    INTENT_LAST, also move the metadata keys last in every tool schema."""
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
    if INTENT_LAST:
        for tool in payload.get("tools") or []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict):
                _intent_last(fn.get("parameters"))
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
            (BACKFILL_BASH_DESCRIPTION or BACKFILL_INTENT)
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
    if BACKFILL_INTENT:
        print("  backfills omp's required `i` (intent) argument on tool calls that omit it", file=sys.stderr)
    if REMAP_INTENT:
        print("  moves a payload sent as `i` to the tool's own argument and clamps `i` to 200 chars", file=sys.stderr)
    if INTENT_LAST:
        print("  moves `i` and other metadata keys last in every tool schema (request side)", file=sys.stderr)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
