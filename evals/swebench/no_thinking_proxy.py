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
        await resp.prepare(request)

        try:
            async for chunk in upstream_resp.aiter_raw():
                await resp.write(chunk)
        finally:
            await upstream_resp.aclose()

        await resp.write_eof()
        return resp


def main() -> int:
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", forward)
    print(f"no_thinking_proxy: listening on http://127.0.0.1:{PORT} -> {UPSTREAM}", file=sys.stderr)
    print(f"  injects chat_template_kwargs={{\"enable_thinking\": false}} on POST /v1/chat/completions", file=sys.stderr)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
