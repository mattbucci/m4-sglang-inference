#!/usr/bin/env python3
"""Edit-tool probe: a tool call that has to quote the context back.

The request is the shape a coding-agent harness sends: a system line, a file
in the user turn, one `edit` tool. A reply passes when it is a tool call
whose arguments parse, whose `old` text is an exact substring of the file and
names the right function, and whose `new` text carries the requested line.
The validator's tool check only asks for a one-word argument; this one fails
on a parser or a model that escapes newlines in the arguments, which is what
makes an agent's edit miss its target.

No sampling field is sent unless --sampling gives one, so the default run
exercises the server's default sampling as an agent harness does.

  STRONG   every reply passes
  PARTIAL  some replies pass
  FAIL     no reply passes

Usage:
  python scripts/eval/probe_tool_edit.py [--port PORT] [--n N] [--no-think]
      [--sampling '{"temperature": 0}'] [--max-tokens N] [--out receipt.json]
"""
import argparse
import json
import sys
import urllib.request

CODE = "\n".join(
    f"def handler_{i}(request, response):\n    response.status = {200 + i}\n    return response"
    for i in range(12)
)
TOOL = {"type": "function", "function": {
    "name": "edit", "description": "Replace old text with new text in a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        "required": ["path", "old", "new"]}}}


def grade(choice, usage):
    msg = choice["message"]
    calls = msg.get("tool_calls") or []
    row = {"finish": choice["finish_reason"], "completion_tokens": usage["completion_tokens"],
           "reasoning_chars": len(msg.get("reasoning_content") or ""),
           "content_chars": len(msg.get("content") or ""),
           "tool_calls": [c["function"]["name"] for c in calls], "ok": False}
    if not calls:
        return row
    try:
        args = json.loads(calls[0]["function"]["arguments"])
        old, new = args.get("old", ""), args.get("new", "")
    except (ValueError, AttributeError) as e:
        row["error"] = repr(e)[:120]
        return row
    row["argument_keys"] = sorted(args)
    row["path"] = args.get("path")
    row["old_in_file"] = old in CODE and "handler_3" in old
    row["new_has_line"] = "response.reason" in new
    # A literal backslash-n where the file has a newline: escaped once too often.
    row["escaped_newlines"] = "\\n" in old and old.replace("\\n", "\n") in CODE
    row["ok"] = bool(row["old_in_file"] and row["new_has_line"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--sampling", default="", help="JSON of sampling fields; nothing is sent when empty")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    body = {
        "model": a.model,
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use the tools."},
            {"role": "user", "content": "Here is src/handlers.py:\n\n" + CODE + "\n\nRewrite handler_3 so it "
             "also sets response.reason = 'ok', by calling the edit tool with the old and the new text of "
             "that function."},
        ],
        "tools": [TOOL],
        "max_tokens": a.max_tokens,
    }
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if a.sampling:
        body.update(json.loads(a.sampling))

    rows = []
    for _ in range(a.n):
        req = urllib.request.Request(
            f"http://{a.host}:{a.port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        out = json.load(urllib.request.urlopen(req, timeout=3600))
        rows.append(grade(out["choices"][0], out["usage"]))
        print(json.dumps(rows[-1]))
    passed = sum(r["ok"] for r in rows)
    verdict = "STRONG" if passed == len(rows) else ("PARTIAL" if passed else "FAIL")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"no_think": a.no_think, "max_tokens": a.max_tokens,
                       "sampling": a.sampling or None, "verdict": verdict, "rows": rows}, fh, indent=1)
    print(f"TOOL EDIT {passed}/{len(rows)}")
    print(f"VERDICT: {verdict}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
