#!/usr/bin/env python3
"""The validator's greedy thinking check, run while other requests decode.

validate_capabilities.py sends one reasoning prompt at temperature 0 and fails
it when the response runs to max_tokens. Alone, that tells you about one
greedy trajectory. This probe sends the same request alone, then again while
1 and 3 filler streams are decoding, then once more with the fillers arriving
after it has started, and reports for each run where the response stops and
how far it follows the solo run. Each run also returns the log-probability of
every token it chose, and the table gives the mean and largest difference
from the solo run over the tokens both runs share: text that parts ways says
only that a near-tie fell the other way, the log-probabilities say how far
apart the two paths were while they still agreed.

Modes: ``solo``, ``solo-repeat``, ``with-K`` (K fillers decoding first),
``with-K-late`` (fillers start 4 s after the request), ``with-K-late@S``
(S seconds after). ``mode*N`` runs a mode N times. With ``--temperature``
above 0 (and ``--top-p``, ``--top-k``) the reasoning request is sampled, the
comparison with the solo run means nothing, and the summary by mode is the
result: how many of N draws stop with the answer and their median length.

Use it to attribute a thinking miss under load: a serial arm (one forward per
request) should repeat the solo text; a batched arm (patch 035) follows it
only as far as bf16 rounding lets a greedy trajectory go, and what matters is
whether the check still terminates with the right answer.

    python scripts/eval/probe_thinking_under_load.py --port 23334 --tag arm --out arm.json
"""
import argparse
import http.client
import json
import re
import threading
import time

THINK_PROMPT = (
    "A ball and a bat cost $1.10 together. The bat costs $1.00 more than "
    "the ball. How much does the ball cost? Put the final numeric answer "
    "alone on the last line."
)
FILLER_TOPICS = [
    "the history of the steam engine from Newcomen to the turbine",
    "how a compiler turns a loop into machine code, stage by stage",
    "the water cycle and its role in shaping river valleys",
]


def post(host, port, payload, timeout):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                 {"Content-Type": "application/json"})
    return conn, conn.getresponse()


class Filler(threading.Thread):
    """One streaming request that keeps decoding until it is closed."""

    def __init__(self, host, port, model, topic, max_tokens):
        super().__init__(daemon=True)
        self.args = (host, port)
        self.payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"Write a long, detailed essay on {topic}."}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        self.started = threading.Event()
        self.stop = threading.Event()
        self.chunks = 0
        self.conn = None

    def run(self):
        try:
            self.conn, resp = post(*self.args, self.payload, 900)
            for line in resp:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    self.chunks += 1
                    self.started.set()
                if self.stop.is_set():
                    break
        except Exception:
            pass
        finally:
            self.started.set()
            try:
                self.conn.close()
            except Exception:
                pass


def think(host, port, model, max_tokens, sampling=None):
    payload = {"model": model, "messages": [{"role": "user", "content": THINK_PROMPT}],
               "max_tokens": max_tokens, "temperature": 0, "skip_special_tokens": False,
               "logprobs": True, "top_logprobs": 1}
    payload.update(sampling or {})
    t0 = time.time()
    conn, resp = post(host, port, payload, 1800)
    r = json.loads(resp.read())
    conn.close()
    choice = r["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or ""
    tail = content.split("</think>")[-1]
    lp = (choice.get("logprobs") or {}).get("content") or []
    lines = [l.strip() for l in (reasoning + "\n" + content).splitlines() if l.strip()]
    top = max(set(lines), key=lines.count) if lines else ""
    return {
        "completion_tokens": r.get("usage", {}).get("completion_tokens", 0),
        "finish": choice.get("finish_reason"),
        "answer_ok": bool(re.search(r"\$?0?\.05\b|\b5\s*cents?\b", tail.lower())),
        "closed": bool(reasoning and content) or "</think>" in content,
        "lines": len(lines), "unique_lines": len(set(lines)),
        "most_repeated_line_count": lines.count(top) if lines else 0,
        "most_repeated_line": top[:120],
        "seconds": round(time.time() - t0, 1),
        "reasoning": reasoning, "content": content,
        "tokens": [t.get("token") for t in lp], "logprobs": [t.get("logprob") for t in lp],
    }


def equal_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--filler-tokens", type=int, default=6000)
    p.add_argument("--modes", default="solo,solo-repeat,with-1,with-3,with-3-late")
    p.add_argument("--temperature", type=float, default=0.0, help="sample the reasoning request (fillers stay greedy)")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    conn = http.client.HTTPConnection(a.host, a.port, timeout=30)
    conn.request("GET", "/v1/models")
    model = json.loads(conn.getresponse().read())["data"][0]["id"]
    conn.close()

    sampling = {}
    if a.temperature > 0:
        sampling = {"temperature": a.temperature}
        if a.top_p is not None:
            sampling["top_p"] = a.top_p
        if a.top_k is not None:
            sampling["top_k"] = a.top_k
    modes = []
    for m in a.modes.split(","):
        m, _, times = m.partition("*")
        modes += [m] * (int(times) if times else 1)

    runs = []
    solo_text = solo = None
    for mode in modes:
        name, _, delay = mode.partition("@")
        k = int(name.split("-")[1]) if name.startswith("with-") else 0
        late = name.endswith("-late")
        delay = float(delay) if delay else 4.0
        fillers = [Filler(a.host, a.port, model, FILLER_TOPICS[i], a.filler_tokens) for i in range(k)]
        box = {}
        if late:
            t = threading.Thread(target=lambda: box.update(think(a.host, a.port, model, a.max_tokens, sampling)))
            t.start()
            time.sleep(delay)
            for f in fillers:
                f.start()
            t.join()
        else:
            for f in fillers:
                f.start()
            for f in fillers:
                f.started.wait(300)
            box.update(think(a.host, a.port, model, a.max_tokens, sampling))
        alive = sum(1 for f in fillers if f.is_alive())
        for f in fillers:
            f.stop.set()
            try:
                f.conn.close()
            except Exception:
                pass
        for f in fillers:
            f.join(30)
        time.sleep(5.0)
        text = box["reasoning"] + "\x00" + box["content"]
        if solo_text is None:
            solo_text, solo = text, box
        # Log-probability of the chosen token, compared over the tokens this
        # run shares with the solo run.
        n = 0
        for x, y in zip(box["tokens"], solo["tokens"]):
            if x != y:
                break
            n += 1
        d = [abs(x - y) for x, y in zip(box["logprobs"][:n], solo["logprobs"][:n])]
        box.update(shared_tokens=n, dlogp_mean=sum(d) / len(d) if d else None,
                   dlogp_max=max(d) if d else None, dlogp_nonzero=sum(1 for v in d if v > 0))
        box.update(mode=mode, fillers=k, fillers_alive_at_end=alive,
                   chars=len(text) - 1, equal_prefix_with_solo=min(equal_prefix(text, solo_text), len(text) - 1))
        box["passed"] = box["finish"] != "length" and box["closed"] and box["answer_ok"]
        runs.append(box)
        print(f"  {mode}: {box['completion_tokens']} tok finish={box['finish']} "
              f"{'PASS' if box['passed'] else 'FAIL'}", flush=True)

    print(f"\nmodel {model}  tag {a.tag}")
    print("mode              fillers  alive at end  tokens  finish  answer  chars  equal prefix  shared tok  dlogp mean  dlogp max  nonzero  top line x  verdict")
    for r in runs:
        dm = "-" if r["dlogp_mean"] is None else f"{r['dlogp_mean']:.5f}"
        dx = "-" if r["dlogp_max"] is None else f"{r['dlogp_max']:.5f}"
        print(f"{r['mode']:17s} {r['fillers']:7d} {r['fillers_alive_at_end']:13d} {r['completion_tokens']:7d}  "
              f"{r['finish']:6s}  {'ok' if r['answer_ok'] else 'NO':6s} {r['chars']:6d} {r['equal_prefix_with_solo']:13d} "
              f"{r['shared_tokens']:11d} {dm:>11s} {dx:>10s} {r['dlogp_nonzero']:8d} {r['most_repeated_line_count']:11d}  "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    if len(set(modes)) < len(modes):
        print("\nmode              runs  stop with the answer  at the token cap  median tokens  min  max")
        for m in dict.fromkeys(modes):
            rs = [r for r in runs if r["mode"] == m]
            toks = sorted(r["completion_tokens"] for r in rs)
            print(f"{m:17s} {len(rs):4d} {sum(1 for r in rs if r['passed']):21d} "
                  f"{sum(1 for r in rs if r['finish'] == 'length'):17d} {toks[len(toks) // 2]:14d} {toks[0]:4d} {toks[-1]:4d}")
    failed = [r["mode"] for r in runs if not r["passed"]]
    print("FAIL: " + ", ".join(failed) if failed else "PASS")
    if a.out:
        json.dump({"model": model, "tag": a.tag, "max_tokens": a.max_tokens, "sampling": sampling, "runs": runs},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
