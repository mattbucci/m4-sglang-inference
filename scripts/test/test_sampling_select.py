#!/usr/bin/env python3
"""Unit sanity for MlxModelRunner._select_tokens (no server, CPU-only).

Asserts: all-greedy batches are exact argmax; sampled rows only land on
tokens the filters allow; heterogeneous batches keep greedy rows argmax
with no cross-row bleed; top_k==1 registration aliases to greedy.
"""
import os
import sys

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


class _Stub:
    """Bind the real selection methods to a minimal state container."""

    register_sampling = MlxModelRunner.register_sampling
    _sampler_for = MlxModelRunner._sampler_for
    _select_tokens = MlxModelRunner._select_tokens

    def __init__(self):
        self._req_sampling = {}
        self._sampler_cache = {}


class _Params:
    def __init__(self, temperature=1.0, top_p=1.0, top_k=1 << 30, min_p=0.0):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p


def _logits(rows):
    return mx.array(rows, dtype=mx.float32)


def main() -> int:
    mx.random.seed(1234)
    failures = []

    def check(name, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures.append(name)

    V = 16
    base = [0.0] * V
    # row where token 3 dominates, tokens 3/7/11 are the only plausible set
    row = list(base)
    row[3], row[7], row[11] = 8.0, 6.0, 5.0

    # 1. all-greedy == argmax
    s = _Stub()
    out = s._select_tokens(_logits([row, row]), ["a", "b"])
    check("all-greedy equals argmax",
          out.tolist() == mx.argmax(_logits([row, row]), axis=-1).tolist())

    # 2. top_k==1 registration aliases to greedy (never registered)
    s = _Stub()
    s.register_sampling("a", _Params(temperature=0.7, top_k=1))
    check("top_k==1 aliases to greedy", "a" not in s._req_sampling)

    # 3. homogeneous sampled: top_k=3 keeps draws inside {3, 7, 11}
    s = _Stub()
    s.register_sampling("a", _Params(temperature=1.0, top_k=3))
    s.register_sampling("b", _Params(temperature=1.0, top_k=3))
    allowed = {3, 7, 11}
    ok = True
    for _ in range(50):
        toks = s._select_tokens(_logits([row, row]), ["a", "b"]).tolist()
        ok = ok and all(t in allowed for t in toks)
    check("homogeneous top_k=3 stays in the top-3 set", ok)

    # 4. heterogeneous: greedy row exact argmax every draw, sampled row in set
    s = _Stub()
    s.register_sampling("samp", _Params(temperature=1.0, top_k=3))
    ok_greedy, ok_samp = True, True
    for _ in range(50):
        toks = s._select_tokens(_logits([row, row]), ["greedy", "samp"]).tolist()
        ok_greedy = ok_greedy and toks[0] == 3
        ok_samp = ok_samp and toks[1] in allowed
    check("heterogeneous greedy row stays argmax", ok_greedy)
    check("heterogeneous sampled row stays in set", ok_samp)

    # 5. tight top_p nucleus: token 3 carries ~88% mass -> top_p=0.5 keeps it alone
    s = _Stub()
    s.register_sampling("a", _Params(temperature=1.0, top_p=0.5))
    ok = all(
        s._select_tokens(_logits([row]), ["a"]).tolist() == [3] for _ in range(30)
    )
    check("top_p=0.5 nucleus collapses to the dominant token", ok)

    # 6. sampling actually varies at high temperature (unfiltered)
    s = _Stub()
    s.register_sampling("a", _Params(temperature=2.0))
    draws = {tuple(s._select_tokens(_logits([row]), ["a"]).tolist())
             for _ in range(30)}
    check("temperature=2.0 produces varied draws", len(draws) > 1)

    print(f"== {'FAIL' if failures else 'PASS'}: "
          f"{6 + 1 - len(failures) - 1}/6 groups ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
