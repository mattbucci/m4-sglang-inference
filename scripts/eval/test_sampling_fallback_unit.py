#!/usr/bin/env python3
"""Unit test for the sampling-defaults fallback (patch 041). No server, no model.

A checkpoint that ships no generation_config.json gives the server nothing to
fill an unset request field with, so a chat request that sends no sampling
fields decodes at temperature 1.0 over the whole vocabulary.
``--sampling-defaults-fallback`` supplies the defaults for such a checkpoint,
and only for such a checkpoint.

  1. Resolution: the fallback applies when the checkpoint has no generation
     config, a checkpoint's own config wins, ``--sampling-defaults openai``
     ignores both, and an unsupported key is refused at boot.
  2. Request: an unset chat field takes the fallback, a sent field wins,
     temperature 0 included.
  3. Argument: registered in the serving namespace and its field order.
  4. Launcher: every preset whose cached checkpoint ships no generation
     config passes the argument, and the JSON it passes parses.

  python scripts/eval/test_sampling_fallback_unit.py
"""
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

from transformers import GenerationConfig

from sglang.srt.arg_groups import field_order
from sglang.srt.arg_groups.fields.serving import Serving
from sglang.srt.configs.model_config import (
    ModelConfig,
    _parse_sampling_defaults_fallback,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QWEN3 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
fails = []
count = 0


def check(name, cond, detail=""):
    global count
    count += 1
    if not cond:
        fails.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + str(detail)) if detail and not cond else ''}")


def resolve(*, mode, gen, fallback):
    stub = SimpleNamespace(
        sampling_defaults=mode,
        hf_generation_config=gen,
        sampling_defaults_fallback=_parse_sampling_defaults_fallback(fallback),
    )
    return ModelConfig.get_default_sampling_params(stub)


print("1. resolution")
check("no generation config, no fallback: nothing", resolve(mode="model", gen=None, fallback=None) == {})
check("no generation config: the fallback", resolve(mode="model", gen=None, fallback=dict(QWEN3)) == QWEN3)
check("a JSON string parses to the same", resolve(mode="model", gen=None, fallback=json.dumps(QWEN3)) == QWEN3)
shipped = GenerationConfig(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
got = resolve(mode="model", gen=shipped, fallback=dict(QWEN3))
check("a shipped generation config wins", got.get("temperature") == 0.7 and got.get("top_p") == 0.8, got)
check("a shipped config keeps its own extra fields", got.get("repetition_penalty") == 1.05, got)
check("openai mode ignores the fallback", resolve(mode="openai", gen=None, fallback=dict(QWEN3)) == {})
check("openai mode ignores a shipped config", resolve(mode="openai", gen=shipped, fallback=None) == {})
first = resolve(mode="model", gen=None, fallback=dict(QWEN3))
first["temperature"] = 9.0
check("the returned dict is a copy", resolve(mode="model", gen=None, fallback=dict(QWEN3)) == QWEN3)
try:
    _parse_sampling_defaults_fallback({"temperature": 0.6, "max_tokens": 5})
    refused = False
except ValueError as e:
    refused = "max_tokens" in str(e)
check("an unsupported key is refused", refused)
check("an empty string is no fallback", _parse_sampling_defaults_fallback("") == {})

print("2. request")
msgs = [{"role": "user", "content": "hi"}]
plain = ChatCompletionRequest(model="m", messages=msgs)
p = plain.to_sampling_params(stop=[], model_generation_config=dict(QWEN3))
check("an unset request takes the fallback", (p["temperature"], p["top_p"], p["top_k"]) == (0.6, 0.95, 20), p)
p = plain.to_sampling_params(stop=[], model_generation_config={})
check("no fallback leaves temperature 1.0 and no truncation", (p["temperature"], p["top_p"], p["top_k"]) == (1.0, 1.0, -1), p)
greedy = ChatCompletionRequest(model="m", messages=msgs, temperature=0)
p = greedy.to_sampling_params(stop=[], model_generation_config=dict(QWEN3))
check("temperature 0 from the request wins", p["temperature"] == 0, p)
check("the other unset fields still take the fallback", (p["top_p"], p["top_k"]) == (0.95, 20), p)
sent = ChatCompletionRequest(model="m", messages=msgs, temperature=1.0, top_k=5, top_p=0.5)
p = sent.to_sampling_params(stop=[], model_generation_config=dict(QWEN3))
check("sent fields win", (p["temperature"], p["top_p"], p["top_k"]) == (1.0, 0.5, 5), p)

print("3. argument")
check("field registered with default None", Serving.__struct_defaults__[Serving.__struct_fields__.index("sampling_defaults_fallback")] is None)
order_src = open(field_order.__file__).read()
check("field order lists it after sampling_defaults", '"sampling_defaults",\n    "sampling_defaults_fallback",' in order_src)

print("4. launcher")
launch = open(os.path.join(REPO, "scripts", "launch.sh")).read()
m = re.search(r"^QWEN3_SAMPLING='([^']+)'", launch, re.M)
check("the launcher's Qwen3 defaults parse", bool(m) and json.loads(m.group(1)) == QWEN3, m and m.group(1))
check("the launcher passes the argument", '--sampling-defaults-fallback "$SAMPLING_FALLBACK"' in launch)
hub = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub")
blocks = re.findall(r"^        ([a-z0-9-]+)\)\n(.*?)^            ;;", launch, re.M | re.S)
seen = 0
for preset, body in blocks:
    repo = re.search(r'MODEL="\$\{MODEL:-([^}]+)\}"', body)
    if not repo:
        continue
    snaps = glob.glob(os.path.join(hub, "models--" + repo.group(1).replace("/", "--"), "snapshots", "*"))
    if not snaps:
        continue
    seen += 1
    if any(os.path.exists(os.path.join(s, "generation_config.json")) for s in snaps):
        continue
    check(f"{preset}: no generation config in {repo.group(1)}, fallback set", "SAMPLING_FALLBACK=" in body)
print(f"  ({seen} presets with a cached checkpoint inspected)")

print(f"\n{count - len(fails)}/{count} PASS")
sys.exit(1 if fails else 0)
