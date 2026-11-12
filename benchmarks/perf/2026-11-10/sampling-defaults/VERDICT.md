# Sampling defaults for a checkpoint without a generation config (patch 041)

Run record, 2026-11-11, M4 Pro 64 GB, SGLang v0.5.23 + patches 001 to 041, MLX backend.

## Outcome

- A chat request that sends no sampling fields decoded at temperature 1.0 with no
  truncation on three presets: `qwen3-32b`, `qwen3-moe`, `qwen35-9b-8bit`. Their
  checkpoints ship no `generation_config.json`.
- On `qwen3-32b` that default returns token soup. The default-sampling probe scores 0/8
  there and 8/8 with the fallback. Every existing gate passed throughout, because every
  gate sends temperature 0.
- `qwen3-moe` and `qwen35-9b-8bit` are coherent at the old default (8/8 in both arms).
  The fallback there only moves the default to the base model's recommended sampling.
- The omp scaffold sends no sampling fields, so this is the path an agentic turn takes.
- Patch 041 adds `--sampling-defaults-fallback`; `launch.sh` passes
  `{"temperature": 0.6, "top_p": 0.95, "top_k": 20}` for the three presets.
- The `Qwen3-32B-4bit-DWQ` checkpoint has a second, separate weakness on tool calls
  (below). The fallback does not fix that one.

## Root cause

1. `--sampling-defaults` defaults to `model`.
2. `ModelConfig.get_default_sampling_params()` returns an empty dict when the checkpoint
   has no `generation_config.json`.
3. `ChatCompletionRequest.to_sampling_params` resolves each field as: the request's value,
   then the model's generation config, then the OpenAI default. With an empty dict that is
   temperature 1.0, top-p 1.0, top-k off.

The base models' generation configs, which the conversions dropped:

| Base model | generation config |
|---|---|
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | temperature 0.6, top_k 20, top_p 0.95 |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | temperature 0.6, top_k 20, top_p 0.95 |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | none shipped upstream; the model card recommends 0.6 / 0.95 / 20 for precise coding with thinking |
| [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (reference) | temperature 0.6, top_k 20, top_p 0.95 |

The sister rigs build their own checkpoints and write the file into the model directory.
This rig serves community checkpoints out of the Hub cache, where an added file does not
survive a re-download, so the launcher carries the values instead.

## What the untruncated default samples from

`scripts/eval/check_mlx_logit_tail.py` loads a checkpoint raw, teacher-forces a prose
sentence and a code line after a chat prompt, and reports the probability beyond the top
1000 tokens. That mass is what temperature 1.0 without top-k or top-p can draw from.

| Checkpoint | Preset | generation config | Mean mass beyond top 1000, T 1.0 | Worst position | Mean at T 0.6 | Max minus median logit | Verdict |
|---|---|---|---|---|---|---|---|
| `Qwen3-32B-4bit-DWQ` | `qwen3-32b` | **missing** | 0.3429 | 0.6199 | 0.0005 | 12.6 | HEAVY TAIL |
| `Qwen3-30B-A3B-4bit-DWQ` | `qwen3-moe` | **missing** | 0.0001 | 0.0017 | 0.0000 | 26.6 | OK |
| `Qwen3.5-9B-MLX-8bit` | `qwen35-9b-8bit` | **missing** | 0.0001 | 0.0020 | 0.0000 | 31.1 | OK |
| `Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` | `coder-30b` | present | 0.0000 | 0.0000 | 0.0000 | 45.4 | OK |
| `Qwen3.5-27B-4bit` | `qwen35` | present | 0.0000 | 0.0001 | 0.0000 | 31.0 | OK |
| `Qwen3.6-27B-4bit` | `qwen36-27b` | present | 0.0000 | 0.0002 | 0.0000 | 27.3 | OK |
| `Qwen3.6-35B-A3B-4bit` | `qwen36` | present | 0.0000 | 0.0001 | 0.0000 | 28.8 | OK |
| `Qwen3.8-27B-4bit` | `qwen36-27b with MODEL=` | present | 0.0000 | 0.0003 | 0.0000 | 27.2 | OK |
| `Devstral-Small-2-24B-Instruct-2512-4bit` | `devstral` | present | 0.0025 | 0.0338 | 0.0000 | 21.7 | OK |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | `nemotron-30b` | present | 0.0000 | 0.0004 | 0.0000 | 27.4 | OK |
| `flint-oQ8e` | `flint` | present | not loadable by mlx-lm or mlx-vlm; covered by the served probe below | | | | NOT LOADED |

`Qwen3-32B-4bit-DWQ` is the outlier by two orders of magnitude. Its logits are flat: the
best token sits 12.6 above the median where the other checkpoints sit 22 to 45 above.
Temperature 0.6 sharpens that back to a 0.05 % tail, which is why the model card's sampling
and greedy decode both work.

## Eight sampling arms on `qwen3-32b`

One sentence about the capital of France, 40 tokens, thinking closed. Left: the fallback
off. Right: the launcher default. The cells show the finish reason and the head of the reply.

| Request sends | Fallback off | Fallback on |
|---|---|---|
| nothing sent | length: ` простความสามารถ underestimate Q relu conversations lui advisor Mode Enc Sn Discovery.play` | stop: `The capital of France is Paris, a city renowned for its historical landmarks, cultural ins` |
| temperature 0 | stop: `The capital of France is Paris, a city renowned for its historical landmarks, cultural ins` | stop: `The capital of France is Paris, a city renowned for its historical landmarks, cultural ins` |
| temperature 0.6 | stop: `The capital of France is Paris, a bustling metropolis known for its rich history, iconic l` | stop: `The capital of France is Paris, a city renowned for its historic landmarks, cultural signi` |
| temperature 1.0 | length: `Lastly, human interactions across these discourses occurred through person tuausionsaguomi` | stop: `The capital of France is Paris.` |
| temperature 1.0, top_k 20 | stop: `Paris is the capital of France.` | stop: `The capital of France is Paris, a major global city known forits cultural attractions, his` |
| temperature 1.0, top_p 0.95 | length: `amوافق-par茅台 consoles resident innovation boats Submission林 do Neighbor翌 hashing_OWNER Per` | length: `Paris is the capital and largest city of both France and the region of ÎLE de FRANCE, and ` |
| temperature 0.6, top_k 20, top_p 0.95 | stop: `The capital of France is Paris, a city renowned for its cultural landmarks, historical sig` | stop: `The capital of France is Paris, a city renowned for its cultural heritage, historical land` |
| nothing sent, again | length: `phere-red alguns override texts HERO Little BuildingsIC Story Designer eligible caliber st` | stop: `The capital of France is Paris, a city renowned for its art, culture, and iconic landmarks` |

The resolution is per field. With the fallback on, a request that sends only temperature
1.0 still gets top-p 0.95 and top-k 20. Temperature 1.0 stays fragile on this checkpoint
even inside the top 20: of the right-hand replies at temperature 1.0, one runs words
together and one does not stop within 40 tokens.

## Served verification

`probe_defaults.py` sends four short questions twice with no sampling fields. The edit
tool probe is `probe_tool_edit.py`, also with no sampling fields. The validator sends
temperature 0 and ran on the fallback arm only.

| Preset | Arm | Default-sampling probe | Edit tool, thinking closed | Edit tool, thinking open | Validator |
|---|---|---|---|---|---|
| `qwen3-32b` | `SAMPLING_FALLBACK=` (off) | 0/8 | 0/2 | not run | not run |
| `qwen3-32b` | launcher default (on) | 8/8 | 0/2 | 1/2 | 3/3 |
| `qwen3-moe` | `SAMPLING_FALLBACK=` (off) | 8/8 | 2/2 | not run | not run |
| `qwen3-moe` | launcher default (on) | 8/8 | 2/2 | 2/2 | 3/3 |
| `qwen35-9b-8bit` | `SAMPLING_FALLBACK=` (off) | 8/8 | 2/2 | not run | not run |
| `qwen35-9b-8bit` | launcher default (on) | 8/8 | 2/2 | 2/2 | 3/3 |

Each fallback arm logged `No generation_config.json in the checkpoint; default sampling
params fall back to {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}` at boot. No arm
logged a traceback.

The same probe on the presets whose checkpoints ship a generation config, as launched:

| Preset | Default the server logged at boot | Default-sampling probe |
|---|---|---|
| `qwen36` | `{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}` | 8/8 |
| `coder-30b` | `{'repetition_penalty': 1.05, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}` | 8/8 |
| `qwen36-27b` | `{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}` | 8/8 |
| `qwen35` | `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}` | 8/8 |
| `devstral` | `{'temperature': 0.15}` | 8/8 |
| `nemotron-30b` | `{'temperature': 1.0, 'top_p': 1.0}` | 8/8 |
| `flint` | `{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}` | 8/8 |

`nemotron-30b` ships a config that is itself temperature 1.0 with top-p 1.0. Its tail is
healthy (under 0.01 % beyond the top 1000 tokens), so that default is safe on that checkpoint.

## The edit tool call on Qwen3-32B

With the fallback the reply is no longer soup, and the strict edit check still fails:
0/2 with thinking closed, 1/2 with thinking open. Two behaviours account for it.

- **Over-escaped newlines.** The `old` argument holds a literal backslash followed by `n`
  where the file has a newline, so it is not a substring of the file. A raw
  `/v1/completions` call at temperature 0, with the chat template rendered locally and no
  tool-call parser involved, shows the model writing the newlines of its JSON strings as
  backslash backslash n (5 of 5). The `qwen25` parser decodes that faithfully. The cause is
  the checkpoint's output, not the parser. Receipt: `served/qwen3-32b-fallback/raw-tool.txt`.
- **Answers in prose.** Some replies describe the edit in text and never call the tool.

`qwen3-moe` and `qwen35-9b-8bit` pass the same probe 2/2 in every arm.

## The patch

- `--sampling-defaults-fallback '<JSON>'`: the model defaults for a checkpoint with no
  generation config. Keys: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`.
  An unknown key fails at boot.
- Precedence, highest first: a field the request sends, the checkpoint's generation
  config, the fallback, the OpenAI default. `--sampling-defaults openai` ignores both
  model sources.
- `scripts/launch.sh`: `SAMPLING_FALLBACK`, set per preset; `SAMPLING_FALLBACK=` (empty)
  turns it off.
- `scripts/eval/validate_chat_template.py` warns when a checkpoint has no generation config.
- `scripts/eval/probe_all.sh` runs the default-sampling probe on every preset.
- Unit test, CPU, no server: `scripts/eval/test_sampling_fallback_unit.py`, 22/22.

## Disposition

Retained, on in the three presets.

Open item: `mlx-community/Qwen3-32B-4bit-DWQ` is a weak checkpoint for anything sampled.
It has the heavy tail above, it degrades at temperature 1.0 even with top-k 20, and it
over-escapes tool-call arguments. The candidate fix is a different 4-bit build for the
`qwen3-32b` preset. That swap needs MMLU and HumanEval on the replacement first, under the
rule that DWQ variants are never swapped blind.

## Receipts

- `tail/`: the audit's text and JSON per checkpoint.
- `arms/`: the eight sampling arms, fallback off and on.
- `served/<preset>-none/` and `served/<preset>-fallback/`: probe JSON and text, the
  validator text, the raw completion, and a cut of the server log.
- `served/<preset>-preset/`: the default-sampling probe on the other presets.
