# Prompt-lookup drafts on a lone request's decode (patch 040)

Run dates 2026-11-10 and 2026-11-11. Stack: SGLang v0.5.23 + patches 001-040
(040 comes out of this run), MLX 0.32.4, mlx-lm 0.32.3.dev17, mlx-vlm 0.6.5,
M4 Pro 64 GB. Models: `mlx-community/Qwen3.6-27B-4bit` (preset `qwen36-27b`,
dense with DeltaNet layers, 64 layers, mlx-vlm path),
`mlx-community/Qwen3-32B-4bit-DWQ` (preset `qwen3-32b`, dense, attention
only, 64 layers, mlx-lm path) and `mlx-community/Qwen3.6-35B-A3B-4bit`
(preset `qwen36`, MoE with DeltaNet layers, 40 layers).

`../concurrent/VERDICT.md` measured what a decode forward of several rows
costs and left one candidate open: on a dense checkpoint the second row is
nearly free, so a lone request could spend it on a guess of its own next
token. A coding agent's output repeats its context (a path it just read, the
lines around an edit, the argument names of a tool call), so the guess is a
lookup: find the last few tokens of the context earlier in the context and
copy what followed. This folder takes that from a CPU replay to a serving
patch.

## Method

The method of `../micro/VERDICT.md` (census, raw bench before serving work,
same-tree serving A/B in both arm orders, gates, a numbered patch when
retained, a disposition per experiment), plus:

1. Replay before any GPU time. `scripts/bench/prompt_lookup_acceptance.py`
   re-renders every assistant turn of an archived SWE-bench cell, makes the
   guess at every output position (the context's last 8, 4, 3 or 2 tokens,
   longest match first) and prices draft policies with a measured cost table
   of a k-row forward. CPU only, a whole cell in about a minute.
2. A kill criterion written before the raw run: a serving patch is scoped
   only if the best drafting arm reaches 1.20 times the ordinary loop on a
   dense checkpoint in raw MLX.
3. Raw draft-and-verify on real archived turns before any serving code
   (`scripts/bench/mlx_lookup_decode_probe.py`): the ordinary pipelined loop
   (`plain`), the same loop without its lookahead step (`sync`, what a turn
   pays if no draft is ever offered), and drafting arms.
4. Served A/B on one tree, one booted server per arm, both arm orders
   (`scripts/bench/lookup_served_ab.py`): four archived turns of two omp
   sessions (psf__requests-1963 turns 12 and 22, pylint-dev__pylint-5859
   turns 4 and 9; 6.6K to 10.8K tokens of context), the conversation up to
   the turn sent as token ids, decode rate from the first streamed token to
   the last, pooled over requests. A flag arm counts only if its log shows
   `MLX lookup decode enabled`.
5. Greedy text is not a fidelity measure: two correct paths round
   differently in bf16 and a near-tie flips. Fidelity is teacher-forced on
   the real weights, and the state a drafted request leaves in the radix
   cache is graded by a follow-up served from it against the same follow-up
   cold.
6. The agentic lane is graded as it runs: sampled, with the checkpoint's
   generation config, on a prompt that leaves thinking open.
## Result

Retained as opt-in patch 040 for dense checkpoints. No preset sets the flag.

| measurement, against the ordinary loop | Qwen3.6-27B | Qwen3-32B |
|---|---|---|
| raw draft-and-verify, greedy | x1.63 | x1.29 |
| served, greedy, 6.6K to 10.8K of context | x1.50 | x1.28 |
| served, greedy, 26K to 27K of context | x1.60 | not run |
| served, sampled, thinking closed | x1.24 | x1.20 |
| served, sampled, thinking open (the agentic lane's shape) | x1.11 | x1.08 |
| a request with nothing to draft (raw `sync` arm) | -3.6 % | -1.2 % |

The criterion written before the raw run was 1.20 on a dense checkpoint in
raw MLX. Both dense checkpoints met it and the serving patch was built.
Graded the way the agentic lane runs (method item 6) the flag reads 1.11 and
1.08, under that bar. It is therefore a flag for output that restates its
context with thinking closed, and it is not claimed as an agentic-lane win.
The MoE checkpoint loses and the flag is not for it.
## Acceptance replay

Two archived omp cells, every assistant turn, thinking included. Their text
was decoded under the lane's sampling, so this is acceptance on sampled
output.

| cell | turns | output tokens | positions with a match | first guess right, of all positions | holds 2 tokens | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| `qwen36-26inst-omp-2026-11-08` | 1,286 | 198,720 | 79.9 % | 58.3 % | 47.8 % | 35.5 % | 24.3 % | 14.1 % |
| `qwen36-26inst-omp-p033-2026-11-11` | 1,092 | 222,439 | 80.4 % | 56.4 % | 45.5 % | 33.4 % | 22.7 % | 13.9 % |

Predicted speedup over every output token of a cell, against the pipelined
one-row loop. `free` rolls a rejected draft back at no cost (attention-only
models trim the cache offset); `carry` replays the tokens a rolled-back
forward had accepted in the next forward (models with recurrent layers).
The oracle drafts exactly as far as the guess holds.

| cost table | policy | first cell, free / carry | second cell, free / carry |
|---|---|---|---|
| qwen36-27b at 8K | fixed 2 rows | 1.376 / 1.328 | 1.361 / 1.307 |
| qwen36-27b at 8K | adaptive to 4 rows | 1.477 / 1.388 | 1.447 / 1.353 |
| qwen36-27b at 8K | oracle | 1.719 | 1.692 |
| qwen36-27b at 32K | fixed 2 rows | 1.332 / 1.291 | 1.317 / 1.272 |
| qwen36-27b at 32K | adaptive to 4 rows | 1.432 / 1.348 | 1.402 / 1.314 |
| qwen36-27b at 32K | oracle | 1.611 | 1.590 |
| qwen36 at 8K | adaptive to 4 rows | 1.143 / 1.036 | 1.115 / 1.006 |
| qwen36 at 8K, patches 030-032 on | adaptive to 4 rows | 1.104 / 0.997 | 1.077 / 0.966 |
| qwen36 at 32K | adaptive to 4 rows | 1.080 / 0.970 | 1.054 / 0.941 |
| qwen36 at 128K | adaptive to 4 rows | 0.981 / 0.860 | 0.959 / 0.833 |
| qwen36 at 128K | oracle | 1.214 | 1.209 |

The acceptance is the same for every model; the cost table decides. On the
dense checkpoint the adaptive policy with carry predicts 1.31 to 1.39 over a
whole cell at either depth. On the MoE checkpoint nothing with carry clears
1.04, and at 128K every policy loses.

## What a k-row forward costs

`scripts/bench/mlx_lookup_decode_probe.py --rows-depth`, in units of the
pipelined one-row step at the same depth (a one-row forward alone costs more
than 1.00 because the pipelined loop overlaps its lookahead step).

| model, depth | pipelined step | 1 row | 2 rows | 3 rows | 4 rows | 8 rows |
|---|---|---|---|---|---|---|
| qwen36-27b, 8,192 | 70.3 ms | 1.05 | 1.10 | 1.32 | 1.54 | 3.15 |
| qwen36-27b, 32,768 | 76.6 ms | 1.05 | 1.15 | 1.31 | 1.60 | 3.90 |
| qwen36, 8,192 | 12.1 ms | 1.24 | 1.54 | 1.87 | 2.19 | 4.17 |
| qwen36, 32,768 | 14.6 ms | 1.20 | 1.60 | 1.95 | 2.41 | 6.30 |
| qwen36, 131,072 | 24.3 ms | 1.11 | 1.70 | 2.19 | 2.83 | 10.55 |

A dense step is one read of 15 GB of weights and a second row rides on it
for a tenth of a step. An MoE step routes each row to its own experts, and
at depth each row also pays its own attention over the whole context, so a
row there is never cheap.

## Raw draft-and-verify

Four archived turns, 256 new tokens at most, greedy, raw MLX with the
runner's decode loop. `adapt4` drafts up to three tokens behind the newest
one and moves its draft length with what the last forward kept; `fixed2`
always drafts one.

| model | plain | sync (no draft offered) | fixed2 | adapt4 |
|---|---|---|---|---|
| Qwen3.6-27B, tok/s pooled | 14.08 | 13.58 (x0.964) | 19.86 (x1.411) | 22.95 (x1.630) |
| Qwen3-32B, tok/s pooled | 10.52 | 10.39 (x0.988) | 13.19 (x1.254) | 13.57 (x1.290) |
| qwen36 at 6.6K to 10.8K, tok/s pooled | 77.97 | 65.34 (x0.838) | 79.43 (x1.019, match of 4 required) | 79.99 (x1.026) |
| qwen36 at 50K to 82K, tok/s pooled | 54.93 | 48.62 (x0.885) | 51.81 (x0.943, match of 4 required) | 34.27 (x0.624) |

Per turn on the 27B, `adapt4`: x1.266 (33 of 58 drafted tokens kept), x2.560
(187 of 191, a tool call that restates code from the context), x1.570 (23
of 29), x1.146 (42 of 91). Both dense checkpoints clear the 1.20 criterion.
The MoE checkpoint does not come close and loses at depth, where a rejected
draft on a hybrid replays its accepted tokens at the full per-row cost.

## Fidelity of a drafted step

The same probe, teacher-forced on the plain run's tokens: the logits of a
drafting forward against the one-row loop's, next to an ordinary forward of
several rows against the same loop (the noise two stock paths already have).

| model | arm | KL(one-row, drafting) mean / worst | KL(one-row, one forward) mean / worst | same top token |
|---|---|---|---|---|
| Qwen3.6-27B | adapt4 | 0.00021 / 0.0034 | 0.00019 / 0.0020 | 216 of 217 |
| Qwen3.6-27B | fixed2 | 0.00019 / 0.0034 | 0.00019 / 0.0020 | 216 of 217 |
| Qwen3-32B | adapt4 | 0.00026 / 0.0019 | 0.00026 / 0.0017 | 186 of 188 |
| Qwen3-32B | fixed2 | 0.00026 / 0.0019 | 0.00026 / 0.0017 | 186 of 188 |

A drafted step is as close to the one-row loop as any forward of several
rows. Where the greedy text leaves the plain run's, the plain run's two best
tokens are one bf16 step apart (0.062 in logit units on the 32B) and the
drafting arm took the other one.
## What patch 040 does

`SGLANG_MLX_LOOKUP_DECODE=1` (off unless set) with
`SGLANG_MLX_LOOKUP_DECODE_ROWS` (default 4, the most rows one forward
carries). New module `hardware_backend/mlx/lookup_decode.py` holds the parts
with no MLX state (n-gram index, draft length policy, per-request stream);
the forward, the rollback and the finish live in `model_runner.py`.

- **Who drafts.** A decode step of exactly one request with plain logits: no
  grammar or logit-bias rows, no logprobs, no logits hook. On mlx-vlm Qwen
  checkpoints the prompt must be text only. A step that breaks one of these
  ends drafting for that request for good and puts its caches where the
  ordinary loop keeps them. The flag stays off at boot, with a warning, on
  sliding-window models, under the custom RoPE kernel and under NaN
  sanitization.
- **The guess.** The context's last 8, 4, 3 or 2 tokens, longest first, at
  their most recent earlier occurrence; the tokens that followed there are
  the draft. Two rows after a miss, doubling after every fully kept draft up
  to the row limit. A draft is cut before a stop token and before the
  request's token limit, so no forward runs past a finish the runner can
  foresee.
- **The check.** One forward of the request's waiting tokens plus the draft,
  the same call a prompt chunk makes. Row i is the distribution after the
  first i drafted tokens, and the request's own selection rule picks a token
  from every row: arg-max for a greedy request, its sampler for a sampled
  one. Leading drafts equal to their row's pick are kept, and the pick of the
  first row that disagrees follows them. Tokens verified ahead are handed to
  the scheduler one per step with no forward.
- **Sampled requests.** Every row is drawn on its own noise, and a row is
  only used when the tokens before it were kept, in which case it is the
  ordinary loop's distribution at that position. A draft is therefore kept
  with the probability the model gives it, and the emitted text is
  distributed as the ordinary loop's. A seeded request reads the noise of its
  own position and reproduces the ordinary loop's tokens.
- **Rollback.** An attention-only model lowers its cache offset past the
  rejected rows. A model with recurrent layers keeps a frozen view of each
  recurrent layer from before the forward, restores it on a miss and replays
  the kept tokens in front of the next forward; with more than six tokens
  waiting the next forward only replays.
- **Finish.** Before the radix insert the caches are put at prompt plus
  output minus one token, where the ordinary loop leaves them: rows verified
  ahead of a stop string or an abort are dropped, kept tokens not replayed
  yet are replayed. If a hybrid's recurrent state cannot be put there the
  insert is skipped with a warning.
- **What it costs with nothing to draft.** A drafting request does not use
  the pipelined loop's lookahead step, because its next forward depends on
  the token just picked. That is the `sync` arm of the raw run: 3.6 % on the
  27B, 1.2 % on Qwen3-32B, 16 % on qwen36.

Each finished request logs one line, for example `255 tokens from 65
drafting, 0 replay-only and 3 ordinary forwards, 187 of 189 drafted tokens
kept, 2 rows replayed`.
## Served, greedy

Live tree, one server per arm, arms mirrored (off, rows 4, rows 2, rows 2,
rows 4, off), the four turns twice per arm, 256 new tokens at most, the
turn's thinking block closed in the prompt so the output starts at the tool
call. Pooled decode rate; the two arms of an order in brackets.

| checkpoint | off | rows 4 | rows 2 |
|---|---|---|---|
| qwen36-27b, 16 requests, 2,586 tokens | 13.98 (13.94, 14.01) | 20.99, x1.502 (20.93, 21.04) | 18.82, x1.347 (18.79, 18.86) |
| qwen3-32b, 16 requests, 3,244 tokens | 10.48 (10.45, 10.51) | 13.43, x1.282 (13.32, 13.54) | 12.89, x1.230 (12.73, 13.06) |

Greedy text with rows 4 equals the off arm's on 5 of 8 requests on the 27B
and 6 of 8 on the 32B. The others part at one step (57 and 126 tokens in on
the 27B, 143 on the 32B). The off arm parts from itself the same way on the
27B: its cold pass and its radix-hit pass differ on two of four turns, at
steps 0 and 57, and the rows 4 arm takes the other pass's branch at 57. The
deep contexts below are equal in every arm. Text is therefore not the
fidelity measure here; the teacher-forced table above and the follow-up
table below are. Per turn on the 27B with rows 4
(untimed follow-up arm, one pass): 17.70 / 35.63 / 21.72 / 16.15 tok/s
against 14.06 / 13.91 / 13.84 / 13.81. The second turn is a tool call that
restates code from the context: 255 tokens from 68 forwards, 187 of 189
drafted tokens kept. The fourth is prose about the change: 152 tokens from
108 forwards, 44 of 91 kept.

## Served, sampled

The lane's own decode: temperature 1.0, top-k 20, top-p 0.95 on the 27B
(its generation config) and temperature 0.6 on Qwen3-32B, quiet GPU, arms
mirrored. Two prompt shapes per arm. `closed` ends the thinking block in the
prompt, three passes of the four turns at 256 new tokens. `open` leaves the
thinking block open, so the model reasons before it calls a tool, two passes
at 384 new tokens. A sampled arm draws a new text on every pass, so only
pooled rates are read.

| checkpoint, shape | off | rows 4 | rows 2 |
|---|---|---|---|
| qwen36-27b, closed, 24 requests | 14.59 (14.61, 14.58) | 18.04, x1.236 (17.76, 18.32) | 17.25, x1.182 (17.25, 17.27) |
| qwen3-32b, closed, 24 requests | 10.99 (10.97, 11.00) | 13.18, x1.200 (12.92, 13.45) | not run |
| qwen36-27b, open, 16 requests | 14.57 (14.60, 14.54) | 16.21, x1.113 (16.20, 16.21) | 16.01, x1.099 (16.29, 15.78) |
| qwen3-32b, open, 16 requests | 10.94 (10.96, 10.91) | 11.85, x1.084 (11.50, 12.23) | not run |

What the rows 4 arms did, from the per-request log lines:

| checkpoint, shape | tokens | forwards | tokens per forward | drafted tokens kept | forwards that drafted |
|---|---|---|---|---|---|
| qwen36-27b, greedy deep (next section) | 1,872 | 768 | 2.44 | 1,104 of 1,546 (71 %) | 89 % |
| qwen36-27b, closed | 4,589 | 2,999 | 1.53 | 1,590 of 2,931 (54 %) | 52 % |
| qwen36-27b, open | 4,570 | 3,478 | 1.31 | 1,092 of 2,587 (42 %) | 43 % |
| qwen3-32b, closed | 5,418 | 3,856 | 1.41 | 1,562 of 2,894 (54 %) | 40 % |
| qwen3-32b, open | 6,128 | 5,000 | 1.23 | 1,128 of 3,058 (37 %) | 38 % |

Reasoning text is new text. With thinking open fewer positions have a match
and fewer drafts hold, so the win drops from 1.24 to 1.11 on the 27B and
from 1.20 to 1.08 on the 32B. The 27B also pays replay: 546 of its 3,478
forwards in the open shape only replayed tokens a rolled-back forward had
accepted. The agentic lane runs in the open shape, and there the flag is
under the 1.20 criterion on both checkpoints.

## Served, deep contexts

qwen36-27b, greedy, two turns of an astropy__astropy-7746 session at 26,114
and 27,220 tokens of context, two passes per arm, arms mirrored.

| arm | requests | tokens | tok/s pooled |
|---|---|---|---|
| off | 8 | 1,872 | 13.71 (13.75, 13.66) |
| rows 4 | 8 | 1,872 | 21.91, x1.598 (21.92, 21.89) |

Depth does not take the win away on a dense checkpoint: the rows of a
forward share the weight read, and that is still most of a step at 27K.
## The cache a drafted request leaves

A drafted request ends with its caches ahead of, behind or level with the
tokens it handed out, and the finish puts them where the ordinary loop
leaves them before the radix insert. The check is a follow-up served from
that state against the same follow-up cold, top-20 logprobs, 48 steps.

Follow-up fidelity on qwen36-27b, KL(cold || hit) mean / max over the follow-up's
first steps (48 unless the reply ends sooner). A cold run with the flag on
equals a cold run with it off exactly (KL 0) on every case.

| case | off, chunk 2048 | rows 4, chunk 2048 | off, chunk 1024 | rows 4, chunk 1024 |
|---|---|---|---|---|
| a12 full turn | 0.047555 / 0.084034 (2 steps) | 0.182878 (1 step) | 0.105437 (1 step) | 0.001156 / 0.022654 |
| a12 cut at 24 | 0.000529 / 0.002870 | 0.000564 / 0.004259 | 0.001078 / 0.013999 | 0.001596 / 0.039797 |
| a12 cut at 48 | 0.000828 / 0.004953 | 0.000649 / 0.004890 | turn ends at 47 | turn ends at 47 |
| a12 cut at 72 | 0.000875 / 0.015942 | 0.000653 / 0.005427 | turn ends at 47 | turn ends at 47 |
| b9 | 0.001641 / 0.020367 | 0.001262 / 0.009084 | 0.001583 / 0.021054 (30 steps) | 0.001109 / 0.007803 (30 steps) |
| a22 stop | 0.001055 / 0.011008 | 0.000971 / 0.005309 | 0.001004 / 0.004546 | 0.001439 / 0.017910 |

Reading: the flag's follow-up state sits in the flag-off class on every case
that runs more than two steps. The a12 full-turn follow-up is short in three
arms (the reply ends after one or two tokens) and its first step is a close
call in every arm; with 48 steps to compare (rows 4, chunk 1024) it reads
0.0012 like the rest.

Qwen3-32B (attention only, the trim rollback), chunk 2048, same four turns:
rows 4 reads 0.000173 / 0.000719, 0.000187 / 0.000537, 0.000260 / 0.000818
and 0.000187 / 0.000884; flag off reads 0.000259 / 0.000979 and 0.000235 /
0.001566 on the two turns it shares. Every follow-up was served from the
cache up to the last chunk boundary (for example 11,100 of 11,128 prompt
tokens).
## Gates with the flag on

One server per arm, `SGLANG_MLX_LOOKUP_DECODE=1` against the stock preset.
The edit-tool check is `scripts/eval/probe_tool_edit.py`: a file in the user
turn, one `edit` tool, and a reply passes when its `old` argument is an exact
substring of the file. Depth recall is `scripts/eval/probe_depth_recall.py`
at label 28000.

| gate | `qwen36-27b`, flag on | `qwen36-27b`, flag off | `qwen3-32b`, flag on |
|---|---|---|---|
| `validate_capabilities.py` | 3/3 | preset gate | 3/3 |
| `probe_thinking.py` | THINKING VERIFIED | preset gate | THINKING VERIFIED |
| `probe_vision.py` | STRONG | preset gate | text-only checkpoint |
| `probe_codegen.py` | 8/8 STRONG | preset gate | 8/8 STRONG |
| edit tool call, thinking closed | 2/2 (100 tokens each) | 2/2 (100 tokens each) | see below |
| edit tool call, thinking open | 2/2 (290 and 345 tokens) | 2/2 (324 and 1,144 tokens) | see below |
| depth recall, 32,240 prompt tokens | 6/6, 301 s | 6/6, 319 s | not run |
| tracebacks in the server log | 0 | 0 | 0 |
| drafting requests in the server log | 17 | 0 | 7 |

Each flag arm logs `MLX lookup decode enabled` once. The 27B's tool calls are
drafted text: the thinking-closed call is 99 tokens from 40 drafting, 6
replay-only and 7 ordinary forwards with 46 of 82 drafted tokens kept, and
the recall answer is 41 tokens from 20 drafting forwards.

### The edit tool call on Qwen3-32B

The strict check fails on `qwen3-32b` with the flag on and with it off, for
two reasons that have nothing to do with the flag, so the flag is graded
here by comparing whole replies under greedy decode.

| what the request sends | flag off | flag on |
|---|---|---|
| no sampling field | 0/2 closed, 0/2 open: token soup to the length cap | 0/2 closed, 0/2 open: token soup to the length cap, 799 ordinary forwards and nothing drafted |
| the model card's sampling (0.7 / 0.8 / 20 closed, 0.6 / 0.95 / 20 open) | closed: 2 of 2 replies are an `edit` call with the right keys whose `old` carries a literal backslash-n where the file has a newline, so it is not a substring; open: 2 of 2 replies answer in text without a call | not run |
| temperature 0, thinking closed | an `edit` call, 205 tokens | the same call and content byte for byte, 205 tokens; 204 tokens from 62 drafting and 70 ordinary forwards, 72 of 122 drafted tokens kept |
| temperature 0, thinking open | an `edit` call, 493 tokens, 1,671 characters of reasoning | the same call and content byte for byte, 317 tokens, 956 characters of reasoning; 316 tokens from 92 drafting and 139 ordinary forwards, 85 of 174 drafted tokens kept |

Reading:

- No sampling field. `mlx-community/Qwen3-32B-4bit-DWQ` ships no
  `generation_config.json`, so the server fills an unset request with
  temperature 1.0 and no truncation, and this build holds 11 to 56 % of its
  probability beyond the top 1000 tokens there. Both arms decode noise. The
  finding and its fix are recorded apart, in
  `benchmarks/perf/2026-11-10/sampling-defaults/`.
- The model card's sampling. The reply is coherent and the call has the right
  shape; the arguments are escaped once too often. That is this checkpoint
  with the `qwen25` tool-call parser on this prompt, with the flag off.
- Greedy. The flag-on reply carries the flag-off reply's tool call and
  content byte for byte in both prompt shapes, with 59 % and 49 % of the
  drafted tokens kept. With thinking open the reasoning text parts ways at a
  close call, the rounding class of any forward of several rows, and arrives
  at the same call.
## Side arm: the whole reply in the radix tree on attention-only presets

The follow-up probes above showed a stock attention-only preset serving a
follow-up with about a hundred tokens of the previous reply missing from the
cache. Chained decode steps skip the scheduler's per-step slot allocation,
so a finished request keys only part of its output in the radix tree.
`SGLANG_MLX_COMMIT_CHAINED=1` (patch 028) allocates the rest at release.
The launcher sets it for hybrid presets only. Arms here: `qwen3-32b` and
`coder-30b`, stock preset against the flag, mirrored (off, on, on, off),
`scripts/bench/published_shape_sweep.py` with three greedy runs of 256
tokens at 1K and a chat follow-up at 4K and 8K, plus the follow-up fidelity
probe on the first arm of each kind.

| preset | arm | decode at 1K, tok/s (two arms) | follow-up TTFT at 4K, s | follow-up TTFT at 8K, s | fidelity probe: prompt tokens from the cache | KL(cold, hit) mean / max |
|---|---|---|---|---|---|---|
| coder-30b | stock | 82.81, 83.05 | 0.269, 0.306 | 0.388, 0.278 | 6,228 of 6,342; 9,242 of 9,375 | 0.003713 / 0.009031; 0.001710 / 0.017250 |
| coder-30b | flag | 83.80, 83.13 | 0.125, 0.122 | 0.164, 0.168 | 6,319 of 6,342; 9,352 of 9,375 | 0.004811 / 0.012448; 0.003407 / 0.011751 |
| qwen3-32b | stock | 12.26, 12.29 | 0.696, 0.537 | 0.898, 0.727 | 6,232 of 6,304; 9,246 of 9,528 | 0.000172 / 0.000630; 0.000235 / 0.001566 |
| qwen3-32b | flag | 12.36, 12.42 | 0.533, 0.694 | 0.725, 0.903 | 6,277 of 6,304; 9,500 of 9,528 | 0.000154 / 0.000475; 0.000206 / 0.000936 |

With the flag a follow-up is served from the cache up to the last chunk
boundary on both presets, as it already is on the hybrid presets. On
coder-30b the sweep's follow-up prefills 18 new tokens where the stock
preset prefills 75 to 142, and its first token arrives 0.12 to 0.17 s after
the request against 0.27 to 0.39 s. On qwen3-32b the sweep's reply is under
twenty tokens, so there is nothing to commit and the TTFT pairs follow the
arm's position in the run (0.53 / 0.73 and 0.69 / 0.90 in both kinds), not
the flag. Decode is inside the 1 % noise floor on both. Follow-up fidelity
is in the stock class on both. No tracebacks in any arm.

Disposition: candidate for the attention-only presets. It changes what a
finished request stores, so it goes through the cross-preset settings run
with capability gates on each preset before a launcher default moves.
## Disposition

| item | disposition |
|---|---|
| Patch 040, `SGLANG_MLX_LOOKUP_DECODE` | Retained, opt-in, dense checkpoints only. Off in every preset. |
| Rows per forward | Default 4. Rows 2 is behind rows 4 in every arm (x1.35 against x1.50 greedy, x1.18 against x1.24 sampled closed, x1.10 against x1.11 sampled open on the 27B). |
| As a preset default on `qwen36-27b` or `qwen3-32b` | Not taken. A retained default must lose nowhere, and a request with nothing to draft loses 1.2 to 3.6 %. The agentic shape wins 8 to 11 %, under the 1.20 criterion. |
| On MoE checkpoints (`qwen36`, `coder-30b`, `flint`) | Rejected by the raw run: 16 % lost with nothing to draft on `qwen36`, x0.62 at 50K to 82K. Flint's sliding-window layers also keep the flag off at boot. |
| `SGLANG_MLX_COMMIT_CHAINED` on attention-only presets | Candidate; goes through the cross-preset settings run with gates. |
| A trained drafter for reasoning text | Open. The lookup has nothing to copy while the model reasons, which is where the agentic shape loses its win. |

Why it is kept at all. Every served arm with a code context won in both
arm orders, from 8 % to 60 %. The drafted step is as close to the one-row
loop as any forward of several rows, a sampled request draws from the same
distributions, and the cache a drafted request leaves serves a follow-up in
the same class as an ordinary request's. The capability gates pass with the
flag on on both checkpoints. On the 27B so do a tool call that quotes the
file back, with and without thinking, and depth recall at 32K. On Qwen3-32B
the same tool request returns the flag-off arm's call byte for byte under
greedy decode.

What was not measured. No SWE-bench cell ran with the flag on: the agentic
recommendation is `qwen36`, an MoE checkpoint the flag is not for, and a
cell on a dense 27B takes about five times as long per token. The win on a
whole agentic session is bounded by the open-shape arms above.
## Receipts

| folder | holds |
|---|---|
| `replay/` | acceptance replay of two archived omp cells, whole cell and by depth |
| `raw/` | k-row forward cost by depth, raw draft-and-verify, teacher-forced fidelity, drivers |
| `served-greedy/` | greedy arms on both dense checkpoints with the follow-up arms, driver |
| `followup/` | follow-up fidelity on qwen36-27b at chunk 2048 and 1024, driver |
| `served-sampled/` | sampled arms in both prompt shapes, pooled table, driver |
| `served-deep/` | greedy arms at 26K to 27K of context, driver |
| `commit-chained/` | the side arm on `qwen3-32b` and `coder-30b`, driver |
| `gates/` | capability gates, edit-tool checks (strict, the model card's sampling, whole greedy replies), depth recall with and without the flag, unit test output, driver |

Scripts: `scripts/bench/prompt_lookup_acceptance.py`,
`scripts/bench/mlx_lookup_decode_probe.py`,
`scripts/bench/lookup_served_ab.py`,
`scripts/eval/probe_tool_edit.py`,
`scripts/eval/test_lookup_decode_unit.py`.
