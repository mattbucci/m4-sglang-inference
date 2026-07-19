# M4-F: Port patch replay gates + per-modality probe gate into setup.sh

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | 0.5-1 day (scripts ~2-3h; live-tree gate baseline ~30min; probe subset ~1-2h server time) |
| **GPU time** | No discrete GPU. ~1-2h M4 unified-memory occupancy for the 2-preset probe sweep (devstral + qwen36, sequential single-server); gates a/b/c and import smoke are CPU-only. |
| **Provides to** | The long-context item (M4-A): test_patch_gates.sh with SGLANG_TAG/SGLANG_DIR/PATCH_DIR overrides validates each bisect arm's patch chain.; All subsequent M4 queue items (sampling, bench pin, in-house quant, tool-call gate): trustworthy patch-chain + modality-gate substrate before any published number.; Fleet: closes the last repo without scripted replay gates (the 3090 has them scripted; the R9700's are doctrine in its CLAUDE.md). |

## Objective

setup.sh applies patches with a check-apply-else-skip loop that warns and
continues — the exact mechanism the patch-013 forensics (in git history)
blames for a month of fabricated VLM output: a silently dropped pixel_values
patch served plausible text while every image request ran text-only. Port the
3090's scripted 3-gate pristine-replay test and the R9700's fail-hard apply +
eager-import smoke into the M4 setup flow, and wire the existing behavioral
probe suite in as the mandatory post-setup gate. A manual pristine-replay +
byte-identity check has passed on the current stack (scratch clone,
0 differing files); this task turns that ad-hoc check into committed, gated
tooling.

## Background & receipts

- Current stack: SGLang pin `v0.5.15.post1`, **6 patch files
  (002/003/005/007/008/014)**, applied by setup.sh's glob
  `0[01][0-9]-*.patch`. setup.sh **shallow-clones** (`--depth 1 --branch
  $SGLANG_COMMIT`) — the pinned tag's tip object exists (pristine worktree at
  the tag works), but other tags/history are absent; gate scripts must not
  assume them.
- The patch loop prints '✗ ... (failed git apply --check)' then continues
  with only a WARNING — no abort, no equivalence check, no import smoke.
- Donor 1 — 3-gate replay script:
  `~/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/test_patch_gates.sh`.
  Gate (a) all patches apply in glob order on a PRISTINE tag worktree (a
  skip on pristine = broken chain the idempotent loop hides); gate (b)
  pristine+patches byte-identical to live tree, diff scoped to
  python/sglang, excluding _version.py/egg-info/__pycache__; gate (c) every
  patch FAILS `git apply --check` on the live tree (double-apply rejection).
  Env-overridable SGLANG_DIR/SGLANG_TAG/PATCH_DIR. Donor exits 2 with
  'FATAL: cannot create worktree' if the requested tag object is absent — a
  distinct exit path from a clean GATE-A FAIL.
- Donor 2 — fail-hard apply loop: the R9700 setup.sh collects
  FAILED_PATCHES, prints 'FATAL: N patch(es) FAILED', aborts nonzero.
  CAVEAT: its loop includes a `patch -p1 --fuzz=3 --forward` FUZZY FALLBACK
  before marking a patch FAILED. Do NOT port the fallback — the M4 loop is
  strict `git apply --check` + FATAL only ('zero fallback applications'
  discipline).
- Donor 3 — eager-import boot-chain smoke:
  `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/import_smoke.py`
  — CPU-only importlib walk over every patch-touched module + load-bearing
  attrs, PASS/FAIL table, exit 1 on any failure.
- The behavioral per-modality probe suite ALREADY EXISTS — this task wires it
  as a gate, it does not write probes: `scripts/eval/probe_all.sh`
  (per-preset launch → /health wait → probe_{thinking,vision,video,codegen}
  → verdict JSON) with committed verdicts in
  `benchmarks/quality/probe-trio/*.json`. probe_vision is the exact control
  that caught the patch-013 fabrication. setup.sh references
  probe_all/probe-trio zero times.
- Patched-module inventory for the import smoke: regenerate from
  `grep -h '^+++ b/' patches/*.patch | sort -u` (do not hardcode — the set
  changes with the patch stack).
- Path plumbing: `scripts/common.sh` sets VENV_DIR=$REPO_DIR/.venv and
  SGLANG_DIR=$REPO_DIR/components/sglang, both env-overridable — the gate
  script can default to these and still target scratch clones/bisect arms.

## Method

1. Record live-tree state: `git -C components/sglang rev-parse HEAD` and
   `git describe --tags` (expect the v0.5.15.post1 tag tip). Save as the
   baseline receipt.
2. Port scripts/test_patch_gates.sh from the 3090 donor: defaults
   SGLANG_DIR=$REPO_DIR/components/sglang, SGLANG_TAG=v0.5.15.post1,
   PATCH_DIR=$REPO_DIR/patches using setup.sh's exact glob
   `0[01][0-9]-*.patch`; keep gates (a) pristine glob-order apply, (b)
   `diff -rq $WT/python/sglang $SGLANG_DIR/python/sglang` excluding
   _version.py/egg-info/__pycache__, (c) per-patch `git apply --check` must
   FAIL on the live tree; keep env overrides (bisect arms reuse this);
   bash-3.2-safe (macOS /bin/bash — arrays OK, no mapfile/${var,,}).
3. BASELINE RUN: execute the gate script against the live stack before
   touching setup.sh; save full output as the receipt. A gate-(b) divergence
   means uncaptured live edits — capture each as a numbered patch or record
   the diff as a finding. NEVER reset/re-clone the live tree to force a
   pass.
4. Write scripts/eval/import_smoke.py modeled on the R9700 donor with an M4
   CHECKS list generated from the patched-module inventory (dotted paths)
   plus load-bearing attr checks. Run inside the venv with SGLANG_USE_MLX=1,
   CPU-only, exit 1 on any failure. Modules with import-time MPS/torch side
   effects get a per-check SKIP with a stated reason — never a silent drop.
   Run it: all checks must PASS on the live stack.
5. Harden setup.sh: (i) replace warn-and-continue in the patch loop with the
   FATAL-abort pattern — collect failed patch names, print FATAL, exit 1
   (strict `git apply --check`, no fuzzy fallback); (ii) keep the pin
   env-overridable (`SGLANG_COMMIT="${SGLANG_COMMIT:-v0.5.15.post1}"`)
   without changing its default; (iii) the existing-tree branch runs
   scripts/test_patch_gates.sh instead of any pull; (iv) append
   test_patch_gates.sh + import_smoke.py to the end-of-setup verification so
   every setup.sh run ends with the full gate.
6. Negative tests, deterministic and network-free (prove loudness — the
   property the old loop lacked). (A) FATAL-abort loudness: point the
   setup.sh patch loop at a scratch PATCH_DIR containing one
   deliberately-corrupted-hunk patch and run with
   `SGLANG_REPO=$REPO_DIR/components/sglang SGLANG_DIR=/tmp/scratch-sgl` —
   clone from the LOCAL tree, must exit nonzero at the patch step naming the
   failing patch. (B) Gate loudness on a wrong base: scratch-clone the local
   tree, apply the patch stack, tag the result, then run
   `SGLANG_DIR=<scratch> SGLANG_TAG=<that tag> scripts/test_patch_gates.sh`
   — gate (a) must FAIL loudly (every patch double-applies against the
   already-patched "pristine" worktree). Keep all transcripts as receipts;
   delete the scratch clones.
7. Behavioral modality gate. PREFLIGHT first: confirm probe_all.sh launches
   devstral+qwen36 with the DEFAULT mem-fraction, not a ≥32K long-context
   profile; if a preset inherits a ≥32K profile, oom_guard.sh is MANDATORY
   per the CLAUDE.md rule. Then run `PRESETS="devstral qwen36" bash
   scripts/eval/probe_all.sh` — this 2-preset subset covers all four probe
   classes. Diff the fresh benchmarks/quality/probe-trio/{devstral,qwen36}
   .json verdicts against the committed ones: any STRONG→FAIL flip is a
   stop-and-investigate finding (the patch-013 class), not a
   re-run-until-green. Commit fresh JSONs as the gate receipt.
8. Add one README line documenting the gate protocol (test_patch_gates.sh +
   import_smoke.py + probe gate) as the publish precondition.
9. Commit in small self-contained units: (1) test_patch_gates.sh, (2)
   import_smoke.py, (3) setup.sh hardening, (4) probe-trio receipts + README
   gate-protocol note.

## Baseline & instrument

Pre-change state captured by the new instruments themselves before setup.sh
is touched: the gate run (does live components/sglang == pristine
v0.5.15.post1 + 6 patches? does every patch double-apply-reject?) + the
import smoke on the live venv. Behavioral baseline: the committed verdict
JSONs in benchmarks/quality/probe-trio/. This is a tooling task that
publishes no perf number.

## Success criteria

- scripts/test_patch_gates.sh exits 0 on the live M4 stack — all 6 patches
  apply on pristine v0.5.15.post1, live tree byte-identical, all 6
  double-apply-reject — with any baseline divergence first captured as
  numbered patches or a written finding (receipt: gate transcript).
- scripts/eval/import_smoke.py exits 0 in the venv with SGLANG_USE_MLX=1,
  covering every patch-touched module (any per-check SKIP carries a stated
  reason).
- Negative tests both fail LOUDLY: (A) the corrupted-patch setup.sh run
  exits nonzero at the patch step naming the failing patch; (B) the
  wrong-base gate run reports GATE-A FAIL. Receipts: both transcripts.
- PRESETS="devstral qwen36" probe_all.sh completes with verdicts matching
  the committed probe-trio JSONs — no STRONG→FAIL flip (receipt: fresh
  JSONs + preflight note).
- A re-run of setup.sh on the existing tree executes gates+smoke and exits 0
  (receipt: transcript).

## Kill criteria

- components/sglang on the box is missing or not v0.5.15.post1-lineage →
  stop, record state; re-cloning destroys the live-tree evidence and needs a
  user decision.
- Gate (b) reveals live-tree divergence beyond a few attributable files → do
  not force equivalence; commit the diff receipt as a finding before any
  cleanup.
- No preset boots for reasons independent of this task (models purged,
  broken dylibs) → land the script units, commit boot logs as the
  null-result receipt, mark the probe gate PENDING in README rather than
  silently skipping it.
- Any single debugging rabbit-hole (e.g. an import-smoke failure tracing
  into upstream MLX) exceeds ~2h → record as finding with traceback receipt,
  scope a follow-up, ship the rest.

## Deliverables

- scripts/test_patch_gates.sh — M4 port of the 3090 3-gate script
  (env-overridable, setup.sh glob, bash-3.2-safe).
- scripts/eval/import_smoke.py — eager-import boot-chain smoke over the
  patch-touched modules (inventory generated, not hardcoded).
- Hardened scripts/setup.sh — FATAL-abort patch loop (strict git apply, no
  fuzzy fallback), env-overridable SGLANG_COMMIT, existing-tree branch runs
  gates, end-of-setup runs gates+smoke.
- README gate-protocol note (one paragraph).
- Receipts: baseline gate + import-smoke transcripts; both negative-test
  transcripts; fresh probe-trio JSONs + preflight note; any newly captured
  numbered patches if gate (b) found uncaptured live edits.

## Constraints

- Do NOT change the SGLANG_COMMIT default, re-clone, or reset the live
  components/sglang tree.
- One server at a time on the M4; probe runs are short-context so oom_guard
  is optional ONLY after the preflight confirms a default (non-0.4)
  mem-fraction; never raise MEM_FRAC (fraction of TOTAL system RAM; 0.85 has
  hard-locked the box).
- macOS: no setsid in probe_all.sh's pattern (nohup+disown), /bin/bash is
  3.2 — new shell code must avoid newer bashisms.
- Validate behavior, not exit status: the modality gate is the probe VERDICT
  lines, not /health=200 (the patch-013 fabrication served 200s for a
  month).
- Do NOT port the R9700 fuzzy `patch --fuzz=3 --forward` fallback — strict
  git apply + FATAL only.
- Small self-contained commits as progress is made; negative results are
  findings with receipts.

## Risks

- Import smoke may hit modules with import-time MPS/torch side effects;
  document a per-check skip with reason rather than silently dropping the
  module.
- Probe subset may fail for environmental reasons (HF cache eviction, brew
  ffmpeg/torchcodec dylib drift) — distinguish environment failures from
  patch-chain regressions before recording verdicts.
- The shallow clone lacks non-pinned tags — any future gate variant that
  wants a different-tag worktree must fetch it explicitly or use the
  scratch-clone pattern from the negative test.
