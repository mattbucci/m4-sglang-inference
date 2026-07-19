# M4-D: Re-send the Docker-scoring ask as a callout in the 3090's README (and codify receiving-repo routing)

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | any-checkout |
| **Wall clock** | 1-2 hours (docs + dry-parse verification + two commits) |
| **GPU time** | none |
| **Depends on** | None hard. Pairs with 3090-B (their fleet-audit queue line 10, commit 7d3170f): 3090-B executes the scoring; this task makes it executable without re-derivation. If 3090-B starts first, it should read this callout's command rather than the old exports/README. |
| **Provides to** | 3090-B: an executable, dry-verified scoring command + drop-dir convention (evals/swebench/m4-imports/qwen36-m4-opencode/) + agreed result-return path (commit scores-docker-summary.json back to the m4 repo).; M4 headline update (post-3090-B): official full-26 Docker resolved-rate replaces the 5/13 local-testable cell in m4 README/CLAUDE.md.; Fleet ops: the receiving-repo routing rule in m4 CLAUDE.md, citable by future cross-team asks on all three rigs. |

## Objective

Get the 9-weeks-unseen qwen36 SWE-bench export actually scored by putting an actionable, correct ask where the executing team will read it — the 3090 README — with the exact command, runtime bound, and result-return path; and codify the routing rule (asks go in the RECEIVING repo's README) so no future cross-team handoff dies in the sender's docs. Closes the fleet's only unscored canonical-eval cell path from the M4 side and unblocks 3090-B.


## Background & receipts

- Export is real and unscored: /home/letsrtfm/AI/m4-sglang-inference/evals/swebench/exports/qwen36-predictions.jsonl = 26 JSONL records, 21 with non-empty model_patch, model_name_or_path='sglang/qwen36' (verified by parsing the file); committed 2026-05-18 (m4 commit a5a52d8), i.e. ~9 weeks unseen as of 2026-07-18.
- The ask lived only on the sending side: m4 README ('needs the 3090 Docker harness', recommended-picks section) and evals/swebench/exports/README.md; the 3090 README's only durable M4 mention is a one-line 'Sister teams' bullet (line ~451: 'MLX bridge; cross-checks chat-template + multimodal plumbing') with no ask.
- The 3090 README's Fleet-audit action queue (line 10, commit 7d3170f) now carries the terse checkbox 'Score M4's exported qwen36 predictions...' — that is the paired 3090-B execution item; this task supplies the durable, actionable callout plus the routing-lesson codification, and fixes the sending-side docs the callout points at.
- M4's exports/README.md is STALE on two counts (verified against live files): (a) says '15 of 16 unique instances' but the JSONL has 26/21 — the README predates the N=26 widening (a5a52d8); (b) its 'Score on the 3090' command passes --output, which 3090's evals/swebench/score_docker.py does NOT accept (argparse: --predictions, --dataset, --split, --max-workers, --timeout, --rewrite-reports, --cache-level, --run-id, --filter-helpers). A 3090 agent following it hits an immediate argparse error.
- score_docker.py output contract (read from source, /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/evals/swebench/score_docker.py main()): run_id defaults to the predictions file's parent dir name (or --run-id); writes scores-docker/<run_id>.<model>.report.json and scores-docker-summary.json NEXT TO the predictions file. Defaults: --max-workers 1 (2026-05-10 vfs_getattr_nosec kernel-BUG mitigation), --timeout 1800s/instance, --cache-level env.
- The 3090 checkout has no evals/swebench/m4-imports/ and no runs/ dir (run artifacts are box-local/gitignored) — the drop directory in the callout command must be mkdir'd first; the old exports/README scp target also assumed a dir that doesn't exist.
- The 8 instances M4 cannot score locally are receipted: m4 evals/swebench/README.md line 182 'INSTALL FAIL | 8 | M4 can't build the venv (old Python / native deps); needs 3090 Docker' (astropy x6, matplotlib, scikit-learn per m4 README); local score_local.py cell is 5/13 resolved on the testable subset — Docker scoring converts this to a full-26 official cell.
- Routing-lesson home exists: m4 CLAUDE.md already has a 'Cross-Team Collaboration' section listing both sister repos — the natural place for the 'asks go in the RECEIVING repo README' rule.
- No scoring has happened yet: find over the 3090 checkout returns zero scores-docker-summary.json; task is live, not already done.


## Method

1. Re-verify the export's live state (guards against drift since this spec): python3 -c "import json;d=[json.loads(l) for l in open('/home/letsrtfm/AI/m4-sglang-inference/evals/swebench/exports/qwen36-predictions.jsonl')];print(len(d),sum(1 for x in d if x['model_patch'].strip()))" — expect '26 21'. If it differs, use the live numbers everywhere below.
2. Fix the sending-side doc first (the callout will point at it): in /home/letsrtfm/AI/m4-sglang-inference/evals/swebench/exports/README.md, (a) replace the stale '15 of 16 unique instances' with '26 unique instances, 21 non-empty patches (the 5 empty: django-11019, flask-4045, sphinx-10451, requests-2148, sympy-11870 — see the SWE-bench README)'; (b) replace the 'Score on the 3090' command — score_docker.py has no --output flag — with the verified interface: mkdir -p evals/swebench/m4-imports/qwen36-m4-opencode && cp <export> evals/swebench/m4-imports/qwen36-m4-opencode/predictions.jsonl && python evals/swebench/score_docker.py --predictions evals/swebench/m4-imports/qwen36-m4-opencode/predictions.jsonl (run_id auto-derives from the parent dir name; results land in scores-docker/ + scores-docker-summary.json next to predictions.jsonl).
3. Write the callout in the RECEIVING repo: in /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/README.md '## Sister teams', extend the M4 bullet (line ~451) in their format (bold lead, backticked paths) with: ACTIVE ASK — score m4-sglang-inference/evals/swebench/exports/qwen36-predictions.jsonl (26 instances, 21 non-empty, model sglang/qwen36) on score_docker.py; the exact command from step 2; runtime: hours at the defaults (--max-workers 1 per your 2026-05-10 docker-I/O kernel-BUG mitigation, --timeout 1800s/instance => 13h absolute worst-case, typical far less; first run pays Docker env builds at --cache-level env); results land in evals/swebench/m4-imports/qwen36-m4-opencode/scores-docker-summary.json — commit that file back to the m4 repo (or its numbers into an m4-README update) so M4 can convert its 5/13-local cell into the official full-26 cell incl. the 8 old-Python INSTALL-FAIL instances (astropy x6, matplotlib, scikit-learn) M4 cannot build; cross-reference their existing fleet-audit checkbox (queue line 10) rather than duplicating it.
4. Dry-verify the callout command against their live script before committing: cd /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference && python3 -c "import importlib.util,sys;sys.argv=['x','--predictions','evals/swebench/exports-test/predictions.jsonl'];spec=importlib.util.spec_from_file_location('sd','evals/swebench/score_docker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);print(m.parse_args())" — argparse must accept every flag the callout quotes (no GPU/Docker touched; parse only).
5. Codify the routing lesson in /home/letsrtfm/AI/m4-sglang-inference/CLAUDE.md 'Cross-Team Collaboration': add a rule bullet — 'Asks that require a sister team to ACT go in the RECEIVING repo's README (their action queue / sister-teams section), committed to their repo; our README only mirrors the ask's status. Receipt: qwen36-predictions.jsonl sat unscored 2026-05-18 -> 2026-07-18 because the ask lived only in our README.'
6. Commit: m4 repo (steps 2+5) and 3090 repo (step 3) as separate small commits, each mentioning M4-D and the pairing with their fleet-audit scoring item; push per each repo's convention.


## Baseline & instrument

grep -c 'qwen36-predictions' in the 3090 README currently returns 1 (the terse queue checkbox only, no command/runtime/result-path); grep '15 of 16' in m4 exports/README.md returns the stale count; grep 'RECEIVING repo' in m4 CLAUDE.md returns nothing.


## Success criteria

- 3090 README '## Sister teams' M4 bullet contains: the export's repo-relative path, corrected counts (26 instances / 21 non-empty), a command that parses clean against their score_docker.py argparse (step-4 dry-parse exits 0), the 1800s/instance x 26 worst-case runtime bound, and the scores-docker-summary.json result-landing + return-to-M4 instruction — verifiable by reading the committed README.
- m4 exports/README.md agrees with the live JSONL (grep '26' present, '15 of 16' gone) and no longer quotes the nonexistent --output flag (grep -- '--output' evals/swebench/exports/README.md returns nothing).
- m4 CLAUDE.md contains the receiving-repo routing rule with the qwen36 receipt (grep 'RECEIVING' CLAUDE.md hits).
- Both commits landed and pushed; the 3090 fleet-audit checkbox (their queue line 10) is left unchecked for 3090-B to close.


## Kill criteria

- If /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/evals/swebench/**/scores-docker-summary.json already exists for a 26-record qwen36 predictions input (i.e. 3090-B already ran), drop steps 3-4, keep steps 2 and 5 (docs correction + routing rule), and record the callout as overtaken-by-events in the commit message.
- If the 3090 team has restructured/removed their '## Sister teams' section since commit 7d3170f, do not invent a new section: put the callout wherever their current cross-team/sister content lives, or as a sub-bullet under their existing fleet-audit queue item — and note the deviation in the commit message.


## Deliverables

- Edit to /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/README.md: expanded M4 bullet in '## Sister teams' (~line 451) carrying the full callout (export path, corrected 26/21 counts, exact runnable command, runtime bound, result-landing path, return-to-M4 instruction, pointer to their queue checkbox).
- Edit to /home/letsrtfm/AI/m4-sglang-inference/evals/swebench/exports/README.md: counts corrected 15/16 -> 26/21, broken --output command replaced with the score_docker.py-valid command, result-landing paths corrected to scores-docker-summary.json.
- Edit to /home/letsrtfm/AI/m4-sglang-inference/CLAUDE.md 'Cross-Team Collaboration' section: the routing rule (asks for sister-team action go in the RECEIVING repo's README; our README only mirrors the status).
- Two commits: one in the m4 repo (exports/README fix + CLAUDE.md rule), one in the 3090 repo (Sister-teams callout), each referencing M4-D.


## Constraints

- Docs-only task: no servers, no GPUs, no benches — safe to run any time on any checkout regardless of Rule 1/Rule 2 states on either box.
- Their repo, their format: the 3090 README callout must match its house style (bold-lead bullet, backticked repo-relative paths, receipt links); do not restructure their sections or check off their fleet-audit checkbox — closing it belongs to 3090-B.
- Any command placed in the callout must be dry-verified against score_docker.py's actual argparse before commit (the previous ask died partly because nobody could follow a broken command).
- Do not resolve M4's headline 5/13 number or edit their queue item's expected-rate framing — expected-rate text (60-80% strong, >50% confirms qwen36) is quoted from m4 exports/README.md, updated only for the corrected 21/26 patch-engagement base.
- Small self-contained commits, one per repo (m4 CLAUDE.md working-mode rule); cross-repo commit to the 3090 repo is fleet convention for sister-team asks (their README is the sharing channel).


## Risks

- Divergent-checkout race: the 3090 team's box-local README may be ahead of the checkout used for authoring; mitigate by pulling their repo immediately before editing and keeping the callout a purely additive bullet (no restructuring), so merge conflicts are trivial.
- Numbers drift: if anyone regenerates the export via aggregate.py before 3090-B runs, the callout's 26/21 goes stale — step 1's re-verify plus quoting counts in one place only (the callout points at exports/README.md for detail) bounds the blast radius.
- Expected-rate framing (60-80% strong, >50% confirms) is an M4 manual-review estimate, not a receipt; the callout must label it as expectation, or the adversarial-vetting culture on the 3090 side will discount the whole ask.
- The routing rule is only as good as its adoption on the other two rigs; this task can only place it in m4 CLAUDE.md and demonstrate it by example (the callout itself) — fleet-wide codification is the orchestrator's call.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
