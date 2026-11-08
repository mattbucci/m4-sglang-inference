#!/usr/bin/env python3
"""SWE-bench Lite rollout driver — a coding-agent scaffold against local SGLang.

Phase 1: agent rollout only. For each instance:
  1. clone <repo> at <base_commit> into a temp worktree (ancestor-only)
  2. run the scaffold in the worktree with the problem statement:
       omp      (default) `omp --model sglang/<served> --yolo --no-title -p`,
                the task on stdin, provider + settings written per run to
                <out>/omp-agent/ (PI_CODING_AGENT_DIR), session jsonl under
                <out>/sessions/<instance>/
       opencode `opencode run --dir <worktree> --model sglang/<key> ...`
  3. capture `git diff` as the prediction patch
  4. write predictions/<instance_id>.diff and append to predictions.jsonl

The scaffold choice rests on the 3090 rig's scaffold bake-off: every
opencode / pi-family pair (omp included) is a McNemar tie on the same 300
isolated instances, so one lane is enough and omp is it. opencode stays
selectable (`--scaffold opencode`) to reproduce the archived cells.

Phase 2 (separate, when Docker is available): score predictions.jsonl via the
official SWE-bench harness.

Usage:
    python evals/swebench/run_rollouts.py --model sglang/coder-reap-25b \\
        --instances 3 --out evals/swebench/runs/coder-reap-25b-smoke

    # Full Lite (300):
    python evals/swebench/run_rollouts.py --model sglang/coder-reap-25b \\
        --out evals/swebench/runs/coder-reap-25b-lite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="model id recorded in predictions.jsonl, e.g. sglang/qwen36 "
                        "(opencode: the opencode.json key; omp: any sglang/<name>)")
    p.add_argument("--scaffold", choices=["omp", "opencode"], default="omp",
                   help="coding-agent scaffold that drives the rollout (default omp)")
    p.add_argument("--model-url", default=None,
                   help="OpenAI-compatible base URL the scaffold talks to (default: --server-url; "
                        "point it at no_thinking_proxy.py to force enable_thinking=false). omp only — "
                        "opencode reads ~/.config/opencode/opencode.jsonc")
    p.add_argument("--context-window", type=int, default=131072,
                   help="served context window handed to the scaffold (omp models.yml contextWindow; "
                        "must match the server's CTX, it drives compaction)")
    p.add_argument("--output-budget", type=int, default=8192,
                   help="per-completion output cap handed to the scaffold (omp maxTokens + compaction reserve)")
    p.add_argument("--thinking-level", default="off",
                   help="omp --thinking level (off|minimal|low|medium|high|xhigh|max|auto); off puts "
                        "enable_thinking=false on the wire, which Qwen3.6 honours and the Qwen3.8 "
                        "templates ignore")
    p.add_argument("--disable-tools", default="",
                   help="comma-separated scaffold tools to switch off (omp: subtracted from "
                        f"{OMP_DEFAULT_TOOLS}; task and web_search are always off)")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   help="HF dataset id (Lite=300, Verified=500)")
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=0,
                   help="Limit to first N instances (0 = all)")
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="Specific instance IDs to run (overrides --instances)")
    p.add_argument("--out", required=True, help="Output dir for predictions + logs")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-instance opencode timeout (seconds)")
    p.add_argument("--workdir", default="/tmp/swebench-work",
                   help="Where to clone task repos")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip instances that already have a prediction")
    p.add_argument("--server-url", default="http://127.0.0.1:23334",
                   help="SGLang server base URL (used for preflight)")
    p.add_argument("--served-name", default=None,
                   help="Served model name on the server (defaults to model id after slash)")
    p.add_argument("--max-empty-streak", type=int, default=10,
                   help="Abort if this many consecutive instances produce empty diffs")
    p.add_argument("--venvdir", default="/tmp/swebench-venvs",
                   help="Where to cache per-instance uv venvs (shared with score_local)")
    p.add_argument("--no-venv", action="store_true",
                   help="Skip pre-rollout venv setup — agent runs read-edit-pray "
                        "without a working test loop. Compatible with v1 runs.")
    p.add_argument("--prompt-style", choices=["default", "bash"], default="default",
                   help="Task prompt variant: 'bash' describes a bash-only tool set (for runs "
                        "that disable opencode's read/edit/write tools, as Flint's own harnesses do).")
    p.add_argument("--prompt-extra", default="",
                   help="Text appended to the task prompt (harness-level rule for a specific "
                        "model, e.g. a required-argument reminder). Record it in the run README.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the tool-call-round-trip canary. Required for chat "
                        "templates that don't accept the canary's user→assistant"
                        "(tool_calls)→tool→user message sequence (e.g. Mistral).")
    return p.parse_args()


OMP_BIN = Path(__file__).resolve().parent / "bin" / "omp"
# omp's full tool set is read bash edit eval glob grep task wait todo web_search
# write. `task` (sub-agents: many completions, no edit inside the window on a
# 13-80 tok/s server) and `web_search` (the answer channel) are off, as they
# are for opencode; the rest is the opencode-equivalent read/edit/search set
# plus omp's REPL (`eval`) and `wait`.
OMP_DEFAULT_TOOLS = "read,bash,edit,eval,glob,grep,wait,todo,write"


def omp_version() -> str:
    try:
        return subprocess.run([str(OMP_BIN), "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or "omp/?"
    except Exception as e:  # noqa: BLE001
        return f"omp/? ({type(e).__name__})"


def write_omp_agent_dir(out: Path, model_url: str, served: str, context_window: int,
                        output_budget: int) -> Path:
    """Per-run omp agent dir (PI_CODING_AGENT_DIR): the provider entry for the
    served model and the harness settings. Written here, so the user's own
    ~/.omp/agent is never touched and a new preset needs no static entry.

    reasoning: true keeps omp's reasoning-model handling (reasoning_content,
    thinking-aware turn shaping); compat.supportsReasoningEffort: false keeps
    reasoning_effort off the wire (the template decides). --no-title (on the
    command line) is load-bearing: omp's title generator forks a side turn
    that replays the whole prefix on the served model whenever the todo tool
    initialises a plan, headless too — a second full prefill per step.
    """
    agent = out / "omp-agent"
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "models.yml").write_text(
        "providers:\n"
        "  sglang:\n"
        "    name: SGLang local\n"
        "    api: openai-completions\n"
        f"    baseUrl: {model_url.rstrip('/')}/v1\n"
        '    apiKey: "not-needed"\n'
        "    models:\n"
        f"      - id: {json.dumps(served)}\n"
        f"        name: {json.dumps(served + ' (SGLang local)')}\n"
        "        api: openai-completions\n"
        "        reasoning: true\n"
        "        compat:\n"
        "          supportsReasoningEffort: false\n"
        "        input: [text]\n"
        "        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}\n"
        f"        contextWindow: {int(context_window)}\n"
        f"        maxTokens: {int(output_budget)}\n"
    )
    (agent / "config.yml").write_text(
        "compaction:\n"
        f"  reserveTokens: {int(output_budget)}\n"
        "startup:\n"
        "  checkUpdate: false\n"
        "title:\n"
        "  refreshOnReplan: false\n"
    )
    return agent


def run_omp(served: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
            agent_dir: Path, session_dir: Path, tools: str, thinking: str,
            extra_env: dict[str, str] | None = None,
            server_url: str | None = None) -> tuple[int, str, str]:
    """One omp session in the worktree: task on stdin, JSON event stream on
    stdout (logged), session jsonl under session_dir (pi row format, read by
    audit_rollouts.py)."""
    session_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(OMP_BIN),
        "--model", f"sglang/{served}",
        "--yolo", "--no-title", "-p", "--mode", "json",
        "--cwd", str(repo_dir),
        "--session-dir", str(session_dir),
        "--tools", tools,
        "--thinking", thinking,
    ]
    if sys.platform == "darwin" and os.environ.get("SWEBENCH_SANDBOX", "1") != "0":
        profile = Path(__file__).resolve().parent / "localhost-only.sb"
        cmd = ["sandbox-exec", "-f", str(profile)] + cmd
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["DO_NOT_TRACK"] = "1"
    env["OTEL_SDK_DISABLED"] = "true"
    if extra_env:
        venv_path = extra_env.get("PATH")
        if venv_path:
            env["PATH"] = venv_path + ":" + env.get("PATH", "")
        for k, v in extra_env.items():
            if k != "PATH":
                env[k] = v
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=repo_dir, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        rc = proc.returncode
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command\n{' '.join(cmd)} < PROMPT\n# elapsed {elapsed:.1f}s\n"
            f"# returncode {rc}\n# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        abort_all_requests(server_url)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        log_path.write_text(f"# TIMEOUT after {timeout}s (process group SIGKILLed)\n# stdout\n{stdout}\n# stderr\n{stderr}\n")
        return 124, stdout or "", stderr or ""


def preflight_canary(server_url: str, served_name: str) -> tuple[bool, str]:
    """Send a chat request that mimics opencode's wire format — assistant turn
    with prior tool_calls (arguments as JSON string per OpenAI spec) — to catch
    chat-template bugs BEFORE burning hours on rollouts.
    """
    import urllib.request
    payload = {
        "model": served_name,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "glob",
                                          "arguments": '{"pattern": "**/*.py"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "a.py\nb.py"},
            {"role": "user", "content": "continue"},
        ],
        "max_tokens": 30,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            if "choices" in body and body["choices"]:
                content = body["choices"][0]["message"].get("content") or ""
                return True, f"OK ({len(content)}B content)"
            return False, f"unexpected response: {body!r}"
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())["message"]
        except Exception:
            err = str(e)
        return False, f"{e.code}: {err}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_dataset(dataset_id: str, split: str):
    from datasets import load_dataset
    return load_dataset(dataset_id, split=split)


def ensure_repo(repo: str, base_commit: str, work_root: Path, instance_id: str) -> Path:
    """Clone <repo> at <base_commit> into work_root/<instance_id>. Idempotent.

    Uses a shared mirror at work_root/.mirrors/<repo> to avoid re-fetching the
    same repo across instances of the same project.
    """
    mirror = work_root / ".mirrors" / repo.replace("/", "__")
    inst_dir = work_root / instance_id

    if mirror.exists():
        # A mirror under /tmp can lose its files to the periodic tmp cleanup
        # while the directory skeleton stays (hooks/ info/ objects/ and no
        # HEAD): treat anything git does not recognise as absent.
        ok = subprocess.run(["git", "-C", str(mirror), "rev-parse", "--git-dir"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not ok:
            print(f"  mirror {mirror} is not a git repository — re-cloning", flush=True)
            shutil.rmtree(mirror, ignore_errors=True)
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(mirror)],
            check=True,
        )

    if inst_dir.exists():
        # Rename-then-delete: rmtree on a tmpfs entry can SIGSEGV the process
        # (observed at django__django-11797 — kernel corruption left invisible
        # entries that ls/find skip but rmdir/rm refuse). Renaming gets the
        # path out of the way so clone succeeds even if cleanup fails; the
        # orphaned trash dir gets reaped on next reboot.
        trash = inst_dir.with_name(inst_dir.name + f".trash.{int(time.time())}")
        try:
            inst_dir.rename(trash)
        except OSError:
            pass
        try:
            shutil.rmtree(trash, ignore_errors=True)
        except Exception:
            pass
        # Belt and suspenders: if rename failed and the dir still exists, abort.
        if inst_dir.exists():
            raise RuntimeError(f"could not clear {inst_dir}; tmpfs corruption?")
    # Ancestor-only checkout: fetch exactly base_commit from the mirror into
    # a fresh repository, so the object store holds the commit and its
    # history and nothing after it — no branches, tags or remotes that lead
    # to the upstream fix (`git log --all`, `git branch -a`, `git show <sha>`
    # of a later commit all come back empty). A plain clone of the mirror
    # carried the whole history: 50,492 commits reachable on astropy where
    # the base commit reaches 32,966, the fix merge among them.
    subprocess.run(["git", "-C", str(mirror), "config", "uploadpack.allowAnySHA1InWant", "true"],
                   check=True)
    inst_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=inst_dir, check=True)
    fetch = ["git", "fetch", "-q", f"file://{mirror}", base_commit]
    r = subprocess.run(fetch, cwd=inst_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        # The mirror predates this base commit: refresh it once and retry.
        subprocess.run(["git", "-C", str(mirror), "fetch", "-q", "origin"], check=True)
        subprocess.run(fetch, cwd=inst_dir, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "base", "FETCH_HEAD"], cwd=inst_dir, check=True)
    reachable = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=inst_dir,
                               capture_output=True, text=True, check=True).stdout.strip()
    everything = subprocess.run(["git", "rev-list", "--count", "--all"], cwd=inst_dir,
                                capture_output=True, text=True, check=True).stdout.strip()
    if reachable != everything:
        raise RuntimeError(f"{instance_id}: checkout holds commits past the base ({everything} vs {reachable})")
    subprocess.run(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    subprocess.run(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


PROMPT_TEMPLATE = """\
You are working on a GitHub issue in this repository.

The repo is already installed in editable mode in a virtual environment that
is active on your PATH. You can run `pytest` and `python -c "..."` to verify
imports, exercise edge cases, and re-run tests after each edit. Use this:
write a fix, run the relevant tests, observe failures, refine until green.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct AND the tests
exercise it correctly, stop — your final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""

PROMPT_NO_VENV = """\
You are working on a GitHub issue in this repository. The repo's dependencies
could NOT be installed locally, so `pytest` and `import` will not work — you
must reason about correctness from the source alone.

**You have direct file-system access via tool calls.** The working directory
is the cloned repo at the relevant base commit. Use the `read` tool to view
any file's contents, the `glob` and `grep` tools to navigate, and the `edit`
or `write` tools to apply your fix. **Never ask the user for file contents —
you can read them yourself.** Iterate: read the file, understand the bug,
edit it, then verify by re-reading. Do not stop after a single tool call.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct, stop — your
final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""


PROMPT_NO_VENV_BASH = """\
You are working on a GitHub issue in this repository. The repo's dependencies
could NOT be installed locally, so `pytest` and `import` will not work — you
must reason about correctness from the source alone.

**Your only tool is `bash`**, run in the cloned repo at the relevant base
commit. Use it to inspect files (`sed -n`, `grep -n`, `cat`) and to apply
your fix (`python - <<'EOF'` with `pathlib` read/replace/write, or `sed -i`),
then re-read the changed lines to verify. **Never ask the user for file
contents — you can read them yourself.** Iterate: read the code, understand
the bug, edit it, then verify by re-reading. Do not stop after a single tool
call.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct, stop — your
final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""


def abort_all_requests(server_url: str | None) -> None:
    """Drop every in-flight request on the model server.

    A rollout killed at its timeout leaves its last turn running on the
    server; on the MLX backend that orphan keeps the scheduler busy for
    the rest of its prefill and the next instance's health check reads it
    as a dead server.
    """
    if not server_url:
        return
    import urllib.request
    body = json.dumps({"abort_all": True}).encode()
    req = urllib.request.Request(f"{server_url}/abort_request", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


def run_opencode(model: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
                 extra_env: dict[str, str] | None = None,
                 server_url: str | None = None) -> tuple[int, str, str]:
    cmd = [
        "opencode", "run",
        "--dir", str(repo_dir),
        "--model", model,
        "--format", "json",
        "--dangerously-skip-permissions",
        prompt,
    ]
    # Network seal: on macOS the whole opencode process tree (its bash tool
    # included) runs under a seatbelt profile that allows connections to
    # this host only, so curl/pip/git/webfetch cannot reach GitHub or PyPI
    # for a later version of the code. SWEBENCH_SANDBOX=0 lifts it.
    if sys.platform == "darwin" and os.environ.get("SWEBENCH_SANDBOX", "1") != "0":
        profile = Path(__file__).resolve().parent / "localhost-only.sb"
        cmd = ["sandbox-exec", "-f", str(profile)] + cmd
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.npm-global/bin:{env.get('PATH','')}"
    if extra_env:
        # PATH from extra_env (venv/bin) goes BEFORE npm-global (and host /usr/bin)
        # so the model's `pytest`/`python` resolves to the venv-installed
        # versions during its tool calls.
        venv_path = extra_env.get("PATH")
        if venv_path:
            env["PATH"] = venv_path + ":" + env["PATH"]
        for k, v in extra_env.items():
            if k != "PATH":
                env[k] = v
    t0 = time.time()
    # Run in a fresh process group so SIGKILL on timeout reaps the Node spawned
    # children too (default subprocess kill leaves them dangling — observed at
    # instance 23 where the parent died but a child kept the rollout stalled).
    # stdin must be closed explicitly: `opencode run` reads a non-TTY stdin to
    # EOF before it sends its first model request, so an inherited pipe or
    # socket that stays open (background launchers, task runners) parks the
    # rollout at 0% CPU until the per-instance timeout fires.
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command\n{' '.join(cmd[:-1])} <PROMPT>\n# elapsed {elapsed:.1f}s\n"
            f"# returncode {rc}\n# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        abort_all_requests(server_url)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        log_path.write_text(f"# TIMEOUT after {timeout}s (process group SIGKILLed)\n# stdout\n{stdout}\n# stderr\n{stderr}\n")
        return 124, stdout or "", stderr or ""


def capture_diff(repo_dir: Path) -> str:
    # Stage everything modified, untracked, deleted; capture diff against HEAD.
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = subprocess.run(
        ["git", "diff", "--cached"], cwd=repo_dir, capture_output=True, text=True, check=True
    )
    return res.stdout


def main():
    args = parse_args()

    served = args.served_name or args.model.split("/", 1)[-1]
    if args.scaffold == "omp":
        if not OMP_BIN.exists():
            print(f"omp binary missing at {OMP_BIN} — run: bash evals/swebench/vendor_omp.sh", flush=True)
            sys.exit(2)
        disabled = {t.strip() for t in args.disable_tools.split(",") if t.strip()}
        omp_tools = ",".join(t for t in OMP_DEFAULT_TOOLS.split(",") if t not in disabled)
        scaffold_pin = omp_version()
        print(f"Scaffold: {scaffold_pin}  tools={omp_tools}  thinking={args.thinking_level}  "
              f"context_window={args.context_window}  output_budget={args.output_budget}  "
              f"model_url={args.model_url or args.server_url}", flush=True)
    else:
        scaffold_pin = "opencode"
        omp_tools = ""
    if args.skip_preflight:
        print(f"Preflight: SKIPPED (--skip-preflight). Server health unchecked.", flush=True)
    else:
        print(f"Preflight: canary chat completion against {args.server_url} (model={served})...", flush=True)
        ok, info = preflight_canary(args.server_url, served)
        if not ok:
            print(f"  PREFLIGHT FAILED: {info}", flush=True)
            print(f"  refusing to start rollout — fix the server / chat template first", flush=True)
            print(f"  (use --skip-preflight if you know the model's template doesn't accept the canary)", flush=True)
            sys.exit(2)
        print(f"  preflight {info}", flush=True)

    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    omp_agent_dir = None
    if args.scaffold == "omp":
        omp_agent_dir = write_omp_agent_dir(out, args.model_url or args.server_url, served,
                                            args.context_window, args.output_budget)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset}/{args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.split)
    print(f"  {len(ds)} instances total", flush=True)

    if args.instance_ids:
        ds = [r for r in ds if r["instance_id"] in args.instance_ids]
        print(f"  filtered to {len(ds)} via --instance-ids", flush=True)
    elif args.instances:
        ds = list(ds)[: args.instances]
        print(f"  truncated to first {len(ds)} via --instances", flush=True)

    predictions_path = out / "predictions.jsonl"
    existing = set()
    if args.skip_existing and predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass

    empty_streak = 0
    with predictions_path.open("a") as fp:
        for i, row in enumerate(ds):
            iid = row["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(ds)}] {iid}  SKIP (exists)", flush=True)
                continue

            print(f"[{i+1}/{len(ds)}] {iid}  repo={row['repo']}  base={row['base_commit'][:8]}", flush=True)
            t0 = time.time()

            # Per-instance health check: catch jetsam / mid-sweep server death
            # before burning a 900s+ rollout window. Without this, opencode
            # hangs waiting for chat-completion responses that will never come,
            # producing a series of zero-byte "model failures" that are
            # actually infrastructure failures.
            #
            # Discovered 2026-05-18 when a 5-instance cross-ecosystem sweep
            # of qwen36 had its server jetsam'd ~10 min in; the next 4
            # instances ran against a dead upstream and all returned 0 bytes
            # at 900s wall, contaminating the patch-engagement rate.
            #
            # Uses the lightweight /health endpoint rather than the full
            # tool-call-canary so the check is liveness-only and decoupled
            # from chat-template validation (Mistral-arch templates reject
            # the canary even when the server is healthy).
            # /health generates one token through the scheduler, so a server
            # that is still prefilling (an orphaned turn, a chunk of a dense
            # model at ~20 s) times out exactly like a reaped one. A refused
            # connection is a dead server; a timeout is retried for
            # SWEBENCH_HEALTH_WAIT seconds (default 120) before the verdict.
            import urllib.request
            health_budget = float(os.environ.get("SWEBENCH_HEALTH_WAIT", "120"))
            h0 = time.time()
            while True:
                try:
                    with urllib.request.urlopen(f"{args.server_url}/health", timeout=10) as r:
                        ok = (r.status == 200)
                        hinfo = f"/health HTTP/{r.status}"
                except Exception as e:
                    ok = False
                    hinfo = f"{type(e).__name__}: {e}"
                    refused = isinstance(getattr(e, "reason", None), ConnectionRefusedError) \
                        or isinstance(e, ConnectionRefusedError)
                    if not refused and time.time() - h0 < health_budget:
                        print(f"  health: {hinfo} — server busy? retrying "
                              f"({time.time() - h0:.0f}/{health_budget:.0f} s)", flush=True)
                        time.sleep(5)
                        continue
                break
            if not ok:
                print(f"  SERVER_DEAD: {hinfo} — aborting sweep", flush=True)
                fp.write(json.dumps({
                    "instance_id": iid,
                    "model_name_or_path": args.model,
                    "model_patch": "",
                    "rollout_returncode": -2,
                    "server_dead": True,
                    "server_dead_info": hinfo,
                    "rollout_seconds": round(time.time() - t0, 1),
                }) + "\n")
                fp.flush()
                print(f"\nABORT: server health check failed before instance {i+1}/{len(ds)}. "
                      f"Subsequent instances would also fail. Restart the server "
                      f"(and check macOS Activity Monitor for jetsam events) then "
                      f"rerun with --skip-existing.", flush=True)
                sys.exit(4)

            try:
                try:
                    inst_dir = ensure_repo(row["repo"], row["base_commit"], workdir, iid)
                except subprocess.CalledProcessError as e:
                    print(f"  CLONE FAIL: {e}", flush=True)
                    continue

                # Pre-rollout venv setup so the model can run pytest mid-iteration.
                # If install fails we still attempt the rollout (read-edit-pray fallback),
                # but the prompt warns the model that tests aren't available.
                venv = None
                if not args.no_venv:
                    from eval_env import make_venv, install_deps
                    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
                    spec = MAP_REPO_VERSION_TO_SPECS.get(row["repo"], {}).get(row["version"])
                    if spec:
                        env_log = out / "logs" / f"{iid}.env.log"
                        try:
                            v = make_venv(Path(args.venvdir), iid, spec.get("python", "3.11"))
                            if install_deps(v, inst_dir, spec, env_log):
                                venv = v
                                print(f"  env: venv ready ({spec.get('python', '3.11')}, {len(spec.get('pip_packages', []))} pkgs)", flush=True)
                            else:
                                print(f"  env: install FAILED — falling back to no-venv prompt", flush=True)
                        except subprocess.CalledProcessError as e:
                            print(f"  env: venv setup crashed: {e} — falling back", flush=True)

                if venv:
                    template = PROMPT_TEMPLATE
                elif args.prompt_style == "bash":
                    template = PROMPT_NO_VENV_BASH
                else:
                    template = PROMPT_NO_VENV
                prompt = template.format(
                    problem_statement=row["problem_statement"],
                    hints=row.get("hints_text", "") or "(none)",
                )
                if args.prompt_extra:
                    prompt += "\n\n" + args.prompt_extra.strip() + "\n"
                log_path = out / "logs" / f"{iid}.log"
                extra_env = {}
                if venv:
                    extra_env = {
                        "VIRTUAL_ENV": str(venv),
                        "PATH": f"{venv}/bin",
                    }
                if args.scaffold == "omp":
                    rc, _stdout, _stderr = run_omp(served, inst_dir, prompt, args.timeout, log_path,
                                                   omp_agent_dir, out / "sessions" / iid, omp_tools,
                                                   args.thinking_level, extra_env=extra_env,
                                                   server_url=args.server_url)
                else:
                    rc, _stdout, _stderr = run_opencode(args.model, inst_dir, prompt, args.timeout,
                                                        log_path, extra_env=extra_env,
                                                        server_url=args.server_url)
                diff = capture_diff(inst_dir)
                (out / "predictions" / f"{iid}.diff").write_text(diff)

                entry = {
                    "instance_id": iid,
                    "model_name_or_path": args.model,
                    "model_patch": diff,
                    "rollout_returncode": rc,
                    "rollout_seconds": round(time.time() - t0, 1),
                    "scaffold": scaffold_pin,
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()

                non_empty = "yes" if diff.strip() else "EMPTY"
                print(f"  done rc={rc} elapsed={entry['rollout_seconds']}s diff={non_empty} ({len(diff)}B)", flush=True)

                if diff.strip():
                    empty_streak = 0
                else:
                    empty_streak += 1
                    if empty_streak >= args.max_empty_streak:
                        print(f"\nABORT: {empty_streak} consecutive empty diffs — likely a server/template/permissions regression. "
                              f"Re-run preflight before resuming.", flush=True)
                        sys.exit(3)
            except Exception as e:
                # Per-instance safety net: log and skip so one bad task can't kill 300.
                import traceback
                print(f"  SKIP (instance crashed): {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                fp.write(json.dumps({"instance_id": iid, "model_name_or_path": args.model,
                                     "model_patch": "", "rollout_returncode": -1,
                                     "rollout_error": f"{type(e).__name__}: {e}",
                                     "rollout_seconds": round(time.time() - t0, 1)}) + "\n")
                fp.flush()
                continue


if __name__ == "__main__":
    sys.exit(main())
