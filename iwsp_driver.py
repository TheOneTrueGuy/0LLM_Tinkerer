"""
iWSP autoresearch driver — OpenRouter port of upstream launch_claude.py.

Runs the interpretable-weight-space-programming loop without Claude Code:
an OpenRouter chat model gets confined file tools (read/write/list/grep
inside a stamped run folder) plus `run_eval` which executes
`python interpretable_transformer.py --task <task>` in that folder. The
driver repeats sessions until results/overall_results.csv reaches the
target attempt count, or two sessions in a row add nothing.

Usage:
    python iwsp_driver.py --task parity-upto10-bits --model deepseek/deepseek-chat-v3.1 \
        --target 5 --max-sessions 3

Run folders: iwsp_runs/<model_short>/<task>-<model_short>-s<seed>/
Each contains the vendored harness copy, results/, agent_logs/, metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import wsp_agent

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.join(HERE, "iwsp_harness")
RUNS_ROOT = os.path.join(HERE, "iwsp_runs")

TASKS = {
    "addition-five-digits": "easy", "multiplication-five-digits": "easy",
    "sort-five-digits": "easy", "digit-counting-10": "easy",
    "parity-upto10-bits": "easy", "boolean-circuit-5-bits": "easy",
    "linear-interpolation-two-points": "easy",
    "word-reversal-3x3": "harder", "gcd-three-digits": "harder",
    "decimal-to-binary-8bit": "harder", "edit-distance-8": "harder",
    "smallest-prime-factor": "harder",
    "sentiment-sst2": "real", "paraphrase-mrpc": "real", "nli-snli": "real",
    "trec-coarse": "real", "date-normalization": "real",
}
TARGET = dict(easy=30, harder=40, real=50)
MAX_SESSIONS = 25
MAX_TURNS_PER_SESSION = 60          # hard cap on tool-call rounds per session
EVAL_TIMEOUT_S = 600
READ_CAP = 60_000                   # chars returned to the model per read
OUT_CAP = 12_000                    # chars returned per tool result

READONLY = ("src/", "program.md")   # agent may not overwrite harness files

SYSTEM = (
    "You are an autonomous research agent. You hand-write transformer "
    "weights in closed form to solve a task — no training, no gradient "
    "steps. Follow program.md exactly, iterate experiment after "
    "experiment, and never ask the human whether to continue."
)

FIRST = (
    "Read and follow the instructions in program.md. Your task name is "
    "`{task}`. Notes on this harness: the run folder has already been "
    "created and your file tools are confined to it, so skip the setup "
    "steps that create it; experiments run on CPU. Your tools: read_file, "
    "write_file, list_files, grep — all paths relative to the run folder "
    "— plus run_eval (the ONLY way to execute: it runs `python "
    "interpretable_transformer.py --task {task}`; you may pass extra args "
    "like --n-samples 50 for faster iteration) and log_result (append a "
    "row to results/overall_results.csv — use it after a crash since the "
    "eval script only logs on success). src/ and program.md are "
    "read-only. To learn the task format you may read src/task.py "
    "directly. Keep looping: do at least {target} experiments (each one "
    "logged as a row of results/overall_results.csv); there are {n} so "
    "far."
)
CONT = (
    "Continue the autoresearch loop from program.md for task `{task}`. "
    "results/overall_results.csv now has {n} experiments; keep going "
    "until there are at least {target}, then keep simplifying if time "
    "allows. Same rules and tools as before."
)


# --------------------------------------------------------------------------
# run folder setup
# --------------------------------------------------------------------------

def setup_run(run_dir: str, task: str) -> None:
    """Stamp a fresh copy of the harness into run_dir and fill the task name."""
    if os.path.exists(run_dir):
        raise FileExistsError(run_dir)
    shutil.copytree(HARNESS_DIR, run_dir, ignore=shutil.ignore_patterns("README.md"))
    for d in ("interpretable_transformers_lib", "agent_logs"):
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    pm = os.path.join(run_dir, "program.md")
    s = open(pm, encoding="utf-8").read()
    s = re.sub(r"your task name is `[^`]+`", f"your task name is `{task}`", s, count=1)
    open(pm, "w", encoding="utf-8").write(s)


def n_attempts(run_dir: str) -> int:
    p = os.path.join(run_dir, "results", "overall_results.csv")
    if not os.path.exists(p):
        return 0
    with open(p, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def best_accuracy(run_dir: str) -> float:
    p = os.path.join(run_dir, "results", "overall_results.csv")
    best = 0.0
    if not os.path.exists(p):
        return best
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            try:
                best = max(best, float(row["accuracy"]))
            except (TypeError, ValueError):
                continue
    return best


# --------------------------------------------------------------------------
# confined tool surface
# --------------------------------------------------------------------------

def _confined(run_dir: str, rel: str) -> str:
    rel = (rel or ".").strip().replace("\\", "/")
    if rel.startswith("/") or rel.startswith("~"):
        raise ValueError("absolute paths are not allowed")
    p = os.path.normpath(os.path.join(run_dir, rel))
    if os.path.commonpath([os.path.abspath(run_dir), os.path.abspath(p)]) != os.path.abspath(run_dir):
        raise ValueError(f"path escapes run folder: {rel!r}")
    return p


def _is_readonly(rel: str) -> bool:
    rel = rel.strip().replace("\\", "/")
    return any(rel == r.rstrip("/") or rel.startswith(r) for r in READONLY)


def make_iwsp_tools() -> list[dict]:
    def fn(name, desc, props=None, required=None):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": props or {},
                           "required": required or []}}}
    return [
        fn("read_file", "Read a file inside the run folder",
           {"path": {"type": "string"}}, ["path"]),
        fn("write_file", "Write (overwrite) a file inside the run folder. "
           "src/ and program.md are read-only",
           {"path": {"type": "string"}, "content": {"type": "string"}},
           ["path", "content"]),
        fn("list_files", "List files in the run folder (or a subdir)",
           {"path": {"type": "string", "description": "subdir, default '.'"}}),
        fn("grep", "Regex search across files in the run folder; returns "
           "file:line: text",
           {"pattern": {"type": "string"},
            "path": {"type": "string", "description": "subdir, default '.'"}},
           ["pattern"]),
        fn("run_eval", "Run the evaluator: `python "
           "interpretable_transformer.py --task <task> [extra_args]`. "
           "extra_args is a space-separated arg string like '--n-samples 50 "
           "--verbose'. Returns exit code + stdout/stderr tail",
           {"extra_args": {"type": "string", "default": ""}}),
        fn("log_result", "Append one row to results/overall_results.csv — "
           "use after a crash (the eval script logs successes itself)",
           {"status": {"type": "string", "enum": ["success", "crash"]},
            "model_shorthand_name": {"type": "string"},
            "accuracy": {"type": "string"},
            "n_params": {"type": "string"},
            "description": {"type": "string"}},
           ["status", "model_shorthand_name", "description"]),
    ]


def make_iwsp_executor(run_dir: str, task: str):
    """Execute tool calls, confined to run_dir."""
    def execute(name: str, args: dict) -> str:
        try:
            if name == "read_file":
                p = _confined(run_dir, args.get("path", ""))
                if not os.path.isfile(p):
                    return json.dumps({"error": f"no file {args.get('path')!r}"})
                s = open(p, encoding="utf-8", errors="replace").read()
                if len(s) > READ_CAP:
                    s = s[:READ_CAP] + f"\n... [truncated at {READ_CAP} chars]"
                return s or "(empty file)"
            if name == "write_file":
                rel = args.get("path", "")
                if _is_readonly(rel):
                    return json.dumps({"error": f"{rel!r} is read-only"})
                p = _confined(run_dir, rel)
                os.makedirs(os.path.dirname(p) or run_dir, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(args.get("content", ""))
                return json.dumps({"written": rel,
                                   "bytes": len(args.get("content", ""))})
            if name == "list_files":
                p = _confined(run_dir, args.get("path", "."))
                out = []
                for root, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in ("agent_logs", "__pycache__")]
                    for fn_ in files:
                        out.append(os.path.relpath(os.path.join(root, fn_), run_dir))
                        if len(out) >= 400:
                            break
                return json.dumps({"files": sorted(out)})
            if name == "grep":
                p = _confined(run_dir, args.get("path", "."))
                pat = re.compile(args.get("pattern", ""), re.IGNORECASE)
                hits = []
                for root, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for fn_ in files:
                        fp = os.path.join(root, fn_)
                        try:
                            for i, line in enumerate(open(fp, encoding="utf-8", errors="replace"), 1):
                                if pat.search(line):
                                    hits.append(f"{os.path.relpath(fp, run_dir)}:{i}: {line.strip()[:200]}")
                                    if len(hits) >= 60:
                                        return json.dumps({"matches": hits, "truncated": True})
                        except OSError:
                            continue
                return json.dumps({"matches": hits})
            if name == "run_eval":
                extra = args.get("extra_args") or ""
                try:
                    extra_argv = shlex.split(extra)
                except ValueError as e:
                    return json.dumps({"error": f"could not parse extra_args: {e}"})
                if not all(os.path.basename(a) == a for a in extra_argv):
                    return json.dumps({"error": "invalid args"})
                argv = [sys.executable, "interpretable_transformer.py",
                        "--task", task] + extra_argv
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                try:
                    r = subprocess.run(argv, cwd=run_dir, capture_output=True,
                                       text=True, timeout=EVAL_TIMEOUT_S,
                                       env=env)
                except subprocess.TimeoutExpired:
                    return json.dumps({"error": f"timed out after {EVAL_TIMEOUT_S}s"})
                tail = (r.stdout + r.stderr)[-OUT_CAP:]
                return json.dumps({"exit_code": r.returncode, "output": tail})
            if name == "log_result":
                p = os.path.join(run_dir, "results", "overall_results.csv")
                with open(p, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=[
                        "task", "accuracy", "status",
                        "model_shorthand_name", "n_params",
                        "description"]).writerow({
                        "task": task,
                        "accuracy": args.get("accuracy", ""),
                        "status": args.get("status", "crash"),
                        "model_shorthand_name": args.get("model_shorthand_name", ""),
                        "n_params": args.get("n_params", ""),
                        "description": args.get("description", "")})
                return json.dumps({"logged": True})
            return json.dumps({"error": f"unknown tool {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return execute


# --------------------------------------------------------------------------
# session + driver loops
# --------------------------------------------------------------------------

def run_session(run_dir: str, task: str, or_model: str, prompt: str,
                execute, k: int, temperature: float, max_turns: int) -> dict:
    """One agent session: tool-call loop until the model stops or the turn
    cap hits. Appends transcript lines to agent_logs/session_<k>.jsonl."""
    log_path = os.path.join(run_dir, "agent_logs", f"session_{k:02d}.jsonl")
    messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    turns = 0
    with open(log_path, "w", encoding="utf-8") as lf:
        while turns < max_turns:
            resp = wsp_agent.chat(or_model, messages, make_iwsp_tools(),
                                  temperature)
            msg = resp["choices"][0]["message"]
            messages.append(msg)
            u = resp.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            usage["completion_tokens"] += u.get("completion_tokens", 0)
            lf.write(json.dumps({"type": "assistant", "message": msg},
                                default=str) + "\n")
            calls = msg.get("tool_calls") or []
            turns += 1
            if not calls:
                break
            for c in calls:
                fname = c["function"]["name"]
                try:
                    fargs = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    fargs = {}
                print(f"    [{turns}] {fname}({json.dumps(fargs)[:100]})", flush=True)
                result = execute(fname, fargs)
                lf.write(json.dumps({"type": "tool", "name": fname,
                                     "args": fargs, "result": result[:4000]},
                                    default=str) + "\n")
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": result[:OUT_CAP]})
            lf.flush()
    return {"turns": turns, "usage": usage,
            "log": os.path.relpath(log_path, run_dir)}


def run_one(task: str, seed: int, or_model: str, runs_root: str = RUNS_ROOT,
            target: int | None = None, max_sessions: int = MAX_SESSIONS,
            max_turns: int = MAX_TURNS_PER_SESSION,
            temperature: float = 0.4) -> dict:
    short = or_model.split("/")[-1].replace(":", "-")
    tag = f"{task}-{short}-s{seed}"
    runs_subdir = os.path.join(runs_root, short)
    run_dir = os.path.join(runs_subdir, tag)
    if not os.path.exists(run_dir):
        os.makedirs(runs_subdir, exist_ok=True)
        setup_run(run_dir, task)
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else dict(
        run=tag, task=task, seed=seed, model=or_model,
        driver="iwsp_driver.py (OpenRouter tools)",
        target=target or TARGET[TASKS[task]],
        sessions=[], started=datetime.now(timezone.utc).isoformat())
    meta["target"] = target or meta["target"]
    stalls = 0
    execute = make_iwsp_executor(run_dir, task)
    while len(meta["sessions"]) < max_sessions:
        n0 = n_attempts(run_dir)
        if n0 >= meta["target"]:
            break
        prompt = (FIRST if not meta["sessions"] else CONT).format(
            task=task, target=meta["target"], n=n0)
        k = len(meta["sessions"])
        t0 = time.time()
        try:
            sess = run_session(run_dir, task, or_model, prompt, execute, k,
                               temperature, max_turns)
        except Exception as e:
            sess = {"turns": 0, "usage": {}, "log": "", "error": str(e)}
        n1 = n_attempts(run_dir)
        meta["sessions"].append(dict(
            k=k, exit_error=sess.get("error"), turns=sess["turns"],
            usage=sess.get("usage"), seconds=round(time.time() - t0),
            attempts_before=n0, attempts_after=n1, log=sess["log"]))
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"{tag} session {k}: {n0} -> {n1} attempts, "
              f"{sess['turns']} turns, {round(time.time()-t0)}s", flush=True)
        stalls = stalls + 1 if n1 == n0 else 0
        if stalls >= 2:
            break
    meta.update(finished=datetime.now(timezone.utc).isoformat(),
                attempts=n_attempts(run_dir), best_acc=best_accuracy(run_dir))
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"RESULT {tag}: {meta['attempts']} attempts, "
          f"best_acc={meta['best_acc']:.4f}, {len(meta['sessions'])} sessions")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017",
                    help="OpenRouter model id")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", type=int, default=None,
                    help="attempt count (default per difficulty: "
                    "easy 30 / harder 40 / real 50; use small for smoke tests)")
    ap.add_argument("--max-sessions", type=int, default=MAX_SESSIONS)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS_PER_SESSION,
                    help="tool-call rounds per session")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    a = ap.parse_args()
    run_one(a.task, a.seed, a.model, runs_root=a.runs_root, target=a.target,
            max_sessions=a.max_sessions, max_turns=a.max_turns,
            temperature=a.temperature)
