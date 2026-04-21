#!/usr/bin/env python3
"""cc_run_round.py — Claude Code orchestrator for the v0.4 CC-bridge flow.

Runs one round of `review-verification-harness` in an environment without
`ANTHROPIC_API_KEY`. The LLM work is deferred to subagents dispatched by the
Claude Code session.

Usage:
    cc_run_round.py PAPER.tex [--bib PAPER.bib] --workspace DIR
        [--phase {emit,finalize}] [--fast] [--round N]

Emit phase (default):
    - Copies PAPER.tex into workspace/rN/tex/
    - Runs preflight, extract_claims, verify_internal_consistency (all deterministic)
    - Runs the three LLM verifiers in `--judge-backend batch-emit` mode,
      producing a task JSONL per verifier.
    - Writes a manifest JSON to stdout.
    - Exits 7 if any task file is non-empty (pending LLM work) or 0 otherwise.

Finalize phase:
    - For each verifier with a state file, invokes it in `--judge-backend batch-ingest`
      mode. If the contradiction verifier needs a second wave, exits 7 so the
      caller re-dispatches.
    - Otherwise writes `cc_bridge_summary.json` and exits 0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZE = 10

EXIT_READY = 0
EXIT_PENDING = 7
EXIT_FAIL = 1


# ----------------------------- helpers -----------------------------


def _round_dir(workspace: Path, n: int) -> Path:
    # If the workspace already ends with an rN directory, respect it.
    if workspace.name.startswith("r") and workspace.name[1:].isdigit():
        return workspace
    return workspace / f"r{n}"


def _ensure_dirs(round_dir: Path) -> None:
    for sub in ("tex", "claims", "verifier", "judge_tasks",
                "judge_results", "state"):
        (round_dir / sub).mkdir(parents=True, exist_ok=True)


def _count_task_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
               if ln.strip())


def _suggested_batch_size(task_count: int) -> int:
    if task_count <= 0:
        return DEFAULT_BATCH_SIZE
    return min(DEFAULT_BATCH_SIZE, max(1, task_count))


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a subprocess, return (returncode, combined_output)."""
    res = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, check=False,
    )
    return res.returncode, (res.stdout or "") + (res.stderr or "")


# ----------------------------- emit phase -----------------------------


def phase_emit(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    paper = Path(args.paper).resolve()
    if not paper.is_file():
        print(f"error: paper file not found: {paper}", file=sys.stderr)
        return EXIT_FAIL
    bib = Path(args.bib).resolve() if args.bib else None
    if bib and not bib.is_file():
        print(f"error: bib file not found: {bib}", file=sys.stderr)
        return EXIT_FAIL

    round_dir = _round_dir(workspace, args.round)
    _ensure_dirs(round_dir)

    # Snapshot paper + bib.
    tex_dest = round_dir / "tex" / paper.name
    shutil.copyfile(paper, tex_dest)
    bib_dest = None
    if bib:
        bib_dest = round_dir / "tex" / bib.name
        shutil.copyfile(bib, bib_dest)

    claims_dir = round_dir / "claims"
    verifier_dir = round_dir / "verifier"
    tasks_dir = round_dir / "judge_tasks"
    state_dir = round_dir / "state"

    # Preflight — best-effort, never fatal in emit mode.
    preflight_json = round_dir / "_preflight.json"
    rc, out = _run([
        sys.executable, str(SCRIPTS_DIR / "_preflight.py"), "--json",
    ])
    try:
        preflight_json.write_text(out, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    # extract_claims (deterministic, no LLM)
    extract_cmd = [
        sys.executable, str(SCRIPTS_DIR / "extract_claims.py"),
        "--tex", str(tex_dest),
        "--out-dir", str(claims_dir),
    ]
    if bib_dest:
        extract_cmd += ["--bib", str(bib_dest)]
    rc, out = _run(extract_cmd)
    # Non-fatal; continue even if claims extraction produced no files.

    # scope (no LLM)
    _run([
        sys.executable, str(SCRIPTS_DIR / "verify_internal_consistency.py"),
        "--tex", str(tex_dest),
        "--out", str(verifier_dir / "scope.json"),
    ])

    tasks_manifest: list[dict[str, Any]] = []

    # 1) verify_citations_full --judge-backend batch-emit
    citations_tasks = tasks_dir / "citations.jsonl"
    citations_state = state_dir / "citations.json"
    cite_out = verifier_dir / "citations.json"
    cite_claims = claims_dir / "citation_claims.jsonl"
    if cite_claims.is_file():
        cite_cmd = [
            sys.executable, str(SCRIPTS_DIR / "verify_citations_full.py"),
            "--claims", str(cite_claims),
            "--out", str(cite_out),
            "--judge-backend", "batch-emit",
            "--judge-tasks-out", str(citations_tasks),
            "--state-file", str(citations_state),
        ]
        if args.fast:
            cite_cmd.append("--fast")
        _run(cite_cmd)
        n_tasks = _count_task_lines(citations_tasks)
        if n_tasks > 0 or citations_state.is_file():
            tasks_manifest.append({
                "verifier": "verify_citations_full",
                "tasks_file": str(citations_tasks.relative_to(workspace))
                              if citations_tasks.is_file() else None,
                "state_file": str(citations_state.relative_to(workspace))
                              if citations_state.is_file() else None,
                "out_file": str(cite_out.relative_to(workspace))
                            if cite_out.is_file() else None,
                "n_tasks": n_tasks,
                "suggested_batch_size": _suggested_batch_size(n_tasks),
                "phase": "primary_judge",
            })

    # 2) verify_internal_contradiction --judge-backend batch-emit
    contra_tasks = tasks_dir / "contradiction_triples.jsonl"
    contra_state = state_dir / "contradiction.json"
    contra_out = verifier_dir / "contradiction.json"
    contra_cmd = [
        sys.executable, str(SCRIPTS_DIR / "verify_internal_contradiction.py"),
        "--tex", str(tex_dest),
        "--out", str(contra_out),
        "--judge-backend", "batch-emit",
        "--judge-tasks-out", str(contra_tasks),
        "--state-file", str(contra_state),
    ]
    if args.fast:
        contra_cmd.append("--fast")
    _run(contra_cmd)
    n_contra = _count_task_lines(contra_tasks)
    if n_contra > 0 or contra_state.is_file():
        tasks_manifest.append({
            "verifier": "verify_internal_contradiction",
            "tasks_file": str(contra_tasks.relative_to(workspace))
                          if contra_tasks.is_file() else None,
            "state_file": str(contra_state.relative_to(workspace))
                          if contra_state.is_file() else None,
            "out_file": str(contra_out.relative_to(workspace))
                        if contra_out.is_file() else None,
            "n_tasks": n_contra,
            "suggested_batch_size": _suggested_batch_size(n_contra),
            "phase": "triples",
        })

    # 3) verify_math_sympy --judge-backend batch-emit
    math_tasks = tasks_dir / "math.jsonl"
    math_state = state_dir / "math.json"
    math_out = verifier_dir / "math.json"
    math_cmd = [
        sys.executable, str(SCRIPTS_DIR / "verify_math_sympy.py"),
        "--tex", str(tex_dest),
        "--out", str(math_out),
        "--judge-backend", "batch-emit",
        "--judge-tasks-out", str(math_tasks),
        "--state-file", str(math_state),
    ]
    _run(math_cmd)
    n_math = _count_task_lines(math_tasks)
    if n_math > 0 or math_state.is_file():
        tasks_manifest.append({
            "verifier": "verify_math_sympy",
            "tasks_file": str(math_tasks.relative_to(workspace))
                          if math_tasks.is_file() else None,
            "state_file": str(math_state.relative_to(workspace))
                          if math_state.is_file() else None,
            "out_file": str(math_out.relative_to(workspace))
                        if math_out.is_file() else None,
            "n_tasks": n_math,
            "suggested_batch_size": _suggested_batch_size(n_math),
            "phase": "math_llm_fallback",
        })

    total_tasks = sum(t.get("n_tasks", 0) or 0 for t in tasks_manifest)

    manifest = {
        "round": args.round,
        "workspace": str(workspace),
        "round_dir": str(round_dir),
        "phase": "emit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks_manifest,
        "total_llm_tasks": total_tasks,
        "next_action": (
            f"Dispatch each task file through CC subagents; store results in "
            f"{round_dir / 'judge_results'} as <verifier>.results.jsonl; then "
            f"re-invoke {SCRIPTS_DIR / 'cc_run_round.py'} --phase finalize "
            f"--workspace {workspace} --round {args.round}"),
        "template_path": str(SCRIPTS_DIR / "cc_dispatch_template.md"),
    }
    manifest_path = round_dir / "cc_bridge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if total_tasks > 0:
        return EXIT_PENDING
    print("no LLM tasks pending; run with --phase finalize or treat as complete",
          file=sys.stderr)
    return EXIT_READY


# ----------------------------- finalize phase -----------------------------


def phase_finalize(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    round_dir = _round_dir(workspace, args.round)
    if not round_dir.is_dir():
        print(f"error: round dir not found: {round_dir}", file=sys.stderr)
        return EXIT_FAIL

    state_dir = round_dir / "state"
    results_dir = round_dir / "judge_results"
    tasks_dir = round_dir / "judge_tasks"
    verifier_dir = round_dir / "verifier"

    # Map verifier name -> (state, results, out, script, [second_wave_tasks])
    verifiers = [
        ("verify_citations_full", state_dir / "citations.json",
         results_dir / "verify_citations_full.results.jsonl",
         verifier_dir / "citations.json",
         SCRIPTS_DIR / "verify_citations_full.py",
         tasks_dir / "citations.compare.jsonl"),
        ("verify_internal_contradiction",
         state_dir / "contradiction.json",
         results_dir / "verify_internal_contradiction.results.jsonl",
         verifier_dir / "contradiction.json",
         SCRIPTS_DIR / "verify_internal_contradiction.py",
         tasks_dir / "contradiction_compare.jsonl"),
        ("verify_math_sympy", state_dir / "math.json",
         results_dir / "verify_math_sympy.results.jsonl",
         verifier_dir / "math.json",
         SCRIPTS_DIR / "verify_math_sympy.py",
         tasks_dir / "math.second.jsonl"),
    ]

    second_wave: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"verifiers": []}
    any_pending = False

    for name, state, results, out, script, second_tasks in verifiers:
        if not state.is_file():
            continue
        if not results.is_file():
            print(f"warning: results file missing for {name}: {results}",
                  file=sys.stderr)
            summary["verifiers"].append({
                "verifier": name,
                "status": "missing_results",
                "state_file": str(state),
                "expected_results": str(results),
            })
            any_pending = True
            continue
        cmd = [
            sys.executable, str(script),
            "--out", str(out),
            "--judge-backend", "batch-ingest",
            "--judge-results-in", str(results),
            "--state-file", str(state),
            "--judge-tasks-out", str(second_tasks),
        ]
        rc, output = _run(cmd)
        # Re-read the verifier's own JSON output to determine if it emitted
        # another wave of tasks.
        verifier_report: dict[str, Any] = {}
        if out.is_file():
            try:
                verifier_report = json.loads(
                    out.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError:
                verifier_report = {}
        summary["verifiers"].append({
            "verifier": name,
            "status": verifier_report.get("status"),
            "severity": verifier_report.get("severity_suggestion"),
            "summary": verifier_report.get("summary"),
            "out_file": str(out),
            "exit_code": rc,
        })
        if verifier_report.get("status") == "pending_llm":
            any_pending = True
            if second_tasks.is_file():
                n_tasks = _count_task_lines(second_tasks)
                second_wave.append({
                    "verifier": name,
                    "tasks_file": str(second_tasks),
                    "state_file": str(state),
                    "n_tasks": n_tasks,
                    "suggested_batch_size": _suggested_batch_size(n_tasks),
                    "phase": "compare",
                })

    summary["round"] = args.round
    summary["round_dir"] = str(round_dir)
    summary["phase"] = "finalize"
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["second_wave"] = second_wave

    summary_path = round_dir / "cc_bridge_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return EXIT_PENDING if any_pending else EXIT_READY


# ----------------------------- CLI -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Claude Code orchestrator for review-verification-harness CC-bridge",
    )
    parser.add_argument("paper", nargs="?", help="Paper .tex file (emit phase)")
    parser.add_argument("--bib", help="BibTeX file for citations")
    parser.add_argument("--workspace", required=True,
                        help="Workspace directory (rN layout)")
    parser.add_argument("--phase", choices=("emit", "finalize"),
                        default="emit",
                        help="emit (default) or finalize")
    parser.add_argument("--round", type=int, default=1,
                        help="Round number (for rN/ layout)")
    parser.add_argument("--fast", action="store_true",
                        help="Pass --fast to verifiers that support it")
    args = parser.parse_args()

    if args.phase == "emit":
        if not args.paper:
            print("error: paper argument required in emit phase", file=sys.stderr)
            return EXIT_FAIL
        return phase_emit(args)
    return phase_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
