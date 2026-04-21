"""_judge_backend.py — shared helpers for the CC-bridge judge backend.

The three LLM-using verifiers (`verify_citations_full`, `verify_internal_contradiction`,
and `verify_math_sympy`'s LLM fallback) accept a `--judge-backend` flag with
three values:

- ``sdk`` — call the Anthropic SDK directly (the historical behavior). Requires
  ``ANTHROPIC_API_KEY`` and the ``anthropic`` package.
- ``batch-emit`` — instead of calling the LLM, write one JSONL task per pending
  judgment to ``--judge-tasks-out`` and stash the verifier's partial state
  (resolved refs, pending targets) to ``--state-file``. The verifier exits 0
  with an interim JSON report marked ``status=pending_llm``. Downstream, a
  Claude Code session dispatches subagents for each task and writes results
  to a JSONL file.
- ``batch-ingest`` — read ``--state-file`` and ``--judge-results-in`` (a JSONL
  file keyed by ``task_id``) and produce the final verifier report. The
  contradiction verifier additionally uses this phase to emit a second wave
  of compare tasks after triple extraction.

This module contains just the contract plumbing — each verifier owns the
mapping between its own pending targets and the opaque task bodies.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BACKEND_CHOICES = ("sdk", "batch-emit", "batch-ingest")


@dataclass
class BackendContext:
    """Resolved CC-bridge backend configuration for a single verifier run.

    ``mode`` is one of ``sdk`` / ``batch-emit`` / ``batch-ingest``. ``sdk_ok``
    is cheap to check at construction time so the verifier can decide if it
    should fall back to a stub. ``tasks_out_secondary`` is used by the
    contradiction verifier in batch-ingest mode when it needs to emit a
    second-wave compare task file.
    """

    mode: str
    tasks_out: Path | None = None
    results_in: Path | None = None
    state_file: Path | None = None
    sdk_ok: bool = False
    tasks_out_secondary: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sdk(self) -> bool:
        return self.mode == "sdk"

    @property
    def is_emit(self) -> bool:
        return self.mode == "batch-emit"

    @property
    def is_ingest(self) -> bool:
        return self.mode == "batch-ingest"


def resolve_mode(explicit: str | None) -> str:
    """Pick a backend mode given an explicit CLI choice (may be None)."""
    if explicit in BACKEND_CHOICES:
        return explicit
    # Default: prefer SDK when key + package are present; otherwise batch-emit.
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore  # noqa: F401
            return "sdk"
        except ImportError:
            pass
    return "batch-emit"


def sdk_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def add_backend_args(parser) -> None:
    """Attach the shared CLI flags to an argparse parser."""
    parser.add_argument(
        "--judge-backend", choices=BACKEND_CHOICES, default=None,
        help=("Which LLM-judge backend to use. Default: sdk when "
              "ANTHROPIC_API_KEY is set, otherwise batch-emit (writes task "
              "manifests for Claude Code subagents)."),
    )
    parser.add_argument(
        "--judge-tasks-out", type=Path, default=None,
        help="Where to write the JSONL of pending LLM tasks (batch-emit mode).",
    )
    parser.add_argument(
        "--judge-results-in", type=Path, default=None,
        help="JSONL of previously-produced LLM results (batch-ingest mode).",
    )
    parser.add_argument(
        "--state-file", type=Path, default=None,
        help=("JSON file storing verifier partial state between emit and "
              "ingest phases (batch-emit / batch-ingest modes)."),
    )


def build_context(args) -> BackendContext:
    mode = resolve_mode(getattr(args, "judge_backend", None))
    return BackendContext(
        mode=mode,
        tasks_out=getattr(args, "judge_tasks_out", None),
        results_in=getattr(args, "judge_results_in", None),
        state_file=getattr(args, "state_file", None),
        sdk_ok=sdk_available(),
    )


def emit_task(tasks_out: Path, task: dict[str, Any]) -> None:
    """Append a single task object as one JSONL line."""
    tasks_out.parent.mkdir(parents=True, exist_ok=True)
    with tasks_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(task, ensure_ascii=False) + "\n")


def write_tasks(tasks_out: Path, tasks: list[dict[str, Any]]) -> None:
    """Overwrite the tasks file with ``tasks`` (one JSON object per line)."""
    tasks_out.parent.mkdir(parents=True, exist_ok=True)
    with tasks_out.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")


def load_results(results_in: Path) -> dict[str, dict[str, Any]]:
    """Read a JSONL results file; return ``task_id -> result`` dict.

    Malformed or non-JSON lines are skipped with a warning on stderr. Missing
    files return an empty dict (caller must decide whether that is fatal).
    """
    out: dict[str, dict[str, Any]] = {}
    if not results_in or not results_in.is_file():
        return out
    for line in results_in.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        tid = obj.get("task_id")
        if isinstance(tid, str) and tid:
            out[tid] = obj
    return out


def parse_json_body(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the `body` string from a result dict and parse as JSON.

    Falls back to a ``{"verdict": "insufficient_context", "confidence": "low",
    "reason": "..."}``-shaped dict on any failure so callers never need to
    defend against raw KeyError / JSONDecodeError.
    """
    if not result.get("ok", True):
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": f"subagent reported error: {result.get('error', 'unknown')}",
            "parse_error": result.get("error"),
        }
    body = result.get("body") or ""
    if isinstance(body, dict):
        return body
    if not isinstance(body, str):
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "result body is neither dict nor string",
            "parse_error": f"type={type(body).__name__}",
        }
    match = re.search(r"\{.*\}", body, flags=re.DOTALL)
    if not match:
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "subagent response had no JSON object",
            "parse_error": "no JSON object found",
            "raw_body": body[:500],
        }
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "subagent response malformed JSON",
            "parse_error": str(exc),
            "raw_body": body[:500],
        }


def read_state(state_file: Path) -> dict[str, Any]:
    """Load a state JSON file; raise FileNotFoundError if missing."""
    if not state_file or not state_file.is_file():
        raise FileNotFoundError(f"state file not found: {state_file}")
    return json.loads(state_file.read_text(encoding="utf-8"))


def write_state(state_file: Path, obj: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def build_task(
    *,
    task_id: str,
    verifier: str,
    target_locator: str,
    kind: str,
    system: str,
    user: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
    expected_format: str = "json",
) -> dict[str, Any]:
    """Construct a task JSON object with the standard schema."""
    return {
        "task_id": task_id,
        "verifier": verifier,
        "target_locator": target_locator,
        "kind": kind,
        "system": system,
        "user": user,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "expected_format": expected_format,
    }


def ensure_emit_args(ctx: BackendContext) -> None:
    if ctx.tasks_out is None or ctx.state_file is None:
        raise ValueError(
            "batch-emit mode requires --judge-tasks-out and --state-file")


def ensure_ingest_args(ctx: BackendContext) -> None:
    if ctx.results_in is None or ctx.state_file is None:
        raise ValueError(
            "batch-ingest mode requires --judge-results-in and --state-file")
