"""_preflight.py — environment health check for the review-verification harness.

Determines, before any verifier is invoked, which verifiers can actually do
their work and which will degrade to `unverifiable` purely because the local
environment is missing a dependency (API key, network, optional package).

Distinguishing "paper is actually broken" from "the host is misconfigured" is
the main fix shipping in version 0.3-stageB2: downstream consumers route
`unverifiable_kind=env` to `disposition=setup_needed` rather than
`gate_blocker`, so false P0s from missing deps stop drowning real signal.

Checks performed:
  * `ANTHROPIC_API_KEY` env var (presence only — never validated via an API
    call; `valid=None`).
  * `anthropic` package importable.
  * `sympy` importable; version logged.
  * `antlr4-python3-runtime` importable (sympy's LaTeX parser needs it).
  * `CrossRef` reachable (`GET /works/10.1038/nature12373`, 5s timeout).
  * `arXiv` reachable (`GET /api/query?id_list=1701.01234`, 5s timeout).

CLI:
    python _preflight.py                  # human-readable report
    python _preflight.py --json           # raw JSON
    exit 0: ready
    exit 1: degraded (safe to run verifiers — some will unverifiable_env)
    exit 2: setup_incomplete (abort — cannot even run)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

PREFLIGHT_VERSION = "0.1"

NETWORK_TIMEOUT = 5.0
CROSSREF_PROBE = "https://api.crossref.org/works/10.1038/nature12373"
ARXIV_PROBE = "https://export.arxiv.org/api/query?id_list=1701.01234"
USER_AGENT = "review-verification-harness-preflight/0.1 (mailto:yudabitrends@gmail.com)"

# Mapping: which verifier is affected when which check fails.
# `kind` distinguishes env (API key / network / dep) vs tool (sympy parse gap).
# Only `env`-level misses are listed here — `tool`-level (e.g. macro-expansion
# coverage) is intrinsic to the verifier and can't be detected by preflight.
_VERIFIER_EFFECTS = {
    "anthropic_api_key": [
        ("verify_citations_full", "will_unverifiable_env",
         "LLM judge cannot run without ANTHROPIC_API_KEY; "
         "citation content-match will be stubbed."),
        ("verify_internal_contradiction", "will_unverifiable_env",
         "Contradiction extraction requires an LLM; "
         "no ANTHROPIC_API_KEY → top-level unverifiable."),
    ],
    "anthropic_sdk": [
        ("verify_citations_full", "will_unverifiable_env",
         "`anthropic` SDK not importable; judge calls are stubbed."),
        ("verify_internal_contradiction", "will_unverifiable_env",
         "`anthropic` SDK not importable; contradiction verifier cannot run."),
        ("verify_math_sympy", "will_unverifiable_env_fallback",
         "LLM math fallback disabled (no `anthropic`); sympy-only coverage."),
    ],
    "sympy": [
        ("verify_math_sympy", "will_unverifiable_env",
         "sympy not installed; equation verifier will emit top-level "
         "unverifiable with unverifiable_kind=env."),
    ],
    "antlr4_runtime": [
        ("verify_math_sympy", "will_unverifiable_tool",
         "antlr4-python3-runtime missing; sympy's LaTeX parser will fail "
         "for most real equations — per-equation unverifiable_kind=tool."),
    ],
    "network_crossref": [
        ("verify_citations_full", "will_unverifiable_env",
         "CrossRef unreachable; DOIs cannot be resolved."),
    ],
    "network_arxiv": [
        ("verify_citations_full", "will_unverifiable_env",
         "arXiv unreachable; arXiv IDs cannot be resolved (Semantic "
         "Scholar fallback may still work)."),
    ],
}


# ---------------- individual checks ----------------


def _check_anthropic_api_key() -> dict[str, Any]:
    present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return {
        "present": present,
        "valid": None,
        "note": (
            "ANTHROPIC_API_KEY is set (presence only — not validated)."
            if present else "ANTHROPIC_API_KEY not set; LLM-using verifiers "
            "will stub their judge calls."
        ),
    }


def _check_importable(module: str, *, friendly_name: str | None = None) -> dict[str, Any]:
    name = friendly_name or module
    spec = importlib.util.find_spec(module)
    if spec is None:
        return {
            "importable": False,
            "version": None,
            "note": f"{name} is not importable in this interpreter.",
        }
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        return {
            "importable": False,
            "version": None,
            "note": f"{name} import raised: {type(exc).__name__}: {exc}",
        }
    version = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
    if version is not None:
        version = str(version)
    return {
        "importable": True,
        "version": version,
        "note": (f"{name} importable"
                 f"{f' (version {version})' if version else ''}."),
    }


def _check_anthropic_sdk() -> dict[str, Any]:
    return _check_importable("anthropic", friendly_name="anthropic SDK")


def _check_sympy() -> dict[str, Any]:
    return _check_importable("sympy")


def _check_antlr4_runtime() -> dict[str, Any]:
    # The package name is antlr4-python3-runtime; the import name is antlr4.
    return _check_importable("antlr4", friendly_name="antlr4-python3-runtime")


def _probe_network(url: str, label: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as resp:
            code = resp.getcode()
            # Read a handful of bytes so we confirm data flow, then close.
            resp.read(1024)
        latency = int((time.monotonic() - t0) * 1000)
        ok = code == 200
        return {
            "reachable": ok,
            "latency_ms": latency,
            "note": (
                f"{label} reachable (HTTP {code}, {latency} ms)."
                if ok else f"{label} responded HTTP {code} (latency {latency} ms)."
            ),
        }
    except urllib.error.HTTPError as exc:
        latency = int((time.monotonic() - t0) * 1000)
        return {
            "reachable": False,
            "latency_ms": latency,
            "note": f"{label} HTTP error: {exc.code} {exc.reason}.",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        latency = int((time.monotonic() - t0) * 1000)
        return {
            "reachable": False,
            "latency_ms": latency,
            "note": f"{label} unreachable: {type(exc).__name__}: {exc}.",
        }
    except Exception as exc:  # noqa: BLE001
        latency = int((time.monotonic() - t0) * 1000)
        return {
            "reachable": False,
            "latency_ms": latency,
            "note": f"{label} probe crashed: {type(exc).__name__}: {exc}.",
        }


def _check_network_crossref() -> dict[str, Any]:
    return _probe_network(CROSSREF_PROBE, "CrossRef")


def _check_network_arxiv() -> dict[str, Any]:
    return _probe_network(ARXIV_PROBE, "arXiv")


# ---------------- aggregation ----------------


def _collect_affected(checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    affected: list[dict[str, Any]] = []

    def _failed(key: str) -> bool:
        c = checks.get(key) or {}
        if key == "anthropic_api_key":
            return not c.get("present", False)
        if key in ("anthropic_sdk", "sympy", "antlr4_runtime"):
            return not c.get("importable", False)
        if key in ("network_crossref", "network_arxiv"):
            return not c.get("reachable", False)
        return False

    for key, entries in _VERIFIER_EFFECTS.items():
        if not _failed(key):
            continue
        for verifier, status, reason in entries:
            affected.append({
                "verifier": verifier,
                "status": status,
                "reason": reason,
                "triggered_by": key,
            })
    return affected


def _derive_status(checks: dict[str, dict[str, Any]],
                   affected: list[dict[str, Any]]) -> str:
    # Setup-incomplete: cannot even run the verifier machinery. In practice we
    # only hit this if the interpreter is missing basic stdlib paths (should
    # be impossible here) or if `run_preflight` itself crashed upstream.
    # The preflight function always runs under CPython with stdlib, so we
    # reserve setup_incomplete for the "no Python importable, no json, no
    # urllib" cases, which are covered by the harness refusing to load at all.
    if not affected:
        return "ready"
    # Everything missing here is degraded — the harness still produces
    # readable JSON, just with more unverifiable_kind=env targets.
    return "degraded"


def _summary_line(status: str, checks: dict[str, dict[str, Any]],
                  affected: list[dict[str, Any]]) -> str:
    if status == "ready":
        return (
            f"preflight: ready. API key present, "
            f"sympy={checks['sympy'].get('version') or 'ok'}, "
            "network+crossref+arxiv reachable."
        )
    missing_labels: list[str] = []
    if not checks["anthropic_api_key"].get("present"):
        missing_labels.append("ANTHROPIC_API_KEY")
    if not checks["anthropic_sdk"].get("importable"):
        missing_labels.append("anthropic SDK")
    if not checks["sympy"].get("importable"):
        missing_labels.append("sympy")
    if not checks["antlr4_runtime"].get("importable"):
        missing_labels.append("antlr4-python3-runtime")
    if not checks["network_crossref"].get("reachable"):
        missing_labels.append("CrossRef")
    if not checks["network_arxiv"].get("reachable"):
        missing_labels.append("arXiv")
    verifiers = sorted({a["verifier"] for a in affected})
    return (
        f"preflight: {status}. Missing: {', '.join(missing_labels) or '(none)'}"
        f". Affected verifiers: {', '.join(verifiers) or '(none)'}."
    )


def run_preflight() -> dict[str, Any]:
    """Execute every environment check and return a summary dict.

    See module docstring for the exact contract. Never raises — each individual
    probe is wrapped so a crash in (say) `urllib` does not take the whole
    preflight down, because the preflight script is run synchronously from
    `run_round.sh` at the top of every pipeline.
    """
    checks: dict[str, dict[str, Any]] = {}
    probes = [
        ("anthropic_api_key", _check_anthropic_api_key),
        ("anthropic_sdk", _check_anthropic_sdk),
        ("sympy", _check_sympy),
        ("antlr4_runtime", _check_antlr4_runtime),
        ("network_crossref", _check_network_crossref),
        ("network_arxiv", _check_network_arxiv),
    ]
    for name, fn in probes:
        try:
            checks[name] = fn()
        except Exception as exc:  # noqa: BLE001
            checks[name] = {
                "importable": False, "present": False, "reachable": False,
                "valid": None, "version": None, "latency_ms": None,
                "note": f"preflight probe {name!r} crashed: "
                        f"{type(exc).__name__}: {exc}",
            }

    affected = _collect_affected(checks)
    status = _derive_status(checks, affected)
    summary = _summary_line(status, checks, affected)

    return {
        "preflight_version": PREFLIGHT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "affected_verifiers": affected,
        "summary": summary,
    }


def _format_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"[preflight] status: {report['status']}")
    lines.append(f"[preflight] {report['summary']}")
    lines.append("")
    lines.append("Checks:")
    for key, check in report["checks"].items():
        if "present" in check:
            marker = "OK " if check["present"] else "MISS"
            detail = check.get("note", "")
        elif "importable" in check:
            marker = "OK " if check["importable"] else "MISS"
            detail = check.get("note", "")
        elif "reachable" in check:
            marker = "OK " if check["reachable"] else "MISS"
            detail = check.get("note", "")
        else:
            marker = "?   "
            detail = str(check)
        lines.append(f"  [{marker}] {key}: {detail}")
    if report["affected_verifiers"]:
        lines.append("")
        lines.append("Affected verifiers (env/tool coverage will degrade):")
        for entry in report["affected_verifiers"]:
            lines.append(
                f"  - {entry['verifier']}: {entry['status']} "
                f"(trigger={entry['triggered_by']}) — {entry['reason']}"
            )
    return "\n".join(lines)


def _exit_code_for(status: str) -> int:
    return {"ready": 0, "degraded": 1, "setup_incomplete": 2}.get(status, 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preflight environment checks for the review-verification harness.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON report on stdout (for machine consumption).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional path to write the JSON report (used by run_round.sh).",
    )
    args = parser.parse_args()
    report = run_preflight()

    if args.out:
        out_path = args.out
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"[preflight] warning: failed to write {out_path}: {exc}",
                  file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(_format_human(report))
    return _exit_code_for(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
