"""verify_math_sympy.py — symbolic / numerical equation check for displayed math.

Catches silent algebra, sign, and identity errors in displayed equations
(taxonomy class U1: "algebra goes wrong but looks plausible"). LLM reviewers
almost never catch these; sympy does so reliably.

Pipeline:
  1. Extract displayed equations from a LaTeX file
     (`\\[..\\]`, `$$..$$`, equation/align/eqnarray/gather/multline envs,
     align/eqnarray/gather bodies are split on `\\\\`).
  2. Split each on top-level `=` into (lhs, rhs) pairs. Multi-step chains
     `A = B = C` become `(A,B)` and `(B,C)`. Skip `:=`, `==`, and `=` nested
     inside braces/brackets/parens (so `x_{i=1}` is safe).
  3. For each pair, parse both sides with `sympy.parsing.latex.parse_latex`.
     a. Try symbolic `simplify(lhs - rhs) == 0`.
     b. If inconclusive, draw `--seeds` (default 3) random real-valued
        substitutions for free symbols in [0.1, 2.0] with a seeded RNG,
        compare with relative tolerance.
     c. On failure, test sign-flip heuristic `simplify(lhs + rhs) == 0` and
        report "sign-flip detected".

Graceful degradation: if sympy cannot be imported, emit a top-level
`status=unverifiable` severity `P0`. A per-equation exception is caught and
the target is marked `unverifiable` — a single bad equation never crashes
the run. Output conforms to `references/VERIFIER_CONTRACT.md`.

Usage:
    python verify_math_sympy.py --tex paper.tex --out workspace/verifier/math.json
    python verify_math_sympy.py --tex paper.tex --out math.json --seeds 5 --tolerance 1e-6
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import signal
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERIFIER_ID = "verify_math_sympy"
VERIFIER_VERSION = "0.1"


class _SympyTimeout(Exception):
    """Raised when a per-equation sympy operation exceeds the timeout."""


@contextmanager
def _equation_timeout(seconds: int = 5):
    """Per-equation wall-clock guard. POSIX-only (uses signal.SIGALRM).

    Raises `_SympyTimeout` after `seconds`. On non-POSIX platforms (Windows)
    `signal.SIGALRM` is absent; in that case this context manager becomes a
    no-op rather than failing to import.
    """
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _SympyTimeout(f"sympy operation exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# Operators / macros sympy.parsing.latex cannot reliably handle. Presence of
# any on either side → skip the pair as unverifiable rather than produce a
# misleading parse.
UNPARSEABLE_TOKENS = (
    r"\int", r"\oint", r"\iint", r"\iiint",
    r"\sum", r"\prod",
    r"\lim", r"\liminf", r"\limsup",
    r"\partial", r"\nabla", r"\Delta",
    r"\mathrm{d}", r"\,d", r"\mathbb",
    r"\mathcal", r"\mathbf", r"\mathscr",
    r"\langle", r"\rangle",
    r"\det", r"\tr", r"\Tr",
    r"\hat", r"\tilde", r"\overline", r"\underline", r"\bar",
    r"\text", r"\mbox",
)

MAX_EQ_QUOTE_CHARS = 200
RNG_SEED = 42


def _sympy_available() -> tuple[bool, str | None]:
    try:
        import sympy  # noqa: F401
        from sympy.parsing.latex import parse_latex  # noqa: F401
        return True, None
    except ImportError as exc:
        return False, f"sympy missing: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"sympy unavailable: {exc}"


# ----------------------------- LaTeX equation extraction -----------------------------


_DISPLAY_ENVS = [
    # (env_name, split_on_linebreak)
    ("equation", False), ("equation*", False),
    ("align", True), ("align*", True),
    ("eqnarray", True), ("eqnarray*", True),
    ("gather", True), ("gather*", True),
    ("multline", False), ("multline*", False),
]
_BRACKET_DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_DOLLAR_DISPLAY = re.compile(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", re.DOTALL)


def _strip_comments(tex: str) -> str:
    """Drop LaTeX comments so we do not parse commented-out equations."""
    return re.sub(r"(?<!\\)%[^\n]*", "", tex)


def extract_equations(tex: str) -> list[dict[str, Any]]:
    """Return ordered list of displayed-equation items with `body`, `raw`, `start`."""
    tex = _strip_comments(tex)
    items: list[dict[str, Any]] = []
    for m in _DOLLAR_DISPLAY.finditer(tex):
        items.append({"raw": m.group(0), "body": m.group(1), "start": m.start(), "line_index": 0})
    for m in _BRACKET_DISPLAY.finditer(tex):
        items.append({"raw": m.group(0), "body": m.group(1), "start": m.start(), "line_index": 0})
    for env, split_lines in _DISPLAY_ENVS:
        start_re = re.compile(r"\\begin\{" + re.escape(env) + r"\}")
        end_re = re.compile(r"\\end\{" + re.escape(env) + r"\}")
        pos = 0
        while True:
            s = start_re.search(tex, pos)
            if not s:
                break
            e = end_re.search(tex, s.end())
            if not e:
                break
            body = tex[s.end():e.start()]
            raw = tex[s.start():e.end()]
            if split_lines:
                for i, line in enumerate(_split_on_latex_newline(body)):
                    if line.strip():
                        items.append({"raw": raw, "body": line, "start": s.start(), "line_index": i})
            else:
                items.append({"raw": raw, "body": body, "start": s.start(), "line_index": 0})
            pos = e.end()
    items.sort(key=lambda d: (d["start"], d["line_index"]))
    return items


def _split_on_latex_newline(body: str) -> list[str]:
    r"""Split a LaTeX math body on `\\` at brace-depth 0. Skip `\\[1ex]` length arg."""
    parts, current, depth, i = [], [], 0, 0
    while i < len(body):
        ch = body[i]
        if ch == "{":
            depth += 1; current.append(ch); i += 1; continue
        if ch == "}":
            depth = max(0, depth - 1); current.append(ch); i += 1; continue
        if depth == 0 and ch == "\\" and i + 1 < len(body) and body[i + 1] == "\\":
            parts.append("".join(current)); current = []; i += 2
            if i < len(body) and body[i] == "[":
                close = body.find("]", i)
                if close != -1:
                    i = close + 1
            continue
        current.append(ch); i += 1
    parts.append("".join(current))
    return parts


# ----------------------------- Equality splitting -----------------------------


def split_equalities(body: str) -> list[tuple[str, str]]:
    r"""Split `A = B = C` into [(A,B),(B,C)]. Skip `:=`, `==`, `\neq`, `\equiv`,
    and any `=` inside braces/parens/brackets or sub/super-scripts like `x_{i=1}`."""
    cleaned = body.replace("&", " ")
    cleaned = re.sub(r"\\label\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\tag\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\nonumber", " ", cleaned)
    segs = _split_top_level_equals(cleaned)
    if len(segs) < 2:
        return []
    return [(a.strip(), b.strip()) for a, b in zip(segs, segs[1:]) if a.strip() and b.strip()]


def _split_top_level_equals(text: str) -> list[str]:
    """Split on `=` at brace/paren/bracket depth 0. Skip `:=`, `==`, `\\neq`, `\\leq`, `\\geq`."""
    out: list[str] = []
    depth, i, n = 0, 0, len(text)
    current: list[str] = []
    while i < n:
        ch = text[i]
        if ch in "{[(":
            depth += 1; current.append(ch); i += 1; continue
        if ch in "}])":
            depth = max(0, depth - 1); current.append(ch); i += 1; continue
        if ch == "\\":
            m = re.match(r"\\[A-Za-z]+", text[i:])
            if m:
                current.append(m.group(0)); i += len(m.group(0)); continue
            current.append(ch); i += 1; continue
        if depth == 0 and ch == "=":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            if prev == ":" or prev == "=" or nxt == "=":
                current.append(ch); i += 1; continue
            out.append("".join(current)); current = []; i += 1; continue
        current.append(ch); i += 1
    out.append("".join(current))
    return out


def unparseable_hits(body: str) -> list[str]:
    return [tok for tok in UNPARSEABLE_TOKENS if tok in body]


def is_definition(body: str) -> bool:
    return ":=" in body or r"\equiv" in body or r"\coloneqq" in body


# ----------------------------- Sympy-based verification -----------------------------


def verify_pair(
    lhs_tex: str, rhs_tex: str, *,
    seeds: int, tolerance: float, rng: random.Random,
) -> dict[str, Any]:
    """Parse both sides, try symbolic check, then numerical. Return a detail dict."""
    import sympy  # type: ignore
    from sympy.parsing.latex import parse_latex  # type: ignore

    detail: dict[str, Any] = {
        "lhs_tex": lhs_tex[:MAX_EQ_QUOTE_CHARS], "rhs_tex": rhs_tex[:MAX_EQ_QUOTE_CHARS],
        "lhs": None, "rhs": None, "free_symbols": [],
        "symbolic_equal": None, "seeds_evaluated": 0, "seed_results": [],
        "sign_flip_detected": False, "status": "unverifiable", "judge_notes": "",
    }
    try:
        lhs_expr = parse_latex(lhs_tex)
    except Exception as exc:  # noqa: BLE001
        detail["judge_notes"] = f"LaTeX parse failed on LHS: {exc}"; return detail
    try:
        rhs_expr = parse_latex(rhs_tex)
    except Exception as exc:  # noqa: BLE001
        detail["judge_notes"] = f"LaTeX parse failed on RHS: {exc}"; return detail
    if lhs_expr is None or rhs_expr is None:
        detail["judge_notes"] = "parse_latex returned None for one side"; return detail
    detail["lhs"], detail["rhs"] = str(lhs_expr), str(rhs_expr)

    try:
        free_syms = sorted(
            set(lhs_expr.free_symbols) | set(rhs_expr.free_symbols), key=lambda s: str(s),
        )
    except Exception as exc:  # noqa: BLE001
        detail["judge_notes"] = f"free_symbols unavailable: {exc}"; return detail
    detail["free_symbols"] = [str(s) for s in free_syms]

    # 1) Symbolic check
    try:
        diff = sympy.simplify(lhs_expr - rhs_expr)
        if diff == 0 or diff.equals(0):
            detail["symbolic_equal"] = True; detail["status"] = "verified"
            detail["judge_notes"] = "simplify(lhs - rhs) == 0"
            return detail
        detail["symbolic_equal"] = False
    except Exception as exc:  # noqa: BLE001
        detail["judge_notes"] = f"symbolic simplify error: {exc}"

    # 2) Numerical seeds
    seed_results: list[dict[str, Any]] = []
    any_disagree = False
    finite_seed_seen = False
    for _ in range(seeds):
        values = {str(sym): rng.uniform(0.1, 2.0) for sym in free_syms}
        subs = {sym: sympy.Float(values[str(sym)]) for sym in free_syms}
        try:
            lhs_val = complex(sympy.N(lhs_expr.subs(subs)))
            rhs_val = complex(sympy.N(rhs_expr.subs(subs)))
        except (ZeroDivisionError, ValueError, TypeError) as exc:
            seed_results.append({"values": values, "error": f"eval: {exc}", "match": None})
            continue
        except Exception as exc:  # noqa: BLE001
            seed_results.append({"values": values, "error": f"eval-unexpected: {exc}", "match": None})
            continue
        if not (_finite(lhs_val) and _finite(rhs_val)):
            seed_results.append({
                "values": values, "lhs_val": _repr_complex(lhs_val),
                "rhs_val": _repr_complex(rhs_val),
                "match": None, "note": "non-finite or complex result",
            })
            continue
        finite_seed_seen = True
        lhs_r, rhs_r = lhs_val.real, rhs_val.real
        rel = tolerance * max(1.0, abs(lhs_r))
        match = abs(lhs_r - rhs_r) < rel
        seed_results.append({
            "values": values, "lhs_val": lhs_r, "rhs_val": rhs_r,
            "abs_diff": abs(lhs_r - rhs_r), "tolerance": rel, "match": match,
        })
        if not match:
            any_disagree = True
    detail["seeds_evaluated"] = seeds
    detail["seed_results"] = seed_results

    if not finite_seed_seen:
        detail["status"] = "unverifiable"
        detail["judge_notes"] = _join_notes(
            detail["judge_notes"],
            "all numerical seeds produced NaN/inf/complex values; cannot decide",
        )
        return detail
    if any_disagree:
        detail["status"] = "failed"
        detail["judge_notes"] = _join_notes(
            detail["judge_notes"],
            "at least one numerical seed disagreed beyond tolerance",
        )
        # Sign-flip heuristic
        try:
            diff_plus = sympy.simplify(lhs_expr + rhs_expr)
            if diff_plus == 0 or diff_plus.equals(0):
                detail["sign_flip_detected"] = True
                detail["judge_notes"] += (
                    " | simplify(lhs + rhs) == 0 → equation appears to be "
                    "sign-flipped (sign error)"
                )
        except Exception:  # noqa: BLE001
            pass
        return detail

    agreeing = sum(1 for r in seed_results if r.get("match") is True)
    detail["status"] = "verified"
    detail["judge_notes"] = _join_notes(
        detail["judge_notes"],
        f"all {agreeing} numerical seeds agreed within tolerance",
    )
    return detail


def _finite(z: complex) -> bool:
    return math.isfinite(z.real) and math.isfinite(z.imag) and abs(z.imag) < 1e-9


def _repr_complex(z: complex) -> str:
    try:
        return f"{z.real:.6g}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.3g}j"
    except Exception:  # noqa: BLE001
        return str(z)


def _join_notes(existing: str, note: str) -> str:
    return (existing + " | " if existing else "") + note


# ----------------------------- Target building -----------------------------


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_target(
    index: int, eq_item: dict[str, Any], pair_idx: int, detail: dict[str, Any],
) -> dict[str, Any]:
    severity = {"verified": "minor", "failed": "P0", "unverifiable": "P0"}.get(detail["status"], "major")
    locator = f"eq:{index}" if pair_idx == 0 else f"eq:{index}.{pair_idx}"
    sign_note = " (sign-flip heuristic matched)" if detail.get("sign_flip_detected") else ""
    evidence = {
        "quote": _collapse_ws(eq_item["raw"])[:MAX_EQ_QUOTE_CHARS],
        "lhs": detail.get("lhs"), "rhs": detail.get("rhs"),
        "lhs_tex": detail.get("lhs_tex"), "rhs_tex": detail.get("rhs_tex"),
        "free_symbols": detail.get("free_symbols", []),
        "symbolic_equal": detail.get("symbolic_equal"),
        "seeds_evaluated": detail.get("seeds_evaluated", 0),
        "seed_results": detail.get("seed_results", []),
        "sign_flip_detected": detail.get("sign_flip_detected", False),
        "judge_notes": (detail.get("judge_notes") or "") + sign_note,
        "judge_confidence": "high" if detail["status"] == "failed" else "medium",
        "finding_kind": "equation_algebra",
    }
    return {
        "locator": locator, "status": detail["status"],
        "severity_suggestion": severity, "evidence": evidence,
        "root_cause_key": f"math-sympy-{detail['status']}-{index}-{pair_idx}",
    }


def _skip_target(
    idx: int, pair_idx: int, item: dict[str, Any], reason: str,
    lhs_tex: str | None = None, rhs_tex: str | None = None,
) -> dict[str, Any]:
    locator = f"eq:{idx}" if pair_idx == 0 else f"eq:{idx}.{pair_idx}"
    evidence: dict[str, Any] = {
        "quote": _collapse_ws(item["raw"])[:MAX_EQ_QUOTE_CHARS],
        "judge_notes": reason, "judge_confidence": "medium",
        "finding_kind": "unparseable",
    }
    if lhs_tex is not None:
        evidence["lhs_tex"] = lhs_tex[:MAX_EQ_QUOTE_CHARS]
    if rhs_tex is not None:
        evidence["rhs_tex"] = rhs_tex[:MAX_EQ_QUOTE_CHARS]
    return {
        "locator": locator, "status": "unverifiable",
        "severity_suggestion": "P0", "evidence": evidence,
        "root_cause_key": f"math-sympy-unparseable-{idx}-{pair_idx}",
    }


# ----------------------------- Orchestration -----------------------------


def run(
    *,
    tex: str,
    out_path: Path,
    seeds: int,
    tolerance: float,
    per_equation_timeout_seconds: int = 5,
    max_equations: int = 500,
) -> dict[str, Any]:
    ok, err = _sympy_available()
    if not ok:
        report = _build_report(
            status="unverifiable", severity="P0",
            summary=f"sympy unavailable ({err}); cannot verify equation algebra",
            targets=[], tex_len=len(tex), seeds=seeds, tolerance=tolerance,
            n_items=0, sympy_ok=False, errors=[err or "sympy import failed"],
        )
        _write(out_path, report)
        return report

    items = extract_equations(tex)
    rng = random.Random(RNG_SEED)
    targets: list[dict[str, Any]] = []
    errors: list[str] = []
    counts = {"verified": 0, "failed": 0, "unverifiable": 0}

    if len(items) > max_equations:
        errors.append(
            f"truncated: paper has {len(items)} equation(s); "
            f"verifying only the first {max_equations}"
        )
        items = items[:max_equations]

    for idx, item in enumerate(items):
        body = item["body"]
        if is_definition(body):
            targets.append(_skip_target(
                idx, 0, item,
                reason="equation is a definition (:=, \\equiv, \\coloneqq); skipping",
            ))
            counts["unverifiable"] += 1
            continue
        body_hits = unparseable_hits(body)
        pairs = split_equalities(body)
        if not pairs:
            continue
        for pair_idx, (lhs, rhs) in enumerate(pairs):
            hits = unparseable_hits(lhs) + unparseable_hits(rhs) + body_hits
            if hits:
                targets.append(_skip_target(
                    idx, pair_idx, item,
                    reason="sympy cannot parse operator(s): " + ", ".join(sorted(set(hits))),
                    lhs_tex=lhs, rhs_tex=rhs,
                ))
                counts["unverifiable"] += 1
                continue
            try:
                with _equation_timeout(per_equation_timeout_seconds):
                    detail = verify_pair(
                        lhs, rhs, seeds=seeds, tolerance=tolerance, rng=rng,
                    )
            except _SympyTimeout as exc:
                errors.append(f"eq {idx}.{pair_idx}: {exc}")
                locator = f"eq:{idx}" if pair_idx == 0 else f"eq:{idx}.{pair_idx}"
                targets.append({
                    "locator": locator,
                    "status": "unverifiable",
                    "severity_suggestion": "P0",
                    "evidence": {
                        "quote": _collapse_ws(item["raw"])[:MAX_EQ_QUOTE_CHARS],
                        "lhs_tex": lhs[:MAX_EQ_QUOTE_CHARS],
                        "rhs_tex": rhs[:MAX_EQ_QUOTE_CHARS],
                        "judge_notes": (
                            f"sympy per-equation timeout after "
                            f"{per_equation_timeout_seconds}s; cannot decide"
                        ),
                        "judge_confidence": "medium",
                        "finding_kind": "sympy_timeout",
                    },
                    "root_cause_key": f"math-sympy-timeout-{idx}-{pair_idx}",
                })
                counts["unverifiable"] += 1
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"eq {idx}.{pair_idx}: {exc}")
                detail = {
                    "lhs_tex": lhs[:MAX_EQ_QUOTE_CHARS], "rhs_tex": rhs[:MAX_EQ_QUOTE_CHARS],
                    "status": "unverifiable", "judge_notes": f"verifier exception: {exc}",
                }
            target = build_target(idx, item, pair_idx, detail)
            targets.append(target)
            counts[target["status"]] = counts.get(target["status"], 0) + 1

    if not targets:
        targets.append({
            "locator": "eq:none", "status": "verified",
            "severity_suggestion": "minor",
            "evidence": {
                "quote": "", "finding_kind": "no_equations",
                "judge_notes": "No displayed equalities were found. Nothing to verify.",
                "judge_confidence": "medium",
            },
            "root_cause_key": "math-sympy-no-equations",
        })

    if counts["failed"]:
        status, severity = "failed", "P0"
        summary = (
            f"{counts['failed']} of {len(targets)} equation pair(s) failed algebraic check; "
            f"{counts['unverifiable']} unverifiable; {counts['verified']} verified"
        )
    elif counts["unverifiable"]:
        status, severity = "unverifiable", "P0"
        summary = (
            f"{counts['unverifiable']} of {len(targets)} equation pair(s) unverifiable; "
            f"{counts['verified']} verified"
        )
    else:
        status, severity = "verified", "minor"
        summary = f"all {counts['verified']} checkable equation pair(s) agreed symbolically/numerically"

    report = _build_report(
        status=status, severity=severity, summary=summary, targets=targets,
        tex_len=len(tex), seeds=seeds, tolerance=tolerance,
        n_items=len(items), sympy_ok=True, errors=errors,
    )
    _write(out_path, report)
    return report


def _build_report(
    *, status: str, severity: str, summary: str, targets: list[dict[str, Any]],
    tex_len: int, seeds: int, tolerance: float, n_items: int, sympy_ok: bool,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "verifier_id": VERIFIER_ID, "verifier_version": VERIFIER_VERSION,
        "status": status, "severity_suggestion": severity, "summary": summary,
        "targets": targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0, "model": None, "cached": False,
            "inputs": {
                "tex_chars": tex_len, "seeds": seeds, "tolerance": tolerance,
                "n_equations_extracted": n_items, "n_targets": len(targets),
                "sympy_available": sympy_ok,
            },
        },
        "errors": errors,
    }


def _write(out_path: Path, report: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------- CLI -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify displayed-equation algebra via sympy symbolic + numerical checks.",
    )
    parser.add_argument("--tex", required=True, type=Path, help="LaTeX file to parse")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random numerical substitutions when symbolic check is inconclusive",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-8,
        help="Relative tolerance: |lhs-rhs| < tol*max(1,|lhs|)",
    )
    parser.add_argument(
        "--equation-timeout", type=int, default=5, dest="equation_timeout",
        help="Per-equation sympy operation timeout in seconds (POSIX only; 0 disables)",
    )
    parser.add_argument(
        "--max-equations", type=int, default=500, dest="max_equations",
        help="Truncate verification after this many extracted equations",
    )
    args = parser.parse_args()

    if not args.tex.is_file():
        print(f"error: tex file not found: {args.tex}", file=sys.stderr)
        return 2
    tex = args.tex.read_text(encoding="utf-8", errors="replace")
    report = run(
        tex=tex,
        out_path=args.out,
        seeds=args.seeds,
        tolerance=args.tolerance,
        per_equation_timeout_seconds=args.equation_timeout,
        max_equations=args.max_equations,
    )
    print(f"[{VERIFIER_ID}] {report['status'].upper()}: {report['summary']}")
    for t in report["targets"][:20]:
        print(f"  - {t['locator']}: {t['status']} (suggested={t['severity_suggestion']})")
    if len(report["targets"]) > 20:
        print(f"  ... and {len(report['targets']) - 20} more")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
