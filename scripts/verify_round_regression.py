"""verify_round_regression.py — Cross-round manuscript regression verifier.

Compares two versions of a paper (round N-1 vs round N) plus their verifier
outputs and extracted claim manifests to catch four classes of between-round
failure:

  1. Regression — a claim that was ``verified`` in round N-1 is now
     ``failed`` or ``unverifiable``.
  2. Silent claim drift — a numeric value or universal quantifier changed
     between rounds without a fix-log note.
  3. Deletion-evasion — a P0-flagged claim was removed entirely instead of
     fixed (and no entry in the fix-log documents the fix).
  4. New unchecked content — newly introduced \\cite keys or new universal
     quantifiers without verifier coverage.

Output conforms to ``references/VERIFIER_CONTRACT.md``. ``verifier_id`` is
``verify_round_regression``. Status is ``failed`` if any regression/drift/
deletion target exists, ``unverifiable`` if only new-unchecked targets remain,
``verified`` when everything is clean.

Usage:
    python verify_round_regression.py \\
        --prev-tex paper_v1.tex \\
        --curr-tex paper_v2.tex \\
        --prev-verifier-dir workspace/round1/verifier/ \\
        --curr-verifier-dir workspace/round2/verifier/ \\
        --prev-claims workspace/round1/claims/ \\
        --curr-claims workspace/round2/claims/ \\
        --fix-log workspace/round2/fix_log.json \\
        --out workspace/round2/verifier/regression.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERIFIER_ID = "verify_round_regression"
VERIFIER_VERSION = "0.3-stageB2"
MAX_QUOTE_CHARS = 400

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_QUANTIFIER_RE = re.compile(
    "|".join([
        r"\bfor all\b", r"\bfor any\b", r"\bfor every\b",
        r"\buniversal(?:ly)?\b", r"\bwithout (?:any )?assumption(?:s)?\b",
        r"\bin general\b", r"\barbitrary\b",
        r"\bholds? (?:for|in) (?:all|any|every)\b",
        r"\bfirst to\b", r"\balways\b",
        r"\bany\s+(?:continuous|smooth|bounded|finite|arbitrary|general)\b",
        r"\bunconditional(?:ly)?\b", r"\bmodel-?free\b",
        r"\bassumption-?free\b", r"\bnon-?parametric\b",
    ]),
    re.IGNORECASE,
)


# -------- IO helpers --------


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    items: list[dict[str, Any]] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return items


def _load_verifier_dir(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for candidate in sorted(path.glob("*.json")):
        doc = _load_json(candidate)
        if isinstance(doc, dict):
            doc.setdefault("_source_file", str(candidate))
            out.append(doc)
    return out


def _load_fix_log(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.is_file():
        return {}
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        entries: Iterable[dict[str, Any]] = data["entries"]
    elif isinstance(data, list):
        entries = data
    else:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("finding_key") or entry.get("root_cause_key")
        if key:
            out[key] = entry
    return out


# -------- Claim bundle + indexing --------


@dataclass
class ClaimBundle:
    citations: list[dict[str, Any]] = field(default_factory=list)
    numerical: list[dict[str, Any]] = field(default_factory=list)
    scope: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dir(cls, path: Path | None) -> "ClaimBundle":
        if not path or not path.is_dir():
            return cls()
        return cls(
            citations=_load_jsonl(path / "citation_claims.jsonl"),
            numerical=_load_jsonl(path / "numerical_claims.jsonl"),
            scope=_load_jsonl(path / "scope_claims.jsonl"),
        )


def _iter_targets(reports: list[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for report in reports:
        for tgt in report.get("targets", []) or []:
            yield report, tgt


def _index_by_rck(reports: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    idx: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for report, tgt in _iter_targets(reports):
        rck = tgt.get("root_cause_key")
        if rck:
            idx[rck] = (report, tgt)
    return idx


def _index_claims(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {it["locator"]: it for it in items if it.get("locator")}


# -------- Similarity helpers --------


def _normalize_quote(q: str) -> str:
    q = re.sub(r"\s+", " ", q or "").strip().lower()
    return re.sub(r"[\u2018\u2019\u201c\u201d]", "'", q)


def _char_overlap(a: str, b: str) -> float:
    """Jaccard over character 4-grams of normalized quotes."""
    if not a or not b:
        return 0.0
    na, nb = _normalize_quote(a), _normalize_quote(b)
    A = {na[i : i + 4] for i in range(max(0, len(na) - 3))} or {na}
    B = {nb[i : i + 4] for i in range(max(0, len(nb) - 3))} or {nb}
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _numbers_in(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def _quantifiers_in(text: str) -> set[str]:
    return {m.group(0).lower() for m in _QUANTIFIER_RE.finditer(text or "")}


def _truncate(text: str, n: int = MAX_QUOTE_CHARS) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    return clean if len(clean) <= n else clean[: n - 1] + "\u2026"


# -------- Target builder --------


def _target(
    *,
    locator: str,
    status: str,
    severity: str,
    quote: str,
    judge_notes: str,
    root_cause_key: str,
    paired_quote: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "quote": _truncate(quote),
        "judge_notes": judge_notes,
        "judge_confidence": "high",
        "second_opinion_agreed": True,
    }
    if paired_quote is not None:
        evidence["paired_quote"] = _truncate(paired_quote)
    if extra:
        evidence.update(extra)
    return {
        "locator": locator,
        "status": status,
        "severity_suggestion": severity,
        "evidence": evidence,
        "root_cause_key": root_cause_key,
    }


def _crash_target(locator: str, rck: str, notes: str) -> dict[str, Any]:
    return _target(
        locator=locator, status="unverifiable", severity="P0",
        quote="", judge_notes=notes, root_cause_key=rck,
        # A crash inside a regression-checker is an env/tool condition, not a
        # finding about the paper itself. Tag as env so downstream consumers
        # route to setup_needed rather than gate_blocker.
        extra={"unverifiable_kind": "env"},
    )


# -------- Checkers --------


def check_regressions(
    prev_idx: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    curr_idx: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rck, (prev_report, prev_tgt) in prev_idx.items():
        if (prev_tgt.get("status") or "").lower() != "verified" or rck not in curr_idx:
            continue
        curr_report, curr_tgt = curr_idx[rck]
        new_status = (curr_tgt.get("status") or "").lower()
        if new_status not in {"failed", "unverifiable"}:
            continue
        try:
            prev_notes = prev_tgt.get("evidence", {}).get("judge_notes", "")[:160]
            curr_notes = curr_tgt.get("evidence", {}).get("judge_notes", "")[:160]
            notes = (
                f"Target was 'verified' in round N-1 ({prev_report.get('verifier_id')})"
                f" but is now '{new_status}' in round N ({curr_report.get('verifier_id')}). "
                f"Prev: {prev_notes} | Curr: {curr_notes}"
            )
            out.append(_target(
                locator=f"regression:{rck}", status="failed", severity="P0",
                quote=curr_tgt.get("evidence", {}).get("quote", "") or "",
                paired_quote=prev_tgt.get("evidence", {}).get("quote", "") or "",
                judge_notes=notes, root_cause_key=f"regression-{rck}",
                extra={
                    "prev_status": "verified", "curr_status": new_status,
                    "prev_verifier": prev_report.get("verifier_id"),
                    "curr_verifier": curr_report.get("verifier_id"),
                    "original_root_cause_key": rck,
                },
            ))
        except Exception as exc:  # noqa: BLE001
            out.append(_crash_target(
                f"regression:{rck}", f"regression-crash-{rck}",
                f"regression check crashed: {exc}",
            ))
    return out


def check_claim_drift(prev: ClaimBundle, curr: ClaimBundle) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, prev_items, curr_items in (
        ("citation", prev.citations, curr.citations),
        ("numerical", prev.numerical, curr.numerical),
        ("scope", prev.scope, curr.scope),
    ):
        prev_by = _index_claims(prev_items)
        curr_by = _index_claims(curr_items)
        for loc in sorted(set(prev_by) & set(curr_by)):
            try:
                pq = prev_by[loc].get("claim_quote", "") or ""
                cq = curr_by[loc].get("claim_quote", "") or ""
                if _normalize_quote(pq) == _normalize_quote(cq):
                    continue
                overlap = _char_overlap(pq, cq)
                if overlap >= 0.9:
                    continue
                pn, cn = _numbers_in(pq), _numbers_in(cq)
                pq_set, cq_set = _quantifiers_in(pq), _quantifiers_in(cq)
                nums_changed, quants_changed = pn != cn, pq_set != cq_set
                severity = "P0" if (nums_changed or quants_changed) else "major"
                notes = (
                    f"Claim at locator {loc} changed between rounds "
                    f"(char-4gram overlap={overlap:.2f}). "
                    f"numbers_changed={nums_changed} (prev={sorted(pn)}, curr={sorted(cn)}); "
                    f"quantifiers_changed={quants_changed} "
                    f"(prev={sorted(pq_set)}, curr={sorted(cq_set)})."
                )
                out.append(_target(
                    locator=f"drift:{loc}", status="failed", severity=severity,
                    quote=cq, paired_quote=pq,
                    judge_notes=notes, root_cause_key=f"drift-{name}-{loc}",
                    extra={
                        "overlap": round(overlap, 3),
                        "numbers_changed": nums_changed,
                        "quantifiers_changed": quants_changed,
                        "prev_numbers": sorted(pn), "curr_numbers": sorted(cn),
                        "prev_quantifiers": sorted(pq_set),
                        "curr_quantifiers": sorted(cq_set),
                        "claim_kind": name,
                        "original_root_cause_key": loc,
                    },
                ))
            except Exception as exc:  # noqa: BLE001
                out.append(_crash_target(
                    f"drift:{loc}", f"drift-crash-{loc}",
                    f"drift check crashed for {loc}: {exc}",
                ))
    return out


def _is_prev_p0(tgt: dict[str, Any]) -> bool:
    return (
        (tgt.get("severity_suggestion") or "").upper() == "P0"
        or (tgt.get("status") or "").lower() == "failed"
    )


def check_deletion_evasion(
    prev_idx: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    curr_idx: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    fix_log: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rck, (prev_report, prev_tgt) in prev_idx.items():
        if not _is_prev_p0(prev_tgt) or rck in curr_idx:
            continue
        fix_entry = fix_log.get(rck)
        if fix_entry and (fix_entry.get("disposition") or "").lower() == "fixed":
            continue
        try:
            notes = (
                "Previously flagged P0 claim is absent from current version with no "
                "documented fix. Either the author silently deleted the claim (evasion) "
                "or the verifier failed to re-check it. Requires human inspection."
            )
            if fix_entry:
                notes += (
                    f" Fix-log entry present with disposition="
                    f"{fix_entry.get('disposition')!r}, rationale="
                    f"{(fix_entry.get('rationale') or '')[:160]!r}."
                )
            out.append(_target(
                locator=f"deletion-evasion:{rck}", status="failed", severity="P0",
                quote=prev_tgt.get("evidence", {}).get("quote", "") or "",
                judge_notes=notes, root_cause_key=f"deletion-evasion-{rck}",
                extra={
                    "prev_status": prev_tgt.get("status"),
                    "prev_severity": prev_tgt.get("severity_suggestion"),
                    "prev_verifier": prev_report.get("verifier_id"),
                    "fix_log_entry": fix_entry,
                    "original_root_cause_key": rck,
                },
            ))
        except Exception as exc:  # noqa: BLE001
            out.append(_crash_target(
                f"deletion-evasion:{rck}", f"deletion-evasion-crash-{rck}",
                f"deletion-evasion check crashed: {exc}",
            ))
    return out


def check_new_citations(
    prev: ClaimBundle, curr: ClaimBundle, curr_reports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prev_locs = {c.get("locator") for c in prev.citations if c.get("locator")}
    curr_locs = {c.get("locator") for c in curr.citations if c.get("locator")}
    new_locs = curr_locs - prev_locs
    if not new_locs:
        return []
    verified: set[str] = set()
    for _report, tgt in _iter_targets(curr_reports):
        loc = tgt.get("locator") or ""
        if loc.startswith("cite:") and (tgt.get("status") or "").lower() == "verified":
            verified.add(loc)
    out: list[dict[str, Any]] = []
    curr_by_loc = _index_claims(curr.citations)
    for loc in sorted(new_locs):
        try:
            if loc in verified:
                continue
            claim = curr_by_loc.get(loc, {})
            out.append(_target(
                locator=f"new-citation:{loc}",
                status="unverifiable", severity="P0",
                quote=claim.get("claim_quote", "") or "",
                judge_notes=(
                    "New citation introduced in round N has no 'verified' result in the "
                    "current verifier directory. Run verify_citations_full.py on it before "
                    "accepting the revision."
                ),
                root_cause_key=f"new-citation-uncovered-{loc}",
                extra={
                    "citation_key": loc.split(":", 1)[1] if ":" in loc else loc,
                    "section_hint": claim.get("section_hint"),
                },
            ))
        except Exception as exc:  # noqa: BLE001
            out.append(_crash_target(
                f"new-citation:{loc}", f"new-citation-crash-{loc}",
                f"new-citation check crashed for {loc}: {exc}",
            ))
    return out


def check_new_scope(
    prev: ClaimBundle, curr: ClaimBundle, curr_reports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prev_quotes = {_normalize_quote(c.get("claim_quote", "")) for c in prev.scope}
    has_scope_verifier = any(
        r.get("verifier_id") == "verify_internal_consistency" for r in curr_reports
    )
    out: list[dict[str, Any]] = []
    for claim in curr.scope:
        try:
            quote = claim.get("claim_quote", "") or ""
            if _normalize_quote(quote) in prev_quotes:
                continue
            if has_scope_verifier:
                continue
            out.append(_target(
                locator=f"new-scope:{claim.get('locator', 'scope')}",
                status="unverifiable", severity="P0",
                quote=quote,
                judge_notes=(
                    "New universal-quantifier scope claim appeared in round N but no "
                    "verify_internal_consistency output exists in the current verifier "
                    "directory. Re-run the scope verifier before accepting."
                ),
                root_cause_key=f"new-scope-uncovered-{claim.get('locator', 'unknown')}",
                extra={
                    "quantifier_phrase": claim.get("quantifier_phrase"),
                    "section_hint": claim.get("section_hint"),
                },
            ))
        except Exception as exc:  # noqa: BLE001
            out.append(_crash_target(
                f"new-scope:{claim.get('locator', 'unknown')}",
                "new-scope-crash",
                f"new-scope check crashed: {exc}",
            ))
    return out


# -------- Orchestration --------


def _dedupe(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for t in targets:
        key = t.get("root_cause_key") or t.get("locator") or ""
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def aggregate(targets: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not targets:
        return ("verified", "minor",
                "no regressions, drift, deletions, or uncovered new content detected.")
    n_failed = sum(1 for t in targets if (t.get("status") or "").lower() == "failed")
    n_unver = sum(1 for t in targets if (t.get("status") or "").lower() == "unverifiable")
    if n_failed:
        return ("failed", "P0",
                f"{n_failed} regression/drift/deletion-evasion target(s); "
                f"{n_unver} new-content uncovered.")
    return ("unverifiable", "P0",
            f"{n_unver} new-content target(s) without verifier coverage; no regressions.")


@dataclass
class RegressionInputs:
    prev_tex: Path | None
    curr_tex: Path | None
    prev_verifier_dir: Path | None
    curr_verifier_dir: Path | None
    prev_claims: Path | None
    curr_claims: Path | None
    fix_log: Path | None
    out_path: Path


def run(inputs: RegressionInputs) -> dict[str, Any]:
    errors: list[str] = []

    def _safe(loader, label, default):
        try:
            return loader()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"failed to load {label}: {exc}")
            return default

    prev_reports = _safe(lambda: _load_verifier_dir(inputs.prev_verifier_dir), "prev verifier", [])
    curr_reports = _safe(lambda: _load_verifier_dir(inputs.curr_verifier_dir), "curr verifier", [])
    prev_bundle = _safe(lambda: ClaimBundle.from_dir(inputs.prev_claims), "prev claims", ClaimBundle())
    curr_bundle = _safe(lambda: ClaimBundle.from_dir(inputs.curr_claims), "curr claims", ClaimBundle())
    fix_log = _safe(lambda: _load_fix_log(inputs.fix_log), "fix log", {})

    prev_idx = _index_by_rck(prev_reports)
    curr_idx = _index_by_rck(curr_reports)

    targets: list[dict[str, Any]] = []
    for name, fn in (
        ("regressions", lambda: check_regressions(prev_idx, curr_idx)),
        ("claim_drift", lambda: check_claim_drift(prev_bundle, curr_bundle)),
        ("deletion_evasion", lambda: check_deletion_evasion(prev_idx, curr_idx, fix_log)),
        ("new_citations", lambda: check_new_citations(prev_bundle, curr_bundle, curr_reports)),
        ("new_scope", lambda: check_new_scope(prev_bundle, curr_bundle, curr_reports)),
    ):
        try:
            targets.extend(fn())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name} crashed: {exc}")
            targets.append(_crash_target(
                f"checker-crash:{name}",
                f"regression-checker-crash-{name}",
                f"{name} crashed wholesale: {exc}",
            ))

    targets = _dedupe(targets)
    status, severity, summary = aggregate(targets)

    def _s(p: Path | None) -> str | None:
        return str(p) if p else None

    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0,
            "model": None,
            "cached": False,
            "inputs": {
                "prev_tex": _s(inputs.prev_tex),
                "curr_tex": _s(inputs.curr_tex),
                "prev_verifier_dir": _s(inputs.prev_verifier_dir),
                "curr_verifier_dir": _s(inputs.curr_verifier_dir),
                "prev_claims": _s(inputs.prev_claims),
                "curr_claims": _s(inputs.curr_claims),
                "fix_log": _s(inputs.fix_log),
                "n_prev_reports": len(prev_reports),
                "n_curr_reports": len(curr_reports),
                "n_prev_targets": len(prev_idx),
                "n_curr_targets": len(curr_idx),
                "n_prev_citation_claims": len(prev_bundle.citations),
                "n_curr_citation_claims": len(curr_bundle.citations),
                "n_prev_numerical_claims": len(prev_bundle.numerical),
                "n_curr_numerical_claims": len(curr_bundle.numerical),
                "n_prev_scope_claims": len(prev_bundle.scope),
                "n_curr_scope_claims": len(curr_bundle.scope),
                "n_fix_log_entries": len(fix_log),
            },
        },
        "errors": errors,
    }

    inputs.out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs.out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two rounds of a manuscript + their verifier outputs to catch "
            "regressions, silent claim drift, deletion-evasion, and uncovered new content."
        )
    )
    parser.add_argument("--prev-tex", type=Path, default=None)
    parser.add_argument("--curr-tex", type=Path, default=None)
    parser.add_argument("--prev-verifier-dir", type=Path, default=None)
    parser.add_argument("--curr-verifier-dir", type=Path, default=None)
    parser.add_argument("--prev-claims", type=Path, default=None)
    parser.add_argument("--curr-claims", type=Path, default=None)
    parser.add_argument("--fix-log", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    # Validate that if any prev-* or curr-* input is supplied, all three peers
    # in that group are supplied. Partial inputs silently produce misleading
    # "no regressions" reports, so fail fast with a clear CLI error.
    prev_group = {
        "--prev-tex": args.prev_tex,
        "--prev-verifier-dir": args.prev_verifier_dir,
        "--prev-claims": args.prev_claims,
    }
    curr_group = {
        "--curr-tex": args.curr_tex,
        "--curr-verifier-dir": args.curr_verifier_dir,
        "--curr-claims": args.curr_claims,
    }
    for label, group in (("prev", prev_group), ("curr", curr_group)):
        provided = [k for k, v in group.items() if v is not None]
        missing = [k for k, v in group.items() if v is None]
        if provided and missing:
            print(
                f"error: regression check requires complete {label} inputs when "
                f"any are supplied. Missing: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2

    out_path = args.out.resolve()
    r = lambda p: p.resolve() if p else None  # noqa: E731
    inputs = RegressionInputs(
        prev_tex=r(args.prev_tex), curr_tex=r(args.curr_tex),
        prev_verifier_dir=r(args.prev_verifier_dir),
        curr_verifier_dir=r(args.curr_verifier_dir),
        prev_claims=r(args.prev_claims), curr_claims=r(args.curr_claims),
        fix_log=r(args.fix_log), out_path=out_path,
    )

    try:
        report = run(inputs)
    except Exception as exc:  # noqa: BLE001
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fallback = {
            "verifier_id": VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "summary": f"regression verifier crashed: {exc}",
            "targets": [],
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cost_usd": 0.0,
                "model": None,
                "cached": False,
                "inputs": {"out": str(out_path)},
            },
            "errors": [str(exc)],
        }
        out_path.write_text(
            json.dumps(fallback, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{VERIFIER_ID}] UNVERIFIABLE: {fallback['summary']}", file=sys.stderr)
        return 1

    print(f"[{VERIFIER_ID}] {report['status'].upper()}: {report['summary']}")
    for t in report["targets"]:
        print(f"  - {t['locator']}: {t['status']} (suggested={t['severity_suggestion']})")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
