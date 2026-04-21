"""verify_internal_consistency.py — abstract ↔ methods ↔ results scope verifier.

Detects the failure mode where a paper's abstract or introduction commits to a
universal / general claim that the methods section only partially supports. For
example: abstract says "we prove for arbitrary continuous dynamics" while
methods prove only for a block-diagonal Gaussian Ornstein–Uhlenbeck process.

Two complementary checks:
  1. Universality asymmetry — abstract uses strong universal quantifiers that
     the methods section does not match.
  2. Restriction leakage — methods section contains restriction keywords
     (e.g. "block-diagonal", "linear regime", "small noise") that never appear
     in the abstract.

Output conforms to `references/VERIFIER_CONTRACT.md`.

Usage:
    python verify_internal_consistency.py --tex paper.tex --out workspace/verifier/scope.json
    python verify_internal_consistency.py --sections sections.json --out ...

`sections.json` lets callers pre-split if they already have section text:
    {"abstract": "...", "introduction": "...", "methods": "...", "results": "...", "conclusion": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERIFIER_ID = "verify_internal_consistency"
VERIFIER_VERSION = "0.3-stageB2"

SECTION_KEYS = ["abstract", "introduction", "methods", "results", "conclusion"]

UNIVERSAL_MARKERS = [
    r"\buniversal(?:ly)?\b",
    r"\bin general\b",
    r"\bfully general\b",
    r"\bfor (all|any|every|arbitrary)\b",
    r"\barbitrary\b",
    r"\bunconditional(?:ly)?\b",
    r"\bregardless of\b",
    r"\bwithout (any )?assumption(s)?\b",
    r"\bholds? (for|in) all\b",
    r"\bfirst (principle|general|universal)\b",
    r"\bnon-perturbative\b",
    r"\bmodel-free\b",
    r"\bassumption-free\b",
    r"\bnonparametric\b",
    r"\bany (continuous|smooth|bounded|finite) (system|dynamics|process|distribution|function)\b",
]

RESTRICTION_MARKERS = [
    # Distributional restrictions
    r"\bGaussian\b",
    r"\blog-normal\b",
    r"\bexponential family\b",
    r"\bsub-?Gaussian\b",
    # Structural
    r"\bblock-?diagonal\b",
    r"\bdiagonal\b",
    r"\bsymmetric\b",
    r"\bcircular\b",
    r"\bsparse\b",
    # Dynamics
    r"\blinear(?:ized| regime)?\b",
    r"\bquadratic\b",
    r"\bweakly (?:coupled|non-?linear)\b",
    r"\bOrnstein-?Uhlenbeck\b",
    r"\bLangevin\b",
    r"\bMarkov(?:ian)?\b",
    # Regime
    r"\bsteady state\b",
    r"\bequilibrium\b",
    r"\bthermodynamic limit\b",
    r"\blow(?:-| )dimensional\b",
    r"\bhigh(?:-| )temperature\b",
    r"\bsmall noise\b",
    r"\bweak (?:coupling|interaction|noise)\b",
    r"\bdilute\b",
    r"\bmean-field\b",
    # Experimental / numerical restrictions
    r"\bsynthetic (data|benchmark)\b",
    r"\btoy (model|problem|example)\b",
    r"\b1D\b|\bone[- ]dimensional\b",
    r"\b2D\b|\btwo[- ]dimensional\b",
    # Claim-type restrictions
    r"\basymptotically\b",
    r"\bto leading order\b",
    r"\bunder the assumption\b",
    r"\bassuming that\b",
    r"\bprovided that\b",
    r"\bif (?:we|one) assume(s)?\b",
    r"\bwhen .{0,40} is (small|large|positive|bounded)\b",
]

QUANT_THRESHOLD_SCOPE_DIFF = 0.6  # if methods restrictions not covered in abstract by > 60%, flag


@dataclass
class SectionMarkers:
    universal: list[str]
    restrictions: list[str]
    raw_len: int


_HEADING_RE = re.compile(
    r"\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]+)\}"
)


def parse_latex(tex: str) -> dict[str, str]:
    """Extract the five canonical sections from a LaTeX source.

    Accepts \\chapter, \\section, \\subsection, \\subsubsection, \\paragraph
    headings — the last two are common in PRL-style physics letters where
    there is no top-level section structure.

    For `abstract`, prefers the `\\begin{abstract}...\\end{abstract}` env,
    then falls back to text between \\maketitle and the first heading.

    If no methods heading is found but text between abstract and first
    figure/results heading looks like methods, that span is used as methods
    (common in letter-format papers).
    """
    sections: dict[str, str] = {k: "" for k in SECTION_KEYS}

    # Abstract environment
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if abs_match:
        sections["abstract"] = _clean_tex(abs_match.group(1))

    heading_iter = list(_HEADING_RE.finditer(tex))
    for idx, m in enumerate(heading_iter):
        heading = m.group(1).strip().lower()
        start = m.end()
        end = heading_iter[idx + 1].start() if idx + 1 < len(heading_iter) else len(tex)
        body = tex[start:end]
        key = _heading_to_key(heading)
        if key and not sections[key]:
            sections[key] = _clean_tex(body)

    if not sections["abstract"]:
        if heading_iter:
            fallback = tex[: heading_iter[0].start()]
            fallback = re.sub(r".*?\\maketitle", "", fallback, flags=re.DOTALL)
            sections["abstract"] = _clean_tex(fallback)[:3000]

    # PRL-style fallback: if no methods-keyed heading was found, treat the text
    # between abstract (or maketitle) and the first results-keyed heading as
    # the methods body. Better to have an approximate methods than nothing.
    if not sections["methods"] and heading_iter:
        first_results_idx = next(
            (
                i for i, m in enumerate(heading_iter)
                if _heading_to_key(m.group(1).strip().lower()) in {"results", "conclusion"}
            ),
            None,
        )
        if first_results_idx is not None:
            # Start of span: just after abstract env (if any) or maketitle
            span_start = 0
            abs_env_end = re.search(r"\\end\{abstract\}", tex)
            maketitle = re.search(r"\\maketitle", tex)
            if abs_env_end:
                span_start = abs_env_end.end()
            elif maketitle:
                span_start = maketitle.end()
            span_end = heading_iter[first_results_idx].start()
            if span_end > span_start + 200:  # require non-trivial body
                body = tex[span_start:span_end]
                sections["methods"] = _clean_tex(body)
    return sections


def _heading_to_key(heading: str) -> str | None:
    norm = heading.lower()
    if re.search(r"\bintroduction\b|\bintro\b|\bbackground\b", norm):
        return "introduction"
    if re.search(r"\bmethod|approach|model|framework|theory|formulation|derivation", norm):
        return "methods"
    if re.search(r"\bresult|experiment|evaluation|analysis|finding", norm):
        return "results"
    if re.search(r"\bconclusion|discussion|summary|outlook", norm):
        return "conclusion"
    return None


_TEX_CLEAN_PATTERNS = [
    (re.compile(r"%[^\n]*"), ""),  # comments
    (re.compile(r"\\(cite|ref|eqref|label|citep|citet)\{[^}]*\}"), " "),
    (re.compile(r"\\(text(?:it|bf|tt|sf|rm|sc)|emph|mathrm|mathbf|mathit)\{([^}]*)\}"), r" \2 "),
    (re.compile(r"\\\\"), " "),
    (re.compile(r"[ \t]+"), " "),
]


def _clean_tex(body: str) -> str:
    cleaned = body
    for pattern, replacement in _TEX_CLEAN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip()


def scan_markers(text: str) -> SectionMarkers:
    if not text.strip():
        return SectionMarkers([], [], 0)
    lowered = text.lower()
    universal_hits: list[str] = []
    restriction_hits: list[str] = []
    for pat in UNIVERSAL_MARKERS:
        for m in re.finditer(pat, lowered):
            universal_hits.append(m.group(0))
    for pat in RESTRICTION_MARKERS:
        for m in re.finditer(pat, lowered):
            restriction_hits.append(m.group(0))
    return SectionMarkers(
        universal=sorted(set(universal_hits)),
        restrictions=sorted(set(restriction_hits)),
        raw_len=len(text),
    )


def compare_scope(
    abstract: SectionMarkers,
    methods: SectionMarkers,
    intro: SectionMarkers,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    # 1. Universality asymmetry: abstract / intro commits universally, methods restricts.
    upstream_universals = set(abstract.universal) | set(intro.universal)
    if upstream_universals and methods.restrictions:
        # Which restrictions are NEVER echoed in abstract? Those are the leakage.
        missing = [r for r in methods.restrictions if r not in set(abstract.restrictions)]
        leak_ratio = len(missing) / max(1, len(methods.restrictions))
        if leak_ratio >= QUANT_THRESHOLD_SCOPE_DIFF:
            findings.append(
                {
                    "kind": "universality_asymmetry",
                    "severity": "P0",
                    "abstract_universal_markers": sorted(upstream_universals),
                    "methods_restriction_markers": methods.restrictions,
                    "restrictions_not_in_abstract": missing,
                    "leak_ratio": round(leak_ratio, 3),
                    "description": (
                        "Abstract / introduction uses universal quantifiers but methods "
                        "introduces specific restrictions that the abstract does not disclose. "
                        "This is a scope overclaim."
                    ),
                }
            )

    return findings


BROAD_CLAIM_PATTERNS = [
    r"\bshow that\b",
    r"\bprove that\b",
    r"\bdemonstrate(?:s|d)?\b",
    r"\bestablish(?:es|ed)?\b",
    r"\bderive(?:s|d)?\b",
    r"\bobtain(?:s|ed)?\b",
    r"\bprovide(?:s|d)?\b",
]


def compare_scope_with_text(
    abstract_markers: SectionMarkers,
    methods_markers: SectionMarkers,
    intro_markers: SectionMarkers,
    abstract_text: str,
) -> list[dict[str, Any]]:
    """Extended scope comparison that also checks restriction-leakage when the
    abstract uses broad claim verbs (show/prove/derive/...) but no universal
    quantifier, which the first pass misses."""
    findings = compare_scope(abstract_markers, methods_markers, intro_markers)
    if findings:
        return findings

    abstract_lower = abstract_text.lower()
    broad_claim = any(re.search(p, abstract_lower) for p in BROAD_CLAIM_PATTERNS)
    if broad_claim and methods_markers.restrictions:
        missing = [
            r for r in methods_markers.restrictions
            if r not in set(abstract_markers.restrictions)
        ]
        if len(missing) >= 3:
            findings.append(
                {
                    "kind": "restriction_leakage",
                    "severity": "major",
                    "methods_restriction_markers": methods_markers.restrictions,
                    "restrictions_not_in_abstract": missing,
                    "description": (
                        "Methods section introduces multiple restrictions that the abstract "
                        "never mentions, while the abstract uses broad claim verbs. "
                        "Reader of the abstract will over-estimate scope."
                    ),
                }
            )

    if not methods_markers.raw_len:
        findings.append(
            {
                "kind": "methods_missing",
                "severity": "P0",
                "description": (
                    "Methods section could not be extracted; cannot verify abstract scope."
                ),
            }
        )

    return findings


def build_target(
    pair_name: str,
    finding: dict[str, Any],
    abstract_text: str,
    methods_text: str,
) -> dict[str, Any]:
    severity = finding.get("severity", "major")
    quote_a = _truncate(abstract_text, 280)
    quote_m = _truncate(methods_text, 280)

    description = finding.get("description", "")
    judge_notes_lines = [description]
    if "abstract_universal_markers" in finding:
        judge_notes_lines.append(
            f"Abstract universal markers: {finding['abstract_universal_markers']}"
        )
    if "methods_restriction_markers" in finding:
        judge_notes_lines.append(
            f"Methods restrictions: {finding['methods_restriction_markers']}"
        )
    if "restrictions_not_in_abstract" in finding:
        judge_notes_lines.append(
            f"Restrictions MISSING from abstract: {finding['restrictions_not_in_abstract']}"
        )
    if "leak_ratio" in finding:
        judge_notes_lines.append(f"Leak ratio = {finding['leak_ratio']}")

    status = "verified" if finding["kind"] != "methods_missing" else "unverifiable"
    evidence: dict[str, Any] = {
        "quote": quote_a,
        "paired_quote": quote_m,
        "judge_notes": "\n".join(judge_notes_lines),
        "judge_confidence": "high" if finding["kind"] == "universality_asymmetry" else "medium",
        "finding_kind": finding["kind"],
        "detail": finding,
    }
    if status == "unverifiable":
        # Methods section unparseable is a parser-coverage gap (tool), not a
        # paper defect, but it does leave the paper unverifiable until the
        # human confirms the methods exist.
        evidence["unverifiable_kind"] = "tool"
    return {
        "locator": f"pair:{pair_name}",
        "status": status,
        "severity_suggestion": severity,
        "evidence": evidence,
        "root_cause_key": f"internal-consistency-{finding['kind']}",
    }


def _truncate(text: str, n: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= n else clean[: n - 1] + "…"


def run(
    *,
    sections: dict[str, str],
    out_path: Path,
) -> dict[str, Any]:
    abstract_markers = scan_markers(sections.get("abstract", ""))
    intro_markers = scan_markers(sections.get("introduction", ""))
    methods_markers = scan_markers(sections.get("methods", ""))

    findings = compare_scope_with_text(
        abstract_markers,
        methods_markers,
        intro_markers,
        sections.get("abstract", ""),
    )

    # M1 defence-in-depth: if methods is empty, guarantee a methods_missing
    # finding is present even if other compare_scope paths short-circuited.
    if not methods_markers.raw_len and not any(
        f.get("kind") == "methods_missing" for f in findings
    ):
        findings.append({
            "kind": "methods_missing",
            "severity": "P0",
            "description": (
                "Methods section could not be extracted; cannot verify abstract scope."
            ),
        })

    targets = [
        build_target(
            "abstract-methods",
            f,
            sections.get("abstract", ""),
            sections.get("methods", ""),
        )
        for f in findings
    ]

    if not targets:
        targets.append(
            {
                "locator": "pair:abstract-methods",
                "status": "verified",
                "severity_suggestion": "minor",
                "evidence": {
                    "quote": _truncate(sections.get("abstract", ""), 280),
                    "paired_quote": _truncate(sections.get("methods", ""), 280),
                    "judge_notes": (
                        "No universality asymmetry or restriction leakage detected. "
                        "Abstract scope appears consistent with methods scope."
                    ),
                    "judge_confidence": "medium",
                    "finding_kind": "consistent",
                },
                "root_cause_key": "internal-consistency-consistent",
            }
        )
        overall_status = "verified"
        severity = "minor"
        summary = "abstract ↔ methods scope consistent"
    else:
        p0 = sum(1 for f in findings if f["severity"] == "P0")
        if p0:
            overall_status = "failed"
            severity = "P0"
            summary = (
                f"{p0} scope overclaim(s) detected: abstract commits universally "
                f"while methods restricts."
            )
        else:
            overall_status = "failed"
            severity = "major"
            summary = f"{len(findings)} scope drift(s) detected between abstract and methods."

    # If methods section is empty → unverifiable
    if not methods_markers.raw_len:
        overall_status = "unverifiable"
        severity = "P0"
        summary = "methods section missing or unparseable; cannot verify scope."

    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": overall_status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0,
            "model": None,
            "cached": False,
            "inputs": {
                "abstract_chars": abstract_markers.raw_len,
                "introduction_chars": intro_markers.raw_len,
                "methods_chars": methods_markers.raw_len,
                "results_chars": len(sections.get("results", "")),
                "conclusion_chars": len(sections.get("conclusion", "")),
            },
        },
        "errors": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check abstract↔methods scope consistency for overclaim detection."
    )
    parser.add_argument("--tex", type=Path, help="LaTeX file to parse")
    parser.add_argument(
        "--sections",
        type=Path,
        help="Pre-parsed sections JSON with keys abstract/introduction/methods/results/conclusion",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    if not args.tex and not args.sections:
        print("error: must provide --tex or --sections", file=sys.stderr)
        return 2

    if args.sections:
        sections = json.loads(args.sections.read_text(encoding="utf-8"))
        sections = {k: sections.get(k, "") for k in SECTION_KEYS}
    else:
        if not args.tex.is_file():
            print(f"error: tex file not found: {args.tex}", file=sys.stderr)
            return 2
        tex = args.tex.read_text(encoding="utf-8", errors="replace")
        sections = parse_latex(tex)

    report = run(sections=sections, out_path=args.out)
    print(f"[{VERIFIER_ID}] {report['status'].upper()}: {report['summary']}")
    for t in report["targets"]:
        print(f"  - {t['locator']}: {t['status']} (suggested={t['severity_suggestion']})")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
