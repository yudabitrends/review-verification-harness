"""extract_claims.py — Deterministic claim extractor for the verification harness.

Reads a LaTeX manuscript (and optionally a BibTeX bibliography) and emits three
JSONL files ready to be consumed by downstream verifiers:

  * citation_claims.jsonl  — every sentence containing a \\cite family command
  * numerical_claims.jsonl — numeric claims worth auditing (speedups, %, p-values,
                             absolute metrics, error bars)
  * scope_claims.jsonl     — universal-quantifier / "first to" scope commitments

Also writes ``extract_summary.json`` at the top of ``--out-dir`` that conforms
to ``references/VERIFIER_CONTRACT.md``. ``verifier_id`` is ``extract_claims``;
top-level status is ``verified`` when at least one claim was extracted, else
``unverifiable`` (downstream upgrades to P0).

Deterministic — no LLM calls. Pure Python stdlib.

Usage:
    python extract_claims.py \\
        --tex path/to/paper.tex \\
        --bib path/to/paper.bib \\
        --out-dir workspace/claims/
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

VERIFIER_ID = "extract_claims"
VERIFIER_VERSION = "0.1"
MAX_QUOTE_CHARS = 400

# -------- Patterns --------

CITE_CMD_RE = re.compile(
    r"\\(?:cite|citet|citep|citeauthor|citeyear|citealt|citealp|citenum|parencite|textcite)"
    r"(?:\[[^\]]{0,200}\])?(?:\[[^\]]{0,200}\])?\{([^{}]{0,400})\}"
)

NUM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("speedup", re.compile(
        r"\b(\d+(?:\.\d+)?)\s*[\u00d7xX]\s*"
        r"(?:speedup|speed-?up|improvement|faster|better|larger|reduction|smaller|slower)\b",
        re.IGNORECASE)),
    ("percentage", re.compile(
        r"\b(\d+(?:\.\d+)?)\s*\%\s*"
        r"(?:accuracy|improvement|increase|decrease|error|reduction|gain|drop|boost|loss)\b",
        re.IGNORECASE)),
    ("p_value", re.compile(r"\bp\s*[<=]\s*0?\.\d+\b", re.IGNORECASE)),
    ("metric", re.compile(
        r"\b(?:accuracy|precision|recall|f1|auc|auroc|bleu|rouge|mse|rmse|mae)\s+"
        r"(?:of|=)\s*\d+(?:\.\d+)?\s*\%?", re.IGNORECASE)),
    ("errorbar", re.compile(r"\b\d+(?:\.\d+)?\s*(?:\u00b1|\+\-|\+/\-)\s*\d+(?:\.\d+)?\b")),
]

SCOPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bfor all\b", r"\bfor any\b", r"\bfor every\b",
        r"\buniversal(?:ly)?\b", r"\bwithout (?:any )?assumption(?:s)?\b",
        r"\bin general\b", r"\barbitrary\b",
        r"\bholds? (?:for|in) (?:all|any|every)\b",
        r"\bfirst to\b", r"\balways\b",
        r"\bany\s+(?:continuous|smooth|bounded|finite|arbitrary|general|"
        r"system|dynamics|process|distribution|function|network|model)\b",
        r"\bunconditional(?:ly)?\b", r"\bmodel-?free\b",
        r"\bassumption-?free\b", r"\bnon-?parametric\b", r"\bregardless of\b",
    ]
]

_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")
_INLINE_CMD_RE = re.compile(
    r"\\(?:textit|textbf|texttt|textsf|textrm|textsc|emph|mathrm|mathbf|mathit)"
    r"\{([^{}]{0,400})\}"
)
_DROP_CMD_RE = re.compile(
    r"\\(?:label|ref|eqref|hyperref|url|href)\{[^{}]{0,400}\}"
)
_HEADING_RE = re.compile(
    r"\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?"
    r"\{([^{}]{1,400})\}"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")


# -------- LaTeX cleaning + splitting --------


def _soft_clean(tex: str) -> str:
    """Clean for sentence splitting while preserving \\cite commands."""
    t = _COMMENT_RE.sub("", tex)
    t = _DROP_CMD_RE.sub(" ", t)
    t = _INLINE_CMD_RE.sub(r"\1", t)
    t = re.sub(r"\$[^$]+\$", " MATH ", t)
    t = re.sub(r"\\\\", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _truncate(text: str, n: int = MAX_QUOTE_CHARS) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    return clean if len(clean) <= n else clean[: n - 1] + "\u2026"


# -------- Section-hint inference --------


def _heading_hint(heading: str) -> str:
    h = heading.lower()
    if re.search(r"\bintroduction\b|\bintro\b|\bbackground\b", h):
        return "intro"
    if re.search(r"\bmethod|approach|model|framework|theory|formulation|derivation", h):
        return "methods"
    if re.search(r"\bresult|experiment|evaluation|analysis|finding", h):
        return "results"
    if re.search(r"\babstract\b", h):
        return "abstract"
    return "unknown"


def build_section_index(tex: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if abs_match:
        spans.append((abs_match.start(), abs_match.end(), "abstract"))
    heads = list(_HEADING_RE.finditer(tex))
    for idx, m in enumerate(heads):
        start = m.end()
        end = heads[idx + 1].start() if idx + 1 < len(heads) else len(tex)
        spans.append((start, end, _heading_hint(m.group(1))))
    if not abs_match and heads:
        spans.append((0, heads[0].start(), "abstract"))
    return spans


def section_hint_for(pos: int, spans: list[tuple[int, int, str]]) -> str:
    best: tuple[int, str] | None = None
    for start, end, hint in spans:
        if start <= pos < end:
            length = end - start
            if best is None or length < best[0]:
                best = (length, hint)
    return best[1] if best else "unknown"


# -------- BibTeX parsing (minimal, stdlib-only) --------


_BIB_ENTRY_RE = re.compile(
    r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)\n\}", re.DOTALL
)
_BIB_FIELD_RE = re.compile(
    r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)\s*,?", re.DOTALL
)


def _strip_braces(value: str) -> str:
    v = value.strip()
    while len(v) >= 2 and (
        (v.startswith("{") and v.endswith("}"))
        or (v.startswith('"') and v.endswith('"'))
    ):
        v = v[1:-1].strip()
    return re.sub(r"\s+", " ", v.replace("{", "").replace("}", "")).strip()


def parse_bib(bib_text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in _BIB_ENTRY_RE.finditer(bib_text):
        key = m.group("key").strip()
        fields = {
            mm.group(1).strip().lower(): _strip_braces(mm.group(2).strip().rstrip(",").strip())
            for mm in _BIB_FIELD_RE.finditer(m.group("body"))
        }
        authors = [
            a.strip() for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()
        ]
        year: int | None = None
        ym = re.search(r"\d{4}", fields.get("year", ""))
        if ym:
            try:
                year = int(ym.group(0))
            except ValueError:
                pass
        entry = {
            "doi": fields.get("doi") or None,
            "arxiv_id": fields.get("eprint") or fields.get("arxivid") or None,
            "title": fields.get("title") or None,
            "authors": authors,
            "year": year,
        }
        out[key] = {k: v for k, v in entry.items() if v not in (None, "", [])}
    return out


def load_bib(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    try:
        return parse_bib(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        print(f"warning: could not read bib {path}: {exc}", file=sys.stderr)
        return {}


# -------- Extraction --------


@dataclass
class Extracted:
    citation_claims: list[dict[str, Any]] = field(default_factory=list)
    numerical_claims: list[dict[str, Any]] = field(default_factory=list)
    scope_claims: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _sentence_positions(cleaned: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    collapsed_cleaned: str | None = None
    for sent in split_sentences(cleaned):
        idx = cleaned.find(sent, cursor)
        if idx < 0:
            # Fallback: split_sentences may collapse whitespace, so a direct
            # find() can miss otherwise-valid sentences. Re-search after
            # collapsing \s+ → " " on both sides; only used when the primary
            # locator fails so existing callers' offsets are unchanged in the
            # common path.
            if collapsed_cleaned is None:
                collapsed_cleaned = re.sub(r"\s+", " ", cleaned)
            collapsed_sent = re.sub(r"\s+", " ", sent)
            alt = collapsed_cleaned.find(collapsed_sent, cursor)
            if alt < 0:
                continue
            idx = alt
        spans.append((idx, idx + len(sent), sent))
        cursor = idx + len(sent)
    return spans


def extract_citation_claims(
    cleaned: str,
    bib: dict[str, dict[str, Any]],
    section_spans: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    sentence_spans = _sentence_positions(cleaned)

    def sentence_for(pos: int) -> str:
        for s, e, txt in sentence_spans:
            if s <= pos < e:
                return txt
        a, b = max(0, pos - 200), min(len(cleaned), pos + 200)
        return cleaned[a:b]

    records: list[dict[str, Any]] = []
    for m in CITE_CMD_RE.finditer(cleaned):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        sent = sentence_for(m.start())
        hint = section_hint_for(m.start(), section_spans)
        for key in keys:
            ref = bib.get(key, {}) if bib else {}
            records.append({
                "locator": f"cite:{key}",
                "claim_quote": _truncate(sent),
                "reference": ref or {},
                "section_hint": hint,
            })
    return records


def extract_numerical_claims(
    cleaned: str, section_spans: list[tuple[int, int, str]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    idx = 0
    for sent_start, _, sent in _sentence_positions(cleaned):
        for kind, pattern in NUM_PATTERNS:
            for m in pattern.finditer(sent):
                value = m.group(0).strip()
                dedupe = (value.lower(), sent[:60].lower())
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                records.append({
                    "locator": f"numclaim:{idx}",
                    "claim_quote": _truncate(sent),
                    "value": value,
                    "kind": kind,
                    "section_hint": section_hint_for(sent_start, section_spans),
                })
                idx += 1
    return records


def extract_scope_claims(
    cleaned: str, section_spans: list[tuple[int, int, str]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0
    for sent_start, _, sent in _sentence_positions(cleaned):
        sig = sent.lower()[:160]
        if sig in seen:
            continue
        for pattern in SCOPE_PATTERNS:
            m = pattern.search(sent)
            if not m:
                continue
            seen.add(sig)
            records.append({
                "locator": f"scope:{idx}",
                "claim_quote": _truncate(sent),
                "quantifier_phrase": m.group(0),
                "section_hint": section_hint_for(sent_start, section_spans),
            })
            idx += 1
            break
    return records


def extract(tex: str, bib: dict[str, dict[str, Any]]) -> Extracted:
    result = Extracted()
    try:
        section_spans = build_section_index(tex)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"section index failed: {exc}")
        section_spans = []
    try:
        cleaned = _soft_clean(tex)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"latex cleaning failed: {exc}")
        cleaned = tex
    try:
        result.citation_claims = extract_citation_claims(cleaned, bib, section_spans)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"citation extraction failed: {exc}")
    try:
        result.numerical_claims = extract_numerical_claims(cleaned, section_spans)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"numerical extraction failed: {exc}")
    try:
        result.scope_claims = extract_scope_claims(cleaned, section_spans)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"scope extraction failed: {exc}")
    return result


# -------- Reporting / IO --------


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _make_target(
    locator: str, count: int, path: Path, tex_name: str, notes: str, key: str
) -> dict[str, Any]:
    return {
        "locator": locator,
        "status": "verified" if count > 0 else "unverifiable",
        "severity_suggestion": "minor" if count > 0 else "P0",
        "evidence": {
            "quote": f"{count} claim(s) extracted from {tex_name}",
            "external_url": str(path.resolve()),
            "judge_notes": notes,
            "judge_confidence": "high" if count > 0 else "medium",
        },
        "root_cause_key": key,
    }


def build_report(
    tex_path: Path, bib_path: Path | None, out_dir: Path, extracted: Extracted
) -> dict[str, Any]:
    cite_file = out_dir / "citation_claims.jsonl"
    num_file = out_dir / "numerical_claims.jsonl"
    scope_file = out_dir / "scope_claims.jsonl"
    n_cite = write_jsonl(cite_file, extracted.citation_claims)
    n_num = write_jsonl(num_file, extracted.numerical_claims)
    n_scope = write_jsonl(scope_file, extracted.scope_claims)
    total = n_cite + n_num + n_scope
    tex_name = tex_path.name

    cite_notes = (
        f"{n_cite} (sentence, cite-key) pairs emitted. "
        + ("bibliography metadata attached." if bib_path else "no --bib supplied; reference fields empty.")
    )
    num_notes = (
        "Matched speedup/percentage/p-value/metric/errorbar patterns. "
        "Deterministic regex; human must still audit borderline values."
    )
    scope_notes = (
        "Matched universal-quantifier / 'first to' / 'always' / 'for any continuous' phrases. "
        "Empty list means the paper's abstract/intro does not commit to broad scope, "
        "OR the extractor missed custom phrasing."
    )

    targets = [
        _make_target("file:citation_claims.jsonl", n_cite, cite_file, tex_name, cite_notes, "extract-citation-claims"),
        _make_target("file:numerical_claims.jsonl", n_num, num_file, tex_name, num_notes, "extract-numerical-claims"),
        _make_target("file:scope_claims.jsonl", n_scope, scope_file, tex_name, scope_notes, "extract-scope-claims"),
    ]
    # Attach a small sample so consumers can spot-check.
    for tgt, items in zip(targets, (extracted.citation_claims, extracted.numerical_claims, extracted.scope_claims)):
        tgt["evidence"]["sample"] = items[:2]

    summary = (
        f"extracted {n_cite} citation, {n_num} numerical, {n_scope} scope claims"
        if total else
        "no claims extracted — paper may be empty, malformed, or extractor patterns missed."
    )
    status = "verified" if total > 0 else "unverifiable"
    severity = "minor" if total > 0 else "P0"

    return {
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
                "tex": str(tex_path.resolve()),
                "bib": str(bib_path.resolve()) if bib_path else None,
                "out_dir": str(out_dir.resolve()),
                "n_citation_claims": n_cite,
                "n_numerical_claims": n_num,
                "n_scope_claims": n_scope,
            },
        },
        "errors": extracted.errors,
    }


def run(tex_path: Path, bib_path: Path | None, out_dir: Path) -> dict[str, Any]:
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    bib = load_bib(bib_path) if bib_path else {}
    extracted = extract(tex, bib)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(tex_path, bib_path, out_dir, extracted)
    (out_dir / "extract_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic claim extractor: LaTeX -> citation/numerical/scope JSONL."
    )
    parser.add_argument("--tex", required=True, type=Path, help="LaTeX manuscript path")
    parser.add_argument("--bib", type=Path, default=None, help="Optional BibTeX file")
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory; will be created if absent.",
    )
    args = parser.parse_args()

    if not args.tex.is_file():
        print(f"error: tex file not found: {args.tex}", file=sys.stderr)
        return 2
    if args.bib is not None and not args.bib.is_file():
        print(f"warning: bib not found, proceeding without: {args.bib}", file=sys.stderr)
        args.bib = None

    out_dir = args.out_dir.resolve()
    try:
        report = run(
            args.tex.resolve(),
            args.bib.resolve() if args.bib else None,
            out_dir,
        )
    except Exception as exc:  # noqa: BLE001
        out_dir.mkdir(parents=True, exist_ok=True)
        fallback = {
            "verifier_id": VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "summary": f"extractor crashed: {exc}",
            "targets": [],
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cost_usd": 0.0,
                "model": None,
                "cached": False,
                "inputs": {
                    "tex": str(args.tex),
                    "bib": str(args.bib) if args.bib else None,
                    "out_dir": str(out_dir),
                },
            },
            "errors": [str(exc)],
        }
        (out_dir / "extract_summary.json").write_text(
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
