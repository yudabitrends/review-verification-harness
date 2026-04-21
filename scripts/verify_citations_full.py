"""verify_citations_full.py — Citation content-match verifier.

Given a JSONL file of claim/reference pairs extracted from a paper, resolve each
reference via CrossRef / arXiv / Semantic Scholar, fetch its abstract, and ask an
LLM judge whether the abstract actually supports the specific claim attributed
to it.

Outputs JSON conforming to `references/VERIFIER_CONTRACT.md`. Each target's
status is `verified` (judge confirmed match, or judge confirmed mismatch — both
are "we checked"), `unverifiable` (couldn't resolve / fetch / get confident
judgment), or `failed` (claim decisively contradicted by abstract).

Usage:
    python verify_citations_full.py \
        --claims workspace/citation_claims.jsonl \
        --out workspace/verifier/citations.json \
        [--fast] [--no-cache]

Input JSONL format (one object per line):
    {
      "locator": "cite:smith2024",
      "claim_quote": "Smith (2024) shows linear scaling for n<1000.",
      "reference": {
        "doi": "10.1234/smith.2024",       // optional
        "arxiv_id": "2401.01234",           // optional
        "title": "Linear scaling methods",  // optional but recommended
        "authors": ["Smith, J."],
        "year": 2024
      }
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERIFIER_ID = "verify_citations_full"
VERIFIER_VERSION = "0.4-cc-bridge"

# Shared CC-bridge judge-backend helper. Imported defensively so older
# invocations that load this module via importlib from arbitrary cwds still
# work — `_judge_backend` lives beside this file.
try:
    import _judge_backend  # type: ignore
except ImportError:  # pragma: no cover — defensive fallback for test loaders
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import _judge_backend  # type: ignore
    except ImportError:
        _judge_backend = None  # type: ignore

USER_AGENT = f"{VERIFIER_ID}/{VERIFIER_VERSION} (mailto:yudabitrends@gmail.com)"
CACHE_ROOT = Path.home() / ".claude" / "cache" / "review-verification-harness" / VERIFIER_ID
CACHE_TTL_SECONDS = 30 * 24 * 3600

HTTP_TIMEOUT = 15
MAX_RETRIES = 2


# ----------------------------- HTTP + cache helpers -----------------------------


def _cache_key(namespace: str, payload: str) -> Path:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / f"{digest}.json"


def _cache_load(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - data.get("_cached_at", 0) > CACHE_TTL_SECONDS:
        return None
    return data.get("payload")


def _cache_store(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"_cached_at": time.time(), "payload": payload}, ensure_ascii=False),
        encoding="utf-8",
    )


def _http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return resp.getcode(), resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return exc.code, b""
            last_err = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"HTTP GET failed for {url}: {last_err}")


# ----------------------------- Metadata resolvers -----------------------------


@dataclass
class ResolvedRef:
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    external_url: str | None = None
    source: str | None = None  # crossref | arxiv | semanticscholar
    raw_errors: list[str] = field(default_factory=list)

    def sufficient(self) -> bool:
        return bool(self.abstract and self.abstract.strip())


def resolve_by_doi(doi: str, use_cache: bool = True) -> ResolvedRef:
    ref = ResolvedRef()
    doi_clean = doi.strip()
    for prefix in ("https://", "http://", "doi.org/", "dx.doi.org/"):
        doi_clean = doi_clean.removeprefix(prefix)
    doi_clean = doi_clean.removeprefix("doi.org/").removeprefix("dx.doi.org/")
    if not doi_clean:
        ref.raw_errors.append("empty doi after normalization")
        return ref

    ref.external_url = f"https://doi.org/{doi_clean}"

    # 1) CrossRef
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi_clean, safe='')}"
    cache_path = _cache_key("crossref", doi_clean) if use_cache else None
    cached = _cache_load(cache_path) if cache_path else None
    try:
        if cached is None:
            code, body = _http_get(url)
            if code == 200:
                payload = json.loads(body.decode("utf-8", "replace"))
                if cache_path:
                    _cache_store(cache_path, payload)
                cached = payload
            else:
                ref.raw_errors.append(f"crossref returned HTTP {code} for {doi_clean}")
        if cached:
            msg = cached.get("message", {})
            title_list = msg.get("title") or []
            ref.title = title_list[0] if title_list else None
            ref.abstract = _strip_jats(msg.get("abstract"))
            year_parts = (msg.get("issued", {}).get("date-parts") or [[None]])[0]
            if year_parts and year_parts[0]:
                ref.year = int(year_parts[0])
            ref.authors = [
                f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                for a in msg.get("author", [])
            ]
            ref.venue = msg.get("container-title", [None])[0]
            ref.source = "crossref"
    except Exception as exc:  # noqa: BLE001
        ref.raw_errors.append(f"crossref error: {exc}")

    if ref.sufficient():
        return ref

    # 2) Semantic Scholar by DOI (has abstracts crossref often lacks)
    s2 = _semantic_scholar_lookup(f"DOI:{doi_clean}", use_cache=use_cache)
    _merge_resolved(ref, s2)
    return ref


def resolve_by_arxiv(arxiv_id: str, use_cache: bool = True) -> ResolvedRef:
    ref = ResolvedRef()
    clean = arxiv_id.strip().removeprefix("arxiv:").removeprefix("arXiv:")
    clean = clean.split("v")[0] if re.search(r"v\d+$", clean) else clean
    ref.external_url = f"https://arxiv.org/abs/{clean}"
    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(clean)}"
    cache_path = _cache_key("arxiv", clean) if use_cache else None
    cached = _cache_load(cache_path) if cache_path else None
    try:
        if cached is None:
            code, body = _http_get(url)
            if code == 200:
                cached = body.decode("utf-8", "replace")
                if cache_path:
                    _cache_store(cache_path, cached)
            else:
                ref.raw_errors.append(f"arxiv returned HTTP {code} for {clean}")
        if cached:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(cached)
            entries = root.findall("a:entry", ns)
            if entries:
                entry = entries[0]
                ref.title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
                ref.abstract = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
                published = entry.findtext("a:published", default="", namespaces=ns)
                if published[:4].isdigit():
                    ref.year = int(published[:4])
                ref.authors = [
                    (a.findtext("a:name", default="", namespaces=ns) or "").strip()
                    for a in entry.findall("a:author", ns)
                ]
                ref.source = "arxiv"
    except Exception as exc:  # noqa: BLE001
        ref.raw_errors.append(f"arxiv error: {exc}")
    return ref


def resolve_by_title(
    title: str,
    year: int | None = None,
    use_cache: bool = True,
    authors: list[str] | None = None,
) -> ResolvedRef:
    if not title or len(title.strip()) < 6:
        return ResolvedRef(raw_errors=["title too short to search"])
    query_key = f"{title}|{year or ''}|{','.join(sorted(authors or []))}"
    return _semantic_scholar_search(title, year, query_key, use_cache, expected_authors=authors)


def _semantic_scholar_lookup(identifier: str, use_cache: bool) -> ResolvedRef:
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(identifier, safe='')}"
        "?fields=title,abstract,year,authors,venue,externalIds"
    )
    cache_path = _cache_key("s2_lookup", identifier) if use_cache else None
    cached = _cache_load(cache_path) if cache_path else None
    ref = ResolvedRef()
    try:
        if cached is None:
            code, body = _http_get(url)
            if code == 200:
                cached = json.loads(body.decode("utf-8", "replace"))
                if cache_path:
                    _cache_store(cache_path, cached)
            elif code in (404, 429):
                ref.raw_errors.append(f"semantic scholar HTTP {code}")
                return ref
        if cached:
            ref.title = cached.get("title")
            ref.abstract = cached.get("abstract")
            ref.year = cached.get("year")
            ref.authors = [a.get("name", "") for a in cached.get("authors") or []]
            ref.venue = cached.get("venue")
            ext = cached.get("externalIds") or {}
            ref.external_url = (
                f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else ref.external_url
            )
            ref.source = "semanticscholar"
    except Exception as exc:  # noqa: BLE001
        ref.raw_errors.append(f"semanticscholar lookup error: {exc}")
    return ref


def _semantic_scholar_search(
    title: str,
    year: int | None,
    query_key: str,
    use_cache: bool,
    expected_authors: list[str] | None = None,
) -> ResolvedRef:
    params = {
        "query": title,
        "limit": "5",
        "fields": "title,abstract,year,authors,venue,externalIds",
    }
    if year:
        params["year"] = str(year)
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    cache_path = _cache_key("s2_search", query_key) if use_cache else None
    cached = _cache_load(cache_path) if cache_path else None
    ref = ResolvedRef()
    try:
        if cached is None:
            code, body = _http_get(url)
            if code == 200:
                cached = json.loads(body.decode("utf-8", "replace"))
                if cache_path:
                    _cache_store(cache_path, cached)
            else:
                ref.raw_errors.append(f"semanticscholar search HTTP {code}")
                return ref
        if cached and cached.get("data"):
            best = _pick_best_match(
                cached["data"], title=title, year=year, authors=expected_authors
            )
            if best is None:
                titles_seen = [h.get("title") for h in cached["data"][:3]]
                ref.raw_errors.append(
                    f"semanticscholar search: no hit met title+year+author threshold; "
                    f"wanted {title!r} year={year}, saw {titles_seen}"
                )
            else:
                ref.title = best.get("title")
                ref.abstract = best.get("abstract")
                ref.year = best.get("year")
                ref.authors = [a.get("name", "") for a in best.get("authors") or []]
                ref.venue = best.get("venue")
                ext = best.get("externalIds") or {}
                if ext.get("DOI"):
                    ref.external_url = f"https://doi.org/{ext['DOI']}"
                ref.source = "semanticscholar_search"
    except Exception as exc:  # noqa: BLE001
        ref.raw_errors.append(f"semanticscholar search error: {exc}")
    return ref


def _pick_best_match(
    hits: list[dict[str, Any]],
    *,
    title: str,
    year: int | None,
    authors: list[str] | None,
) -> dict[str, Any] | None:
    """Return the first hit whose title overlaps ≥0.7, year matches within ±1,
    and at least one author-surname matches when expected_authors was provided.

    Returning None is intentional: downstream treats as unresolvable so the
    target is flagged P0 rather than silently accepted."""
    expected_surnames = _extract_surnames(authors or [])
    for hit in hits:
        if not _title_close(hit.get("title", ""), title, threshold=0.7):
            continue
        if year is not None and hit.get("year") is not None:
            if abs(int(hit["year"]) - int(year)) > 1:
                continue
        if expected_surnames:
            hit_surnames = _extract_surnames(
                [a.get("name", "") for a in hit.get("authors") or []]
            )
            if not (expected_surnames & hit_surnames):
                continue
        return hit
    return None


def _extract_surnames(authors: list[str]) -> set[str]:
    surnames: set[str] = set()
    for raw in authors:
        if not raw:
            continue
        token = raw.strip()
        if "," in token:
            # "Smith, John"
            surname = token.split(",", 1)[0]
        else:
            # "John Smith" — last whitespace-separated token
            parts = token.split()
            surname = parts[-1] if parts else ""
        surname = re.sub(r"[^a-z]", "", surname.lower())
        if len(surname) >= 2:
            surnames.add(surname)
    return surnames


def _merge_resolved(primary: ResolvedRef, secondary: ResolvedRef) -> None:
    for field_name in ("title", "abstract", "year", "venue", "external_url", "source"):
        if not getattr(primary, field_name) and getattr(secondary, field_name):
            setattr(primary, field_name, getattr(secondary, field_name))
    if not primary.authors and secondary.authors:
        primary.authors = secondary.authors
    primary.raw_errors.extend(secondary.raw_errors)


def _title_close(a: str, b: str, threshold: float = 0.6) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    a_norm, b_norm = norm(a), norm(b)
    if not a_norm or not b_norm:
        return False
    a_words, b_words = set(a_norm.split()), set(b_norm.split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / max(len(a_words), len(b_words))
    return overlap >= threshold


def _strip_jats(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<[^>]+>", " ", text).strip()


# ----------------------------- LLM judge -----------------------------


def _llm_judge_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


JUDGE_SYSTEM = """You are a strict scientific citation auditor.
Your only job is to decide whether a cited paper (given by its abstract) actually supports the specific claim being attributed to it in another paper.

Respond ONLY with a JSON object:
{
  "verdict": "supports" | "contradicts" | "not_addressed" | "insufficient_context",
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence pointing to the specific claim↔abstract alignment or mismatch"
}

Definitions:
- supports: the abstract clearly makes a claim equivalent to, or that entails, the attributed claim.
- contradicts: the abstract clearly says the opposite of the attributed claim, OR describes a different regime where the claim fails.
- not_addressed: the abstract does not touch on the attributed claim at all (wrong paper cited).
- insufficient_context: the abstract is too short or too generic to decide.

Be paranoid. If the attributed claim is specific (numbers, regime, method) but the abstract is generic, return insufficient_context or not_addressed — NOT supports.
"""


ADVERSARIAL_JUDGE_SYSTEM = """You are a skeptical peer reviewer hunting for citation misattribution.
Your job is to argue as forcefully as you can that the cited paper (given by its abstract) does NOT support the specific claim being attributed to it.
Only if, after trying to attack, you still find the claim ↔ abstract match defensible, return supports.

Respond ONLY with a JSON object:
{
  "verdict": "supports" | "contradicts" | "not_addressed" | "insufficient_context",
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence — if you returned supports, explain why the attack failed; otherwise explain the attack"
}

Bias toward contradicts / not_addressed when in doubt. A false "supports" is worse than a false "not_addressed"."""


def llm_judge(
    claim_quote: str,
    ref_title: str | None,
    ref_abstract: str,
    model: str | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    """Invoke the Anthropic API to judge citation-content match.

    Two-pass adversarial design:
      1. Primary judge (strict auditor prompt) — default conservative stance.
      2. Second judge (devil's advocate prompt) — actively tries to disprove.
    Both must return 'supports' with high/medium confidence for the target to
    be flagged `verified`. Any disagreement → `unverifiable` (upgraded to P0).

    Falls back to a deterministic heuristic when ANTHROPIC_API_KEY is not set,
    marking the result as "insufficient_context" so downstream logic upgrades
    the finding to `unverifiable`.
    """
    if not _llm_judge_available():
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "ANTHROPIC_API_KEY not configured; no external judge available",
            "model": None,
            "stub": True,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    try:
        import anthropic  # type: ignore
    except ImportError:
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "anthropic SDK not installed; no external judge available",
            "model": None,
            "stub": True,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    client = anthropic.Anthropic()
    chosen_model = model or os.environ.get(
        "REVIEW_VERIFIER_MODEL", "claude-sonnet-4-6"
    )
    prompt = (
        f"Claim made in paper under review:\n---\n{claim_quote}\n---\n\n"
        f"Abstract of cited reference (title: {ref_title or 'N/A'}):\n---\n{ref_abstract}\n---"
    )
    total_in, total_out = 0, 0
    try:
        resp = client.messages.create(
            model=chosen_model,
            max_tokens=400,
            temperature=0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        body = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        total_in += getattr(resp.usage, "input_tokens", 0)
        total_out += getattr(resp.usage, "output_tokens", 0)
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": f"judge API error: {exc}",
            "model": chosen_model,
            "stub": False,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    verdict = _parse_judge_json(body)
    verdict["model"] = chosen_model
    verdict["stub"] = False
    if fast:
        verdict["input_tokens"] = total_in
        verdict["output_tokens"] = total_out
        return verdict

    # Adversarial second opinion: explicitly attacks the attribution. Requires
    # the claim to survive hostile questioning, not just repeated affirmation.
    try:
        resp2 = client.messages.create(
            model=chosen_model,
            max_tokens=400,
            temperature=0,
            system=ADVERSARIAL_JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Cited paper's abstract (title: {ref_title or 'N/A'}):\n---\n{ref_abstract}\n---\n\n"
                        f"Quote from paper-under-review that attributes a claim to this citation:\n---\n{claim_quote}\n---\n\n"
                        "Attack the attribution. Where might the attributed claim go beyond what the abstract actually says? "
                        "Is the regime, the quantifier, the direction, or the object of the claim subtly different? "
                        "Emit the JSON verdict."
                    ),
                }
            ],
        )
        body2 = "".join(
            block.text for block in resp2.content if getattr(block, "type", "") == "text"
        )
        total_in += getattr(resp2.usage, "input_tokens", 0)
        total_out += getattr(resp2.usage, "output_tokens", 0)
        verdict2 = _parse_judge_json(body2)
        verdict["second_opinion_verdict"] = verdict2.get("verdict")
        verdict["second_opinion_confidence"] = verdict2.get("confidence")
        verdict["second_opinion_reason"] = verdict2.get("reason")
        # Agreement = same verdict AND neither is low-confidence.
        both_supports = (
            verdict.get("verdict") == "supports"
            and verdict2.get("verdict") == "supports"
        )
        both_non_low = (
            verdict.get("confidence") in {"high", "medium"}
            and verdict2.get("confidence") in {"high", "medium"}
        )
        verdict["second_opinion_agreed"] = (
            verdict.get("verdict") == verdict2.get("verdict")
            and (not both_supports or both_non_low)
        )
    except Exception as exc:  # noqa: BLE001
        verdict["second_opinion_agreed"] = False
        verdict["second_opinion_verdict"] = None
        verdict["second_opinion_error"] = str(exc)
    verdict["input_tokens"] = total_in
    verdict["output_tokens"] = total_out
    return verdict


def _parse_judge_json(body: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", body, flags=re.DOTALL)
    if not match:
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "judge response not JSON",
            "parse_error": "no JSON object found",
        }
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "verdict": "insufficient_context",
            "confidence": "low",
            "reason": "judge response JSON malformed",
            "parse_error": str(exc),
        }
    raw_verdict = data.get("verdict")
    if raw_verdict not in {"supports", "contradicts", "not_addressed", "insufficient_context"}:
        # Preserve the bad value under a separate key so we can audit later
        # instead of silently collapsing it to insufficient_context.
        data["raw_verdict"] = raw_verdict
        data["verdict"] = "insufficient_context"
        data["parse_error"] = f"unknown verdict {raw_verdict!r}"
    if data.get("confidence") not in {"high", "medium", "low"}:
        data["confidence"] = "low"
    return data


# ----------------------------- Verifier core -----------------------------


def resolve_reference(ref: dict[str, Any], use_cache: bool) -> ResolvedRef:
    resolved = ResolvedRef()
    if ref.get("doi"):
        resolved = resolve_by_doi(ref["doi"], use_cache=use_cache)
    if not resolved.sufficient() and ref.get("arxiv_id"):
        arxiv_ref = resolve_by_arxiv(ref["arxiv_id"], use_cache=use_cache)
        _merge_resolved(resolved, arxiv_ref)
    if not resolved.sufficient() and ref.get("title"):
        title_ref = resolve_by_title(
            ref["title"],
            ref.get("year"),
            use_cache=use_cache,
            authors=ref.get("authors"),
        )
        _merge_resolved(resolved, title_ref)

    # Author-sanity cross-check: if DOI resolved but authors disagree with the
    # paper's bibliography entry, this is almost always a typo'd DOI pointing
    # at a different paper. Downgrade to insufficient so judge runs with caution.
    if resolved.sufficient() and ref.get("authors") and resolved.authors:
        expected = _extract_surnames(ref["authors"])
        found = _extract_surnames(resolved.authors)
        if expected and found and not (expected & found):
            resolved.raw_errors.append(
                f"author mismatch: bibliography lists {sorted(expected)}, "
                f"DOI resolves to {sorted(found)} — likely wrong DOI or stale reference"
            )
            # Do not clear abstract (so judge can still see it) but tag source
            # as suspicious so classify_target downgrades to unverifiable.
            resolved.source = (resolved.source or "") + ":AUTHOR_MISMATCH"
    return resolved


def classify_target(
    claim_quote: str, ref_meta: dict[str, Any], resolved: ResolvedRef, fast: bool
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "quote": claim_quote,
        "external_url": resolved.external_url,
        "resolver_source": resolved.source,
        "resolver_errors": resolved.raw_errors,
        "resolved_title": resolved.title,
        "resolved_year": resolved.year,
    }

    if not resolved.sufficient():
        evidence["judge_notes"] = (
            "Could not resolve citation to a readable abstract. "
            "Reference may be fabricated, paywalled without API access, or citation metadata is wrong."
        )
        # Network failure (TimeoutError / URLError / 5xx) is an env condition;
        # a clean "not found" from CrossRef/Semantic Scholar is evidence.
        errs = " ".join(resolved.raw_errors).lower()
        network_markers = ("timeout", "urlerror", "failed to", "connect", "network",
                           "http 5", "http 429", "http 403")
        kind = "env" if any(m in errs for m in network_markers) else "evidence"
        evidence["unverifiable_kind"] = kind
        return {
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "unresolvable"),
        }

    judge = llm_judge(
        claim_quote=claim_quote,
        ref_title=resolved.title,
        ref_abstract=resolved.abstract or "",
        fast=fast,
    )
    evidence["judge_verdict"] = judge.get("verdict")
    evidence["judge_confidence"] = judge.get("confidence")
    evidence["judge_notes"] = judge.get("reason")
    evidence["judge_model"] = judge.get("model")
    evidence["judge_stub"] = judge.get("stub", False)
    evidence["second_opinion_agreed"] = judge.get("second_opinion_agreed")
    evidence["second_opinion_verdict"] = judge.get("second_opinion_verdict")
    evidence["second_opinion_reason"] = judge.get("second_opinion_reason")
    evidence["judge_input_tokens"] = judge.get("input_tokens", 0)
    evidence["judge_output_tokens"] = judge.get("output_tokens", 0)
    # If the resolver flagged an author mismatch, force unverifiable so the
    # judge result — which may be "supports" on the wrong paper's abstract —
    # cannot short-circuit the P0 gate.
    author_mismatch = bool(resolved.source and "AUTHOR_MISMATCH" in resolved.source)

    verdict = judge.get("verdict")
    agreed = judge.get("second_opinion_agreed", True) if not fast else True

    if author_mismatch:
        evidence["judge_notes"] = (
            (evidence.get("judge_notes") or "")
            + " | Resolver author mismatch — DOI/title may point to a different paper than cited."
        )
        evidence["unverifiable_kind"] = "evidence"
        return {
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "author-mismatch"),
        }

    if judge.get("stub") or verdict == "insufficient_context" or not agreed:
        # Stub means the LLM judge couldn't run (no API key / SDK) — that's env.
        # insufficient_context / second-opinion disagreement are evidence gaps.
        evidence["unverifiable_kind"] = "env" if judge.get("stub") else "evidence"
        return {
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "judge-unresolved"),
        }
    if verdict == "supports":
        return {
            "status": "verified",
            "severity_suggestion": "minor",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "match"),
        }
    if verdict == "contradicts":
        return {
            "status": "failed",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "contradicts"),
        }
    # not_addressed
    return {
        "status": "failed",
        "severity_suggestion": "P0",
        "evidence": evidence,
        "root_cause_key": _root_cause_key(ref_meta, "not-addressed"),
    }


def _root_cause_key(ref_meta: dict[str, Any], tag: str) -> str:
    ident = (
        ref_meta.get("doi")
        or ref_meta.get("arxiv_id")
        or ref_meta.get("title", "unknown")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", str(ident).lower()).strip("-") or "unknown"
    return f"citation-{tag}-{slug[:60]}"


def load_claims(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"claims file not found: {path}")
    items: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        if not obj.get("locator") or not obj.get("claim_quote"):
            raise ValueError(f"{path}:{line_no} missing required locator/claim_quote")
        ref = obj.get("reference") or {}
        if not ref:
            print(
                f"warning: line {line_no}: citation locator={obj['locator']} has "
                f"empty reference {{}}; all bib fields missing (no --bib supplied?) — "
                "verification will return unverifiable P0",
                file=sys.stderr,
            )
        items.append(obj)
    return items


def aggregate_status(targets: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not targets:
        return "unverifiable", "P0", "no citation targets provided"

    n_failed = sum(1 for t in targets if t["status"] == "failed")
    n_unverifiable = sum(1 for t in targets if t["status"] == "unverifiable")
    n_verified = sum(1 for t in targets if t["status"] == "verified")

    if n_failed:
        return (
            "failed",
            "P0",
            f"{n_failed} of {len(targets)} citations contradicted or misattributed; "
            f"{n_unverifiable} unverifiable; {n_verified} verified",
        )
    if n_unverifiable:
        return (
            "unverifiable",
            "P0",
            f"{n_unverifiable} of {len(targets)} citations could not be verified "
            f"(unresolvable or judge could not decide); {n_verified} verified",
        )
    return (
        "verified",
        "minor",
        f"all {n_verified} citations verified as supporting the attributed claim",
    )


def _estimate_cost_usd(
    model: str, total_in: int, total_out: int
) -> float | None:
    """Approximate cost for common Claude models. None when unknown."""
    # Rates are indicative (USD per MTok); update as Anthropic revises pricing.
    rates = {
        "claude-opus-4-7": (15.0, 75.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5-20251001": (1.0, 5.0),
    }
    base = model.split("[")[0]  # strip any [1m] suffix
    if base not in rates:
        return None
    in_rate, out_rate = rates[base]
    return round((total_in / 1e6) * in_rate + (total_out / 1e6) * out_rate, 4)


def _judge_prompt(claim_quote: str, ref_title: str | None, ref_abstract: str) -> str:
    """Rebuild the exact user prompt the SDK path uses for the primary judge."""
    return (
        f"Claim made in paper under review:\n---\n{claim_quote}\n---\n\n"
        f"Abstract of cited reference (title: {ref_title or 'N/A'}):\n---\n{ref_abstract}\n---"
    )


def _adversarial_prompt(claim_quote: str, ref_title: str | None, ref_abstract: str) -> str:
    return (
        f"Cited paper's abstract (title: {ref_title or 'N/A'}):\n---\n{ref_abstract}\n---\n\n"
        f"Quote from paper-under-review that attributes a claim to this citation:\n---\n{claim_quote}\n---\n\n"
        "Attack the attribution. Where might the attributed claim go beyond what the abstract actually says? "
        "Is the regime, the quantifier, the direction, or the object of the claim subtly different? "
        "Emit the JSON verdict."
    )


def _resolved_to_dict(r: ResolvedRef) -> dict[str, Any]:
    return {
        "title": r.title,
        "abstract": r.abstract,
        "year": r.year,
        "authors": list(r.authors),
        "venue": r.venue,
        "external_url": r.external_url,
        "source": r.source,
        "raw_errors": list(r.raw_errors),
    }


def _dict_to_resolved(d: dict[str, Any]) -> ResolvedRef:
    return ResolvedRef(
        title=d.get("title"),
        abstract=d.get("abstract"),
        year=d.get("year"),
        authors=list(d.get("authors") or []),
        venue=d.get("venue"),
        external_url=d.get("external_url"),
        source=d.get("source"),
        raw_errors=list(d.get("raw_errors") or []),
    )


def _classify_with_prebuilt_judge(
    claim_quote: str,
    ref_meta: dict[str, Any],
    resolved: ResolvedRef,
    judge: dict[str, Any],
    fast: bool,
) -> dict[str, Any]:
    """Same shape as `classify_target` but consumes a judge dict the caller
    assembled from batch-ingest results instead of calling the SDK."""
    evidence: dict[str, Any] = {
        "quote": claim_quote,
        "external_url": resolved.external_url,
        "resolver_source": resolved.source,
        "resolver_errors": resolved.raw_errors,
        "resolved_title": resolved.title,
        "resolved_year": resolved.year,
        "judge_verdict": judge.get("verdict"),
        "judge_confidence": judge.get("confidence"),
        "judge_notes": judge.get("reason"),
        "judge_model": judge.get("model"),
        "judge_stub": judge.get("stub", False),
        "second_opinion_agreed": judge.get("second_opinion_agreed"),
        "second_opinion_verdict": judge.get("second_opinion_verdict"),
        "second_opinion_reason": judge.get("second_opinion_reason"),
        "judge_input_tokens": judge.get("input_tokens", 0),
        "judge_output_tokens": judge.get("output_tokens", 0),
        "judge_method": "cc_bridge",
    }
    author_mismatch = bool(resolved.source and "AUTHOR_MISMATCH" in resolved.source)
    verdict = judge.get("verdict")
    agreed = judge.get("second_opinion_agreed", True) if not fast else True

    if author_mismatch:
        evidence["judge_notes"] = (
            (evidence.get("judge_notes") or "")
            + " | Resolver author mismatch — DOI/title may point to a different paper than cited."
        )
        evidence["unverifiable_kind"] = "evidence"
        return {
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "author-mismatch"),
        }
    if judge.get("stub") or verdict == "insufficient_context" or not agreed:
        evidence["unverifiable_kind"] = "env" if judge.get("stub") else "evidence"
        return {
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "judge-unresolved"),
        }
    if verdict == "supports":
        return {
            "status": "verified",
            "severity_suggestion": "minor",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "match"),
        }
    if verdict == "contradicts":
        return {
            "status": "failed",
            "severity_suggestion": "P0",
            "evidence": evidence,
            "root_cause_key": _root_cause_key(ref_meta, "contradicts"),
        }
    # not_addressed or unknown
    return {
        "status": "failed",
        "severity_suggestion": "P0",
        "evidence": evidence,
        "root_cause_key": _root_cause_key(ref_meta, "not-addressed"),
    }


def _run_batch_emit(
    claims: list[dict[str, Any]],
    out_path: Path,
    fast: bool,
    use_cache: bool,
    backend,
) -> dict[str, Any]:
    """Resolve references deterministically, write LLM tasks + state, exit.

    Produces an interim JSON report whose pending targets carry
    ``status="pending_llm"``. The orchestrator (`cc_run_round.py`) dispatches
    subagents through the task file and then invokes `--judge-backend
    batch-ingest` to finalize.
    """
    _judge_backend.ensure_emit_args(backend)
    pending_targets: list[dict[str, Any]] = []
    final_targets: list[dict[str, Any]] = []
    resolved_refs: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for claim in claims:
        ref_meta = claim.get("reference", {})
        locator = claim["locator"]
        try:
            resolved = resolve_reference(ref_meta, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001
            final_targets.append({
                "locator": locator,
                "status": "unverifiable",
                "severity_suggestion": "P0",
                "evidence": {
                    "quote": claim["claim_quote"],
                    "judge_notes": f"resolver crashed: {exc}",
                    "unverifiable_kind": "env",
                },
                "root_cause_key": _root_cause_key(ref_meta, "resolver-crash"),
            })
            continue
        if not resolved.sufficient():
            # Already a final target — no LLM needed.
            evidence: dict[str, Any] = {
                "quote": claim["claim_quote"],
                "external_url": resolved.external_url,
                "resolver_source": resolved.source,
                "resolver_errors": resolved.raw_errors,
                "resolved_title": resolved.title,
                "resolved_year": resolved.year,
                "judge_notes": (
                    "Could not resolve citation to a readable abstract."),
            }
            errs = " ".join(resolved.raw_errors).lower()
            network_markers = ("timeout", "urlerror", "failed to", "connect",
                               "network", "http 5", "http 429", "http 403")
            evidence["unverifiable_kind"] = (
                "env" if any(m in errs for m in network_markers) else "evidence"
            )
            final_targets.append({
                "locator": locator,
                "status": "unverifiable",
                "severity_suggestion": "P0",
                "evidence": evidence,
                "root_cause_key": _root_cause_key(ref_meta, "unresolvable"),
            })
            continue
        resolved_refs[locator] = _resolved_to_dict(resolved)
        pending_targets.append({
            "locator": locator,
            "status": "pending_llm",
            "severity_suggestion": "major",
            "evidence": {
                "quote": claim["claim_quote"],
                "external_url": resolved.external_url,
                "resolver_source": resolved.source,
                "resolver_errors": resolved.raw_errors,
                "resolved_title": resolved.title,
                "resolved_year": resolved.year,
                "judge_notes": "awaiting LLM judgment via CC-bridge dispatch",
            },
            "root_cause_key": _root_cause_key(ref_meta, "pending-llm"),
            "_claim_quote": claim["claim_quote"],
            "_ref_meta": ref_meta,
        })
        # Primary judge task
        tasks.append(_judge_backend.build_task(
            task_id=f"{locator}:primary",
            verifier=VERIFIER_ID,
            target_locator=locator,
            kind="primary_judge",
            system=JUDGE_SYSTEM,
            user=_judge_prompt(
                claim["claim_quote"], resolved.title, resolved.abstract or ""),
            max_tokens=400, temperature=0, expected_format="json",
        ))
        if not fast:
            tasks.append(_judge_backend.build_task(
                task_id=f"{locator}:devils_advocate",
                verifier=VERIFIER_ID,
                target_locator=locator,
                kind="devils_advocate",
                system=ADVERSARIAL_JUDGE_SYSTEM,
                user=_adversarial_prompt(
                    claim["claim_quote"], resolved.title, resolved.abstract or ""),
                max_tokens=400, temperature=0, expected_format="json",
            ))
    _judge_backend.write_tasks(backend.tasks_out, tasks)

    state = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "phase": "batch-emit",
        "pending_targets": pending_targets,
        "final_targets": final_targets,
        "resolved_refs": resolved_refs,
        "fast": fast,
        "out_path": str(out_path),
        "use_cache": use_cache,
    }
    _judge_backend.write_state(backend.state_file, state)

    # Interim report (all current-status targets written out for inspection).
    interim_targets = [
        {k: v for k, v in t.items() if not k.startswith("_")}
        for t in pending_targets
    ] + final_targets
    interim_status = "pending_llm" if pending_targets else (
        "unverifiable" if final_targets else "verified")
    severity = "minor" if interim_status in {"pending_llm", "verified"} else "P0"
    summary = (
        f"batch-emit: {len(tasks)} task(s) queued across "
        f"{len(pending_targets)} pending target(s); "
        f"{len(final_targets)} already-final target(s)"
    )
    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": interim_status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": interim_targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0,
            "tokens_in": 0, "tokens_out": 0,
            "model": os.environ.get("REVIEW_VERIFIER_MODEL", "claude-sonnet-4-6"),
            "cached": use_cache,
            "judge_backend": "batch-emit",
            "n_llm_tasks_emitted": len(tasks),
            "inputs": {"n_targets": len(interim_targets), "fast_mode": fast},
        },
        "errors": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _run_batch_ingest(
    out_path: Path, backend,
) -> dict[str, Any]:
    """Read state + results and produce the final verifier report."""
    _judge_backend.ensure_ingest_args(backend)
    state = _judge_backend.read_state(backend.state_file)
    results = _judge_backend.load_results(backend.results_in)
    fast = bool(state.get("fast"))
    use_cache = bool(state.get("use_cache", True))
    final_targets: list[dict[str, Any]] = list(state.get("final_targets") or [])
    pending = state.get("pending_targets") or []
    resolved_refs = state.get("resolved_refs") or {}
    errors: list[str] = []

    for ptarget in pending:
        locator = ptarget["locator"]
        claim_quote = ptarget.get("_claim_quote") or ptarget["evidence"].get("quote", "")
        ref_meta = ptarget.get("_ref_meta", {}) or {}
        resolved = _dict_to_resolved(resolved_refs.get(locator, {}))

        primary_res = results.get(f"{locator}:primary")
        if primary_res is None:
            final_targets.append({
                "locator": locator,
                "status": "unverifiable",
                "severity_suggestion": "P0",
                "evidence": {
                    "quote": claim_quote,
                    "judge_notes": "primary-judge result missing in ingest",
                    "unverifiable_kind": "env",
                },
                "root_cause_key": _root_cause_key(ref_meta, "judge-missing"),
            })
            errors.append(f"{locator}: primary result missing")
            continue
        judge = _judge_backend.parse_json_body(primary_res)
        # Normalize verdict
        if judge.get("verdict") not in {"supports", "contradicts",
                                        "not_addressed", "insufficient_context"}:
            judge["raw_verdict"] = judge.get("verdict")
            judge["verdict"] = "insufficient_context"
        if judge.get("confidence") not in {"high", "medium", "low"}:
            judge["confidence"] = "low"
        judge.setdefault("stub", False)
        judge.setdefault("model", primary_res.get("model"))
        judge["input_tokens"] = int(primary_res.get("tokens_in") or 0)
        judge["output_tokens"] = int(primary_res.get("tokens_out") or 0)

        if not fast:
            second = results.get(f"{locator}:devils_advocate")
            if second is None:
                judge["second_opinion_verdict"] = None
                judge["second_opinion_agreed"] = False
                judge["second_opinion_reason"] = "devils-advocate result missing"
            else:
                parsed2 = _judge_backend.parse_json_body(second)
                if parsed2.get("verdict") not in {"supports", "contradicts",
                                                  "not_addressed",
                                                  "insufficient_context"}:
                    parsed2["verdict"] = "insufficient_context"
                if parsed2.get("confidence") not in {"high", "medium", "low"}:
                    parsed2["confidence"] = "low"
                judge["second_opinion_verdict"] = parsed2.get("verdict")
                judge["second_opinion_confidence"] = parsed2.get("confidence")
                judge["second_opinion_reason"] = parsed2.get("reason")
                both_supports = (judge.get("verdict") == "supports"
                                 and parsed2.get("verdict") == "supports")
                both_non_low = (judge.get("confidence") in {"high", "medium"}
                                and parsed2.get("confidence") in {"high", "medium"})
                judge["second_opinion_agreed"] = (
                    judge.get("verdict") == parsed2.get("verdict")
                    and (not both_supports or both_non_low)
                )
                judge["input_tokens"] += int(second.get("tokens_in") or 0)
                judge["output_tokens"] += int(second.get("tokens_out") or 0)

        result = _classify_with_prebuilt_judge(
            claim_quote=claim_quote, ref_meta=ref_meta,
            resolved=resolved, judge=judge, fast=fast,
        )
        result["locator"] = locator
        final_targets.append(result)

    status, severity, summary = aggregate_status(final_targets)
    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": final_targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0,
            "tokens_in": 0, "tokens_out": 0,
            "model": os.environ.get("REVIEW_VERIFIER_MODEL", "claude-sonnet-4-6"),
            "cached": use_cache,
            "judge_backend": "batch-ingest",
            "inputs": {"n_targets": len(final_targets), "fast_mode": fast},
        },
        "errors": errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(
    claims_path: Path | None = None,
    out_path: Path | None = None,
    fast: bool = False,
    use_cache: bool = True,
    *,
    backend=None,
    claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if backend is not None and backend.mode == "batch-ingest":
        return _run_batch_ingest(out_path, backend)

    if claims is None:
        if claims_path is None:
            raise ValueError("either claims_path or claims list must be provided")
        claims = load_claims(claims_path)

    if backend is not None and backend.mode == "batch-emit":
        return _run_batch_emit(claims, out_path, fast, use_cache, backend)

    targets: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    for claim in claims:
        ref_meta = claim.get("reference", {})
        try:
            resolved = resolve_reference(ref_meta, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001
            targets.append(
                {
                    "locator": claim["locator"],
                    "status": "unverifiable",
                    "severity_suggestion": "P0",
                    "evidence": {
                        "quote": claim["claim_quote"],
                        "judge_notes": f"resolver crashed: {exc}",
                    },
                    "root_cause_key": _root_cause_key(ref_meta, "resolver-crash"),
                }
            )
            continue
        try:
            result = classify_target(
                claim["claim_quote"], ref_meta, resolved, fast=fast,
            )
        except Exception as exc:  # noqa: BLE001
            targets.append(
                {
                    "locator": claim["locator"],
                    "status": "unverifiable",
                    "severity_suggestion": "P0",
                    "evidence": {
                        "quote": claim["claim_quote"],
                        "judge_notes": (
                            f"classify crashed: {type(exc).__name__}: {exc}"
                        ),
                        "judge_confidence": "low",
                    },
                    "root_cause_key": _root_cause_key(ref_meta, "classify-crash"),
                }
            )
            continue
        result["locator"] = claim["locator"]
        targets.append(result)
        total_in += int(result.get("evidence", {}).get("judge_input_tokens", 0) or 0)
        total_out += int(result.get("evidence", {}).get("judge_output_tokens", 0) or 0)

    status, severity, summary = aggregate_status(targets)
    model = os.environ.get("REVIEW_VERIFIER_MODEL", "claude-sonnet-4-6")
    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": _estimate_cost_usd(model, total_in, total_out),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "model": model,
            "cached": use_cache,
            "inputs": {
                "claims_file": str(claims_path),
                "n_targets": len(targets),
                "fast_mode": fast,
            },
        },
        "errors": [],
    }
    if len(targets) > 100:
        report["metadata"]["cost_warning"] = (
            f"High citation count ({len(targets)}): estimated ~2 LLM calls per "
            "citation in full mode; consider --fast to halve token usage."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that cited references support the claims they are attached to.")
    parser.add_argument("--claims", type=Path,
                        help="JSONL with {locator, claim_quote, reference}. "
                             "Required for sdk / batch-emit modes.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path (VERIFIER_CONTRACT schema)")
    parser.add_argument("--fast", action="store_true", help="Skip second-opinion judge call")
    parser.add_argument("--no-cache", action="store_true", help="Disable on-disk cache")
    if _judge_backend is not None:
        _judge_backend.add_backend_args(parser)
    args = parser.parse_args()

    backend = _judge_backend.build_context(args) if _judge_backend is not None else None

    try:
        if backend is not None and backend.mode == "batch-ingest":
            report = run(out_path=args.out, backend=backend)
        else:
            if not args.claims:
                print("error: --claims is required for sdk / batch-emit modes",
                      file=sys.stderr)
                return 2
            report = run(args.claims, args.out, fast=args.fast,
                         use_cache=not args.no_cache, backend=backend)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    print(f"[{VERIFIER_ID}] {report['status'].upper()}: {report['summary']}")
    for t in report["targets"]:
        print(f"  - {t['locator']}: {t['status']} (suggested={t['severity_suggestion']})")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
