"""verify_internal_contradiction.py — LLM-grounded internal contradiction detector.

Extends scope-diff to full U10 coverage: abstract ↔ conclusion semantic
contradictions and introduction-promise ↔ results-delivery mismatches.

Algorithm:

  1. Parse LaTeX into canonical sections (reuses `parse_latex` from
     `verify_internal_consistency`).
  2. For each non-empty section, call a claim-triple extractor LLM that
     returns strict JSON triples
     `{subject, predicate, scope, quote_span}` (max 8 per section).
  3. For each relevant section pair
     {(abstract, conclusion), (abstract, methods), (intro, results)},
     run a devil's-advocate comparison LLM that emits JSON contradiction
     pairs `{claim_a, claim_b, contradiction_kind, confidence, reason}`.
     Low-confidence pairs are dropped.
  4. Targets are emitted per surviving contradiction. `direct_opposite`
     and `promise_not_delivered` are P0; other kinds major.

Graceful degradation: missing `ANTHROPIC_API_KEY` or absent `anthropic`
SDK → top-level `status=unverifiable`, severity `P0`.

Output conforms to `references/VERIFIER_CONTRACT.md`.

Usage:
    python verify_internal_contradiction.py --tex paper.tex --out workspace/verifier/contra.json
    python verify_internal_contradiction.py --sections sections.json --out contra.json --fast
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the battle-tested LaTeX section parser.
try:
    from verify_internal_consistency import (  # type: ignore
        SECTION_KEYS,
        parse_latex,
    )
except ImportError:
    # Allow running this file from any cwd by extending sys.path to its own dir.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_internal_consistency import (  # type: ignore  # noqa: E402
        SECTION_KEYS,
        parse_latex,
    )

VERIFIER_ID = "verify_internal_contradiction"
VERIFIER_VERSION = "0.4-cc-bridge"

# Shared CC-bridge backend helper.
try:
    import _judge_backend  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import _judge_backend  # type: ignore
    except ImportError:
        _judge_backend = None  # type: ignore

DEFAULT_MODEL_ENV = "REVIEW_VERIFIER_MODEL"
DEFAULT_MODEL = "claude-sonnet-4-6"

MAX_TRIPLES_PER_SECTION = 8
MAX_SECTION_CHARS = 6000  # LLM context trim; abstracts/introductions rarely exceed this

SECTION_PAIRS = [
    ("abstract", "conclusion"),
    ("abstract", "methods"),
    ("introduction", "results"),
]

P0_KINDS = {"direct_opposite", "promise_not_delivered"}
VALID_KINDS = {
    "direct_opposite",
    "scope_mismatch",
    "promise_not_delivered",
    "number_mismatch",
}
VALID_CONFIDENCE = {"high", "medium", "low"}


# ----------------------------- LLM plumbing -----------------------------


def _llm_ready() -> tuple[bool, str | None]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not configured"
    try:
        import anthropic  # type: ignore  # noqa: F401
    except ImportError:
        return False, "anthropic SDK not installed"
    return True, None


def _current_model() -> str:
    return os.environ.get(DEFAULT_MODEL_ENV, DEFAULT_MODEL)


TRIPLE_EXTRACT_SYSTEM = """You extract atomic scientific claims from a section of an academic paper.
Return ONLY a JSON array of objects of the form:
[
  {"subject": "...", "predicate": "...", "scope": "...", "quote_span": "..."},
  ...
]

Field definitions:
- subject: the object of study or quantity (e.g., "our estimator", "entropy production rate").
- predicate: what is asserted about it (e.g., "converges at rate n^{-1/2}", "outperforms baselines by 12%").
- scope: the regime, dataset, or condition under which the claim holds (e.g., "for bounded rewards", "on CIFAR-10").
- quote_span: the shortest verbatim phrase (≤180 chars) from the section that carries the claim.

Rules:
- Maximum 8 claims. Pick the most load-bearing ones.
- No claim repetition.
- Do NOT paraphrase beyond neutral summarization.
- If the section contains no actionable claims (pure motivation, references), return [].
- Output must be valid JSON; do not wrap in code fences or add commentary."""


COMPARE_SYSTEM = """You are a skeptical peer reviewer hunting for INTERNAL CONTRADICTIONS between two sections of the same paper.

Given two JSON lists of claim triples (section A and section B), argue adversarially that section B fails to deliver or actually contradicts what section A commits to. Focus only on material inconsistencies a referee would flag.

Return ONLY a JSON array of contradiction objects:
[
  {
    "claim_a": "...",                       // quote_span of the offending claim in A
    "claim_b": "...",                       // quote_span of the counter-claim in B (use "<missing>" if B silently fails to deliver)
    "contradiction_kind": "direct_opposite|scope_mismatch|promise_not_delivered|number_mismatch",
    "confidence": "high|medium|low",
    "reason": "one sentence explaining the contradiction"
  },
  ...
]

Kinds:
- direct_opposite: A says X, B says ¬X about the same subject/regime.
- scope_mismatch: A's scope is strictly broader than B's verified scope (overclaim).
- promise_not_delivered: A promises result/analysis; B simply does not contain it.
- number_mismatch: A and B give inconsistent quantitative values for the same metric/constant.

Rules:
- Return [] if there is no contradiction.
- Drop pairs where you cannot argue a specific material conflict; prefer fewer, sharper claims.
- confidence=low should only appear when you are unsure; the downstream caller may drop low-confidence pairs.
- Output valid JSON only; no prose outside the array."""


def _call_llm(system: str, user: str, max_tokens: int = 800) -> dict[str, Any]:
    """Invoke the Anthropic SDK once. Returns `{text, input_tokens, output_tokens, error}`."""
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    model = _current_model()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=30.0,
        )
        body = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )
        return {
            "text": body,
            "input_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
            "model": model,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "model": model,
            "error": str(exc),
        }


def _parse_json_array(body: str) -> tuple[list[Any] | None, str | None]:
    if not body.strip():
        return None, "empty response"
    # Strip code fences if the model ignored the instruction
    fence = re.match(r"```(?:json)?\s*(.*?)```", body.strip(), re.DOTALL)
    if fence:
        body = fence.group(1)
    # Find the outermost JSON array span
    match = re.search(r"\[.*\]", body, flags=re.DOTALL)
    if not match:
        return None, "no JSON array found"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    if not isinstance(data, list):
        return None, "root is not a list"
    return data, None


def extract_triples(section_name: str, text: str) -> dict[str, Any]:
    if not text.strip():
        return {
            "section": section_name,
            "triples": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "error": "section empty",
        }
    trimmed = text[:MAX_SECTION_CHARS]
    user = (
        f"Section name: {section_name}\n\n"
        f"---\n{trimmed}\n---\n\n"
        "Return the JSON array of claim triples now."
    )
    resp = _call_llm(TRIPLE_EXTRACT_SYSTEM, user, max_tokens=800)
    if resp["error"]:
        return {
            "section": section_name,
            "triples": [],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "error": f"LLM error: {resp['error']}",
        }
    data, err = _parse_json_array(resp["text"])
    if data is None:
        return {
            "section": section_name,
            "triples": [],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "error": f"parse error: {err}",
            "raw": resp["text"][:500],
        }
    triples = []
    for item in data[:MAX_TRIPLES_PER_SECTION]:
        if not isinstance(item, dict):
            continue
        triples.append(
            {
                "subject": str(item.get("subject", ""))[:200],
                "predicate": str(item.get("predicate", ""))[:300],
                "scope": str(item.get("scope", ""))[:200],
                "quote_span": str(item.get("quote_span", ""))[:220],
            }
        )
    return {
        "section": section_name,
        "triples": triples,
        "input_tokens": resp["input_tokens"],
        "output_tokens": resp["output_tokens"],
        "error": None,
    }


def compare_pair(
    section_a: str,
    section_b: str,
    triples_a: list[dict[str, Any]],
    triples_b: list[dict[str, Any]],
) -> dict[str, Any]:
    if not triples_a or not triples_b:
        return {
            "contradictions": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "error": "missing triples on one side",
        }
    user = (
        f"Section A = {section_a}:\n{json.dumps(triples_a, ensure_ascii=False)}\n\n"
        f"Section B = {section_b}:\n{json.dumps(triples_b, ensure_ascii=False)}\n\n"
        "Emit the contradiction JSON array now (empty array if none)."
    )
    resp = _call_llm(COMPARE_SYSTEM, user, max_tokens=900)
    if resp["error"]:
        return {
            "contradictions": [],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "error": f"LLM error: {resp['error']}",
        }
    data, err = _parse_json_array(resp["text"])
    if data is None:
        return {
            "contradictions": [],
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "error": f"parse error: {err}",
            "raw": resp["text"][:500],
        }
    contradictions: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("contradiction_kind", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()
        if kind not in VALID_KINDS:
            continue
        if confidence not in VALID_CONFIDENCE:
            confidence = "low"
        contradictions.append({
            "claim_a": str(item.get("claim_a", ""))[:300],
            "claim_b": str(item.get("claim_b", ""))[:300],
            "contradiction_kind": kind,
            "confidence": confidence,
            "reason": str(item.get("reason", ""))[:400],
        })
    return {
        "contradictions": contradictions,
        "input_tokens": resp["input_tokens"],
        "output_tokens": resp["output_tokens"],
        "error": None,
    }


# ----------------------------- Target construction -----------------------------


def build_contradiction_target(
    section_a: str,
    section_b: str,
    index: int,
    contradiction: dict[str, Any],
) -> dict[str, Any]:
    kind = contradiction["contradiction_kind"]
    severity = "P0" if kind in P0_KINDS else "major"
    locator = f"contradiction:{section_a}-{section_b}-{index}"
    evidence = {
        "quote": contradiction.get("claim_a", ""),
        "paired_quote": contradiction.get("claim_b", ""),
        "judge_notes": contradiction.get("reason", ""),
        "judge_confidence": contradiction.get("confidence", "medium"),
        "finding_kind": kind,
        "section_a": section_a,
        "section_b": section_b,
    }
    return {
        "locator": locator,
        "status": "verified",
        "severity_suggestion": severity,
        "evidence": evidence,
        "root_cause_key": f"internal-contradiction-{kind}-{section_a}-{section_b}-{index}",
    }


# ----------------------------- Cost estimate -----------------------------


def _estimate_cost_usd(model: str, total_in: int, total_out: int) -> float | None:
    rates = {
        "claude-opus-4-7": (15.0, 75.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5-20251001": (1.0, 5.0),
    }
    base = model.split("[")[0]
    if base not in rates:
        return None
    in_rate, out_rate = rates[base]
    return round((total_in / 1e6) * in_rate + (total_out / 1e6) * out_rate, 4)


# ----------------------------- Orchestration -----------------------------


def _triple_user_prompt(section_name: str, text: str) -> str:
    trimmed = text[:MAX_SECTION_CHARS]
    return (
        f"Section name: {section_name}\n\n"
        f"---\n{trimmed}\n---\n\n"
        "Return the JSON array of claim triples now."
    )


def _compare_user_prompt(a: str, b: str,
                         triples_a: list[dict[str, Any]],
                         triples_b: list[dict[str, Any]]) -> str:
    return (
        f"Section A = {a}:\n{json.dumps(triples_a, ensure_ascii=False)}\n\n"
        f"Section B = {b}:\n{json.dumps(triples_b, ensure_ascii=False)}\n\n"
        "Emit the contradiction JSON array now (empty array if none)."
    )


def _emit_triples_phase(
    sections: dict[str, str], out_path: Path, fast: bool, backend,
) -> dict[str, Any]:
    _judge_backend.ensure_emit_args(backend)
    relevant_sections: set[str] = set()
    for a, b in SECTION_PAIRS:
        if fast and (a, b) != ("abstract", "conclusion"):
            continue
        relevant_sections.update({a, b})

    tasks: list[dict[str, Any]] = []
    sections_with_text: dict[str, str] = {}
    for name in sorted(relevant_sections):
        text = sections.get(name, "")
        sections_with_text[name] = text
        if not text.strip():
            continue
        tasks.append(_judge_backend.build_task(
            task_id=f"triples:{name}",
            verifier=VERIFIER_ID,
            target_locator=f"triples:{name}",
            kind="triple_extract",
            system=TRIPLE_EXTRACT_SYSTEM,
            user=_triple_user_prompt(name, text),
            max_tokens=800, temperature=0, expected_format="json_array",
        ))
    _judge_backend.write_tasks(backend.tasks_out, tasks)

    state = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "phase": "triples",
        "sections": sections_with_text,
        "fast": fast,
        "out_path": str(out_path),
    }
    _judge_backend.write_state(backend.state_file, state)

    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": "pending_llm",
        "severity_suggestion": "minor",
        "summary": f"batch-emit triples: {len(tasks)} task(s) queued",
        "targets": [{
            "locator": "pair:pending_llm",
            "status": "pending_llm",
            "severity_suggestion": "minor",
            "evidence": {
                "quote": "",
                "judge_notes": (
                    f"awaiting LLM triple extraction across "
                    f"{len(relevant_sections)} section(s)"),
            },
            "root_cause_key": "internal-contradiction-pending-llm",
        }],
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
            "model": _current_model(), "cached": False,
            "judge_backend": "batch-emit",
            "phase": "triples",
            "n_llm_tasks_emitted": len(tasks),
            "inputs": {
                "sections_present": [k for k, v in sections.items() if v.strip()],
                "fast_mode": fast,
            },
        },
        "errors": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _ingest_triples_and_emit_compare(
    out_path: Path, backend,
) -> dict[str, Any]:
    """First-wave ingest: absorb triple-extract results, emit compare tasks.

    Exits with ``status='pending_llm'`` and writes a new tasks file so the
    orchestrator knows to dispatch a second wave.
    """
    _judge_backend.ensure_ingest_args(backend)
    state = _judge_backend.read_state(backend.state_file)
    fast = bool(state.get("fast"))
    sections = state.get("sections") or {}
    results = _judge_backend.load_results(backend.results_in)

    triple_cache: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for name in sections:
        r = results.get(f"triples:{name}")
        if r is None:
            triple_cache[name] = []
            continue
        body = r.get("body") or ""
        data, err = _parse_json_array(body if isinstance(body, str) else json.dumps(body))
        if data is None:
            errors.append(f"triple_extract[{name}]: {err}")
            triple_cache[name] = []
            continue
        triples = []
        for item in data[:MAX_TRIPLES_PER_SECTION]:
            if not isinstance(item, dict):
                continue
            triples.append({
                "subject": str(item.get("subject", ""))[:200],
                "predicate": str(item.get("predicate", ""))[:300],
                "scope": str(item.get("scope", ""))[:200],
                "quote_span": str(item.get("quote_span", ""))[:220],
            })
        triple_cache[name] = triples

    # Now emit compare tasks.
    compare_tasks: list[dict[str, Any]] = []
    for a, b in SECTION_PAIRS:
        if fast and (a, b) != ("abstract", "conclusion"):
            continue
        t_a = triple_cache.get(a, [])
        t_b = triple_cache.get(b, [])
        if not t_a or not t_b:
            continue
        compare_tasks.append(_judge_backend.build_task(
            task_id=f"compare:{a}-{b}",
            verifier=VERIFIER_ID,
            target_locator=f"pair:{a}-{b}",
            kind="compare",
            system=COMPARE_SYSTEM,
            user=_compare_user_prompt(a, b, t_a, t_b),
            max_tokens=900, temperature=0, expected_format="json_array",
        ))

    # Decide where the next-wave task file goes. Prefer explicit
    # `--judge-tasks-out`; fall back to appending a `.compare.jsonl` suffix
    # next to the results file.
    if backend.tasks_out:
        next_tasks_path = backend.tasks_out
    else:
        next_tasks_path = backend.results_in.with_suffix(".compare.jsonl")

    if compare_tasks:
        _judge_backend.write_tasks(next_tasks_path, compare_tasks)

    new_state = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "phase": "compare" if compare_tasks else "done",
        "sections": sections,
        "triples": triple_cache,
        "fast": fast,
        "out_path": str(out_path),
        "errors": errors,
        "compare_tasks_file": str(next_tasks_path) if compare_tasks else None,
    }
    _judge_backend.write_state(backend.state_file, new_state)

    if not compare_tasks:
        # No compare tasks => every pair had at least one empty side. Emit the
        # current state as a final report.
        return _finalize_from_state(out_path, new_state, {})

    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": "pending_llm",
        "severity_suggestion": "minor",
        "summary": (
            f"batch-ingest triples -> batch-emit compare: "
            f"{len(compare_tasks)} task(s) queued"),
        "targets": [{
            "locator": "pair:pending_compare",
            "status": "pending_llm",
            "severity_suggestion": "minor",
            "evidence": {
                "quote": "",
                "judge_notes": (
                    f"awaiting LLM pairwise contradiction comparison "
                    f"({len(compare_tasks)} pair(s))"),
            },
            "root_cause_key": "internal-contradiction-pending-compare",
        }],
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
            "model": _current_model(), "cached": False,
            "judge_backend": "batch-ingest",
            "phase_transition": "triples->compare",
            "n_llm_tasks_emitted": len(compare_tasks),
            "compare_tasks_file": str(next_tasks_path),
        },
        "errors": errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _finalize_from_state(
    out_path: Path, state: dict[str, Any], results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sections = state.get("sections") or {}
    triple_cache = state.get("triples") or {}
    fast = bool(state.get("fast"))

    targets: list[dict[str, Any]] = []
    pair_reports: list[dict[str, Any]] = []
    p0_count = major_count = 0
    errors: list[str] = list(state.get("errors") or [])

    for a, b in SECTION_PAIRS:
        if fast and (a, b) != ("abstract", "conclusion"):
            continue
        t_a = triple_cache.get(a, [])
        t_b = triple_cache.get(b, [])
        if not t_a or not t_b:
            empty_side = a if not t_a else b
            targets.append({
                "locator": f"pair:{a}-{b}",
                "status": "unverifiable",
                "severity_suggestion": "P0",
                "evidence": {
                    "quote": _truncate(sections.get(a, ""), 280),
                    "paired_quote": _truncate(sections.get(b, ""), 280),
                    "judge_notes": (
                        f"section {empty_side!r} produced no triples "
                        "(empty or extraction failed); cannot verify contradiction"),
                    "judge_confidence": "low",
                    "finding_kind": "section_empty",
                    "section_a": a, "section_b": b,
                    "triples_a_count": len(t_a),
                    "triples_b_count": len(t_b),
                    "unverifiable_kind": "tool",
                },
                "root_cause_key": f"internal-contradiction-section-empty-{a}-{b}",
            })
            pair_reports.append({
                "pair": f"{a}-{b}", "skipped": True,
                "reason": f"missing triples (a={len(t_a)}, b={len(t_b)})",
            })
            continue
        r = results.get(f"compare:{a}-{b}")
        if r is None:
            errors.append(f"compare[{a}-{b}]: result missing")
            continue
        body = r.get("body") or ""
        data, err = _parse_json_array(
            body if isinstance(body, str) else json.dumps(body))
        if data is None:
            errors.append(f"compare[{a}-{b}]: {err}")
            continue
        kept = []
        for c in data:
            if not isinstance(c, dict):
                continue
            kind = str(c.get("contradiction_kind", "")).strip()
            conf = str(c.get("confidence", "")).strip().lower()
            if kind not in VALID_KINDS:
                continue
            if conf not in VALID_CONFIDENCE:
                conf = "low"
            if conf == "low":
                continue
            contradiction = {
                "claim_a": str(c.get("claim_a", ""))[:300],
                "claim_b": str(c.get("claim_b", ""))[:300],
                "contradiction_kind": kind,
                "confidence": conf,
                "reason": str(c.get("reason", ""))[:400],
            }
            target = build_contradiction_target(a, b, len(kept), contradiction)
            targets.append(target)
            kept.append(target)
            if target["severity_suggestion"] == "P0":
                p0_count += 1
            elif target["severity_suggestion"] == "major":
                major_count += 1
        pair_reports.append({
            "pair": f"{a}-{b}", "skipped": False,
            "kept_after_confidence_filter": len(kept),
        })

    n_unverifiable = sum(1 for t in targets if t.get("status") == "unverifiable")

    if not targets:
        targets.append({
            "locator": "pair:overall",
            "status": "verified",
            "severity_suggestion": "minor",
            "evidence": {
                "quote": _truncate(sections.get("abstract", ""), 280),
                "paired_quote": _truncate(sections.get("conclusion", ""), 280),
                "judge_notes": "no contradictions detected",
                "judge_confidence": "medium",
                "finding_kind": "no_contradiction",
            },
            "root_cause_key": "internal-contradiction-none",
        })
        overall_status, severity, summary = ("verified", "minor",
            "no internal contradictions detected across section pairs")
    elif p0_count or major_count:
        overall_status = "failed"
        if p0_count:
            severity = "P0"
            summary = (f"{p0_count} P0 + {major_count} major contradiction(s) "
                       f"detected across {len(pair_reports)} section pair(s)")
        else:
            severity = "major"
            summary = (f"{major_count} major contradiction(s) detected "
                       f"across {len(pair_reports)} section pair(s)")
    elif n_unverifiable:
        overall_status, severity = "unverifiable", "P0"
        summary = (
            f"{n_unverifiable} section pair(s) could not be verified "
            "(empty or unparseable sections)")
    else:
        overall_status, severity = "verified", "minor"
        summary = "no internal contradictions detected across section pairs"

    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": overall_status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
            "model": _current_model(), "cached": False,
            "judge_backend": "batch-ingest",
            "phase": "final",
            "inputs": {
                "sections_present": [k for k, v in sections.items() if v.strip()],
                "fast_mode": fast,
                "pair_reports": pair_reports,
                "triples_per_section": {k: len(v) for k, v in triple_cache.items()},
            },
        },
        "errors": errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _ingest_phase(out_path: Path, backend) -> dict[str, Any]:
    state = _judge_backend.read_state(backend.state_file)
    phase = state.get("phase", "triples")
    results = _judge_backend.load_results(backend.results_in)
    if phase == "triples":
        return _ingest_triples_and_emit_compare(out_path, backend)
    # phase == "compare" or "done"
    return _finalize_from_state(out_path, state, results)


def run(
    *,
    sections: dict[str, str] | None = None,
    out_path: Path,
    fast: bool = False,
    backend=None,
) -> dict[str, Any]:
    # CC-bridge paths first — they do not require sections for ingest.
    if backend is not None and backend.mode == "batch-ingest":
        return _ingest_phase(out_path, backend)
    if sections is None:
        raise ValueError("sections required for sdk / batch-emit modes")
    if backend is not None and backend.mode == "batch-emit":
        return _emit_triples_phase(sections, out_path, fast, backend)

    ready, err = _llm_ready()
    total_in = 0
    total_out = 0
    errors: list[str] = []
    model_used = _current_model()

    if not ready:
        report = {
            "verifier_id": VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
            "status": "unverifiable",
            "severity_suggestion": "P0",
            "summary": f"LLM required for contradiction extraction; {err}",
            "targets": [{
                "locator": "pair:preflight",
                "status": "unverifiable",
                "severity_suggestion": "P0",
                "evidence": {
                    "quote": "",
                    "judge_notes": f"LLM unavailable ({err}); cannot extract contradictions.",
                    "judge_confidence": "medium",
                    "finding_kind": "llm_unavailable",
                    "unverifiable_kind": "env",
                },
                "root_cause_key": "internal-contradiction-llm-unavailable",
            }],
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cost_usd": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "model": model_used,
                "cached": False,
                "inputs": {
                    "sections_present": [k for k, v in sections.items() if v.strip()],
                    "fast_mode": fast,
                    "llm_available": False,
                },
            },
            "errors": [err or "llm unavailable"],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    # 1) Extract triples per present section.
    relevant_sections = set()
    for a, b in SECTION_PAIRS:
        if fast and (a, b) != ("abstract", "conclusion"):
            continue
        relevant_sections.update({a, b})

    triple_cache: dict[str, dict[str, Any]] = {}
    for name in relevant_sections:
        text = sections.get(name, "")
        if not text.strip():
            triple_cache[name] = {
                "section": name, "triples": [],
                "input_tokens": 0, "output_tokens": 0,
                "error": "section missing or empty",
            }
            continue
        result = extract_triples(name, text)
        total_in += result["input_tokens"]
        total_out += result["output_tokens"]
        if result.get("error"):
            errors.append(f"triple_extract[{name}]: {result['error']}")
        triple_cache[name] = result

    # 2) Compare configured pairs.
    targets: list[dict[str, Any]] = []
    p0_count = major_count = 0
    pair_reports: list[dict[str, Any]] = []

    for a, b in SECTION_PAIRS:
        if fast and (a, b) != ("abstract", "conclusion"):
            continue
        t_a = triple_cache.get(a, {}).get("triples", [])
        t_b = triple_cache.get(b, {}).get("triples", [])
        if not t_a or not t_b:
            empty_side = a if not t_a else b
            targets.append({
                "locator": f"pair:{a}-{b}",
                "status": "unverifiable",
                "severity_suggestion": "P0",
                "evidence": {
                    "quote": _truncate(sections.get(a, ""), 280),
                    "paired_quote": _truncate(sections.get(b, ""), 280),
                    "judge_notes": (
                        f"section {empty_side!r} produced no triples (empty or "
                        "extraction failed); cannot verify contradiction — paper "
                        "may be incomplete or section parse failed"
                    ),
                    "judge_confidence": "low",
                    "finding_kind": "section_empty",
                    "section_a": a,
                    "section_b": b,
                    "triples_a_count": len(t_a),
                    "triples_b_count": len(t_b),
                    # Empty section = either section parser failed (tool gap)
                    # or LLM returned [] despite input (evidence). We cannot
                    # distinguish perfectly — default to tool (human review).
                    "unverifiable_kind": "tool",
                },
                "root_cause_key": f"internal-contradiction-section-empty-{a}-{b}",
            })
            pair_reports.append({
                "pair": f"{a}-{b}",
                "skipped": True,
                "reason": f"missing triples (a={len(t_a)}, b={len(t_b)})",
            })
            continue
        comparison = compare_pair(a, b, t_a, t_b)
        total_in += comparison["input_tokens"]
        total_out += comparison["output_tokens"]
        if comparison.get("error"):
            errors.append(f"compare[{a}-{b}]: {comparison['error']}")
        contradictions = comparison["contradictions"]
        kept = []
        for idx, c in enumerate(contradictions):
            if c["confidence"] == "low":
                continue
            target = build_contradiction_target(a, b, len(kept), c)
            targets.append(target)
            kept.append(target)
            if target["severity_suggestion"] == "P0":
                p0_count += 1
            elif target["severity_suggestion"] == "major":
                major_count += 1
        pair_reports.append({
            "pair": f"{a}-{b}",
            "skipped": False,
            "raw_contradictions": len(contradictions),
            "kept_after_confidence_filter": len(kept),
        })

    n_unverifiable = sum(1 for t in targets if t.get("status") == "unverifiable")
    n_contradiction = sum(1 for t in targets if t.get("status") == "verified")

    if not targets:
        targets.append({
            "locator": "pair:overall",
            "status": "verified",
            "severity_suggestion": "minor",
            "evidence": {
                "quote": _truncate(sections.get("abstract", ""), 280),
                "paired_quote": _truncate(sections.get("conclusion", ""), 280),
                "judge_notes": (
                    "LLM-grounded triple comparison found no medium/high confidence "
                    "contradictions across the configured section pairs."
                ),
                "judge_confidence": "medium",
                "finding_kind": "no_contradiction",
            },
            "root_cause_key": "internal-contradiction-none",
        })
        overall_status = "verified"
        severity = "minor"
        summary = "no internal contradictions detected across section pairs"
    elif p0_count or major_count:
        if p0_count:
            overall_status = "failed"
            severity = "P0"
            summary = (
                f"{p0_count} P0 + {major_count} major contradiction(s) detected "
                f"across {len(pair_reports)} section pair(s)"
            )
        else:
            overall_status = "failed"
            severity = "major"
            summary = (
                f"{major_count} major contradiction(s) detected "
                f"across {len(pair_reports)} section pair(s)"
            )
    elif n_unverifiable:
        overall_status = "unverifiable"
        severity = "P0"
        summary = (
            f"{n_unverifiable} section pair(s) could not be verified "
            "(empty or unparseable sections)"
        )
    else:
        overall_status = "verified"
        severity = "minor"
        summary = "no internal contradictions detected across section pairs"

    report = {
        "verifier_id": VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
        "status": overall_status,
        "severity_suggestion": severity,
        "summary": summary,
        "targets": targets,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": _estimate_cost_usd(model_used, total_in, total_out),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "model": model_used,
            "cached": False,
            "inputs": {
                "sections_present": [k for k, v in sections.items() if v.strip()],
                "section_char_counts": {k: len(v) for k, v in sections.items()},
                "fast_mode": fast,
                "pair_reports": pair_reports,
                "triples_per_section": {
                    k: len(v.get("triples", [])) for k, v in triple_cache.items()
                },
                "llm_available": True,
            },
        },
        "errors": errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _truncate(text: str, n: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= n else clean[: n - 1] + "…"


# ----------------------------- CLI -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-grounded internal contradiction detector across canonical section pairs."
    )
    parser.add_argument("--tex", type=Path, help="LaTeX file to parse")
    parser.add_argument(
        "--sections", type=Path,
        help="Pre-parsed sections JSON with keys abstract/introduction/methods/results/conclusion",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    parser.add_argument(
        "--fast", action="store_true",
        help="Only run the abstract↔conclusion pair (skip abstract↔methods and intro↔results)",
    )
    if _judge_backend is not None:
        _judge_backend.add_backend_args(parser)
    args = parser.parse_args()

    backend = _judge_backend.build_context(args) if _judge_backend is not None else None

    if backend is not None and backend.mode == "batch-ingest":
        report = run(out_path=args.out, fast=args.fast, backend=backend)
        print(f"[{VERIFIER_ID}] {report['status'].upper()}: {report['summary']}")
        return 7 if report["status"] == "pending_llm" else (
            0 if report["status"] == "verified" else 1)

    if not args.tex and not args.sections:
        print("error: must provide --tex or --sections", file=sys.stderr)
        return 2

    if args.sections:
        data = json.loads(args.sections.read_text(encoding="utf-8"))
        sections = {k: data.get(k, "") for k in SECTION_KEYS}
    else:
        if not args.tex.is_file():
            print(f"error: tex file not found: {args.tex}", file=sys.stderr)
            return 2
        tex = args.tex.read_text(encoding="utf-8", errors="replace")
        sections = parse_latex(tex)

    report = run(sections=sections, out_path=args.out, fast=args.fast, backend=backend)
    print(f"[{VERIFIER_ID}] {report['status'].upper()}: {report['summary']}")
    for t in report["targets"]:
        print(f"  - {t['locator']}: {t['status']} (suggested={t['severity_suggestion']})")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
