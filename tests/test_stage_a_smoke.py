"""End-to-end Stage A smoke tests — exercisable with `python -m pytest` or plain
`python test_stage_a_smoke.py` (no pytest dependency required).

Covers:
- verify_internal_consistency: detects abstract/methods scope mismatch.
- verify_internal_consistency: passes a paper whose abstract is consistent with methods.
- verify_physics_identity: flags wrong Jarzynski sign.
- verify_physics_identity: flags Itô/Stratonovich conflation without conversion.
- consolidate_review_findings: upgrades an `unverifiable` verifier target to gate_blocker.
- consolidate_review_findings: upgrades a `failed` verifier target to gate_blocker even
  when severity_suggestion is `moderate`.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path.home() / ".claude" / "skills"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


internal_consistency = _load(
    "internal_consistency",
    ROOT / "review-verification-harness" / "scripts" / "verify_internal_consistency.py",
)
physics_identity = _load(
    "physics_identity",
    ROOT / "aps-verification-harness" / "scripts" / "verify_physics_identity.py",
)
consolidator = _load(
    "consolidator",
    ROOT / "paper-audit" / "scripts" / "consolidate_review_findings.py",
)


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_scope_mismatch_detected() -> None:
    sections = {
        "abstract": (
            "We prove that for arbitrary continuous dynamics, entropy production "
            "is universally bounded. The result holds without any assumption on "
            "the underlying process and applies in general."
        ),
        "introduction": "",
        "methods": (
            "We restrict to block-diagonal Ornstein-Uhlenbeck dynamics with Gaussian "
            "noise and linearized drift in the steady state. Under the assumption of "
            "small noise and weakly coupled degrees of freedom, we derive the bound."
        ),
        "results": "",
        "conclusion": "",
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "scope.json"
        report = internal_consistency.run(sections=sections, out_path=out)
    assert_true(
        report["status"] == "failed",
        f"expected failed on scope mismatch, got {report['status']}",
    )
    kinds = {t["evidence"]["finding_kind"] for t in report["targets"]}
    assert_true(
        "universality_asymmetry" in kinds,
        f"expected universality_asymmetry, got {kinds}",
    )


def test_scope_consistent_passes() -> None:
    sections = {
        "abstract": (
            "We introduce a block-diagonal Gaussian Ornstein-Uhlenbeck framework and "
            "derive entropy production under the small-noise limit."
        ),
        "introduction": "",
        "methods": (
            "The framework is block-diagonal with Gaussian noise and Ornstein-Uhlenbeck "
            "dynamics. We work in the small-noise regime."
        ),
        "results": "",
        "conclusion": "",
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "scope.json"
        report = internal_consistency.run(sections=sections, out_path=out)
    assert_true(
        report["status"] == "verified",
        f"expected verified on consistent paper, got {report['status']}; "
        f"targets={report['targets']}",
    )


def test_jarzynski_wrong_sign_flagged() -> None:
    tex = (
        "We apply the Jarzynski equality to compute the free energy. "
        "The result follows from \\exp(+\\beta W) = \\exp(-\\beta \\Delta F)."
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "phys.json"
        report = physics_identity.run(tex, out)
    kinds = {t["evidence"]["finding_kind"] for t in report["targets"]}
    assert_true(
        "jarzynski_sign" in kinds,
        f"expected jarzynski_sign finding, got {kinds}",
    )
    assert_true(
        report["status"] == "failed",
        f"expected failed, got {report['status']}",
    )


def test_ito_strat_conflation_flagged() -> None:
    tex = (
        "We model the system with the Ito SDE "
        "dx = a(x) dt + b(x) dW, and separately in Stratonovich "
        "form dx = a(x) dt + b(x) \\circ dW. The entropy production follows."
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "phys.json"
        report = physics_identity.run(tex, out)
    kinds = {t["evidence"]["finding_kind"] for t in report["targets"]}
    assert_true(
        "ito_strat_conflation" in kinds,
        f"expected ito_strat_conflation finding, got {kinds}",
    )


def test_consolidator_upgrades_unverifiable_to_p0() -> None:
    with tempfile.TemporaryDirectory() as td:
        review = Path(td) / "review"
        (review / "comments").mkdir(parents=True)
        (review / "verifier").mkdir(parents=True)

        # Pre-existing LLM finding (moderate, not gate_blocker)
        (review / "comments" / "llm_finding.json").write_text(
            json.dumps(
                [
                    {
                        "title": "Soft wording concern",
                        "quote": "We show rigorously...",
                        "explanation": "Could be clearer.",
                        "comment_type": "presentation",
                        "severity": "moderate",
                        "source_kind": "llm",
                    }
                ]
            )
        )

        # Verifier output with one unverifiable target (severity P0)
        (review / "verifier" / "citations.json").write_text(
            json.dumps(
                {
                    "verifier_id": "verify_citations_full",
                    "verifier_version": "0.1",
                    "status": "unverifiable",
                    "severity_suggestion": "P0",
                    "summary": "1 unverifiable citation",
                    "targets": [
                        {
                            "locator": "cite:ghost2024",
                            "status": "unverifiable",
                            "severity_suggestion": "P0",
                            "evidence": {
                                "quote": "As shown by Ghost et al. (2024)...",
                                "judge_notes": "DOI unresolvable; likely fabricated.",
                                "judge_confidence": "high",
                            },
                            "root_cause_key": "citation-unresolvable-ghost2024",
                        }
                    ],
                    "metadata": {},
                }
            )
        )

        findings = consolidator.load_comment_files(review / "comments")
        findings.extend(consolidator.load_verifier_files(review / "verifier"))
        consolidated = consolidator.consolidate_findings(findings)

    ghosts = [
        f
        for f in consolidated
        if f.get("verifier_id") == "verify_citations_full"
        and f.get("verifier_status") == "unverifiable"
    ]
    assert_true(
        len(ghosts) == 1,
        f"expected exactly 1 verifier-sourced ghost citation finding, got {len(ghosts)}: {consolidated}",
    )
    assert_true(
        ghosts[0]["gate_blocker"] is True,
        f"unverifiable verifier target must upgrade to gate_blocker, got {ghosts[0]}",
    )
    assert_true(
        ghosts[0]["severity"] == "major",
        f"expected severity=major after P0 mapping, got {ghosts[0]['severity']}",
    )


def test_consolidator_upgrades_failed_moderate_to_gate_blocker() -> None:
    with tempfile.TemporaryDirectory() as td:
        review = Path(td) / "review"
        (review / "comments").mkdir(parents=True)
        (review / "verifier").mkdir(parents=True)

        (review / "verifier" / "scope.json").write_text(
            json.dumps(
                {
                    "verifier_id": "verify_internal_consistency",
                    "verifier_version": "0.1",
                    "status": "failed",
                    "severity_suggestion": "major",
                    "summary": "abstract overclaims universality",
                    "targets": [
                        {
                            "locator": "pair:abstract-methods",
                            "status": "failed",
                            "severity_suggestion": "moderate",
                            "evidence": {
                                "quote": "We prove for arbitrary processes...",
                                "judge_notes": "Abstract is universal; methods restrict.",
                                "judge_confidence": "high",
                            },
                            "root_cause_key": "internal-consistency-universality_asymmetry",
                        }
                    ],
                    "metadata": {},
                }
            )
        )

        findings = consolidator.load_verifier_files(review / "verifier")
        consolidated = consolidator.consolidate_findings(findings)

    assert_true(len(consolidated) == 1, f"expected 1 finding, got {consolidated}")
    f = consolidated[0]
    assert_true(
        f["gate_blocker"] is True,
        f"failed verifier target must be gate_blocker even with moderate suggestion: {f}",
    )


def test_consolidator_drops_verified_minor() -> None:
    with tempfile.TemporaryDirectory() as td:
        review = Path(td) / "review"
        (review / "verifier").mkdir(parents=True)
        (review / "verifier" / "clean.json").write_text(
            json.dumps(
                {
                    "verifier_id": "verify_physics_identity",
                    "verifier_version": "0.1",
                    "status": "verified",
                    "severity_suggestion": "minor",
                    "summary": "nothing to flag",
                    "targets": [
                        {
                            "locator": "physics_identity_overall",
                            "status": "verified",
                            "severity_suggestion": "minor",
                            "evidence": {
                                "quote": "",
                                "judge_notes": "clean",
                                "judge_confidence": "medium",
                            },
                            "root_cause_key": "physics-clean",
                        }
                    ],
                    "metadata": {},
                }
            )
        )
        findings = consolidator.load_verifier_files(review / "verifier")
    assert_true(
        findings == [],
        f"verified-minor informational findings should not clutter final bundle, got {findings}",
    )


def main() -> int:
    tests = [
        test_scope_mismatch_detected,
        test_scope_consistent_passes,
        test_jarzynski_wrong_sign_flagged,
        test_ito_strat_conflation_flagged,
        test_consolidator_upgrades_unverifiable_to_p0,
        test_consolidator_upgrades_failed_moderate_to_gate_blocker,
        test_consolidator_drops_verified_minor,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
