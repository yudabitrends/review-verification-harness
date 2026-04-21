# Unfixable Error Taxonomy

15 classes of errors that, if they survive AI review and reach human peer review, are **highly likely to cause rejection** and are **infeasible to fix in camera-ready**. Each class lists precise trigger definition, verifier responsible, gate severity, and handling rule.

Scope A = Stage A (this release: 2 verifiers). B / C = roadmap. A class marked "Stage A (partial)" means the current verifier catches a subset.

---

## U1. Silent algebraic / sign error in derivation

**Trigger**: A stated equality `LHS = RHS` is textually coherent but algebraically wrong.
**Verifier**: `verify_math_sympy.py` (Stage B). Stage A has no coverage; reviewer LLM is the only filter.
**Severity**: P0 if the incorrect equation is load-bearing for a claimed theorem; major otherwise.
**Handling**: Stage B — sympy substitute + numerical check. FATAL_HALT if verified failure.

## U2. Fake or misattributed citation

**Trigger**: A citation is non-existent, OR the cited paper does not make the claim being attributed to it.
**Verifier**: `verify_citations_full.py` (**Stage A ✓**).
**Severity**: P0.
**Handling**: DOI unresolvable → status=unverifiable → upgrade P0. Abstract contradicts attribution → status=verified+failed → P0 FATAL_HALT.

## U3. Data leakage (train → test contamination)

**Trigger**: Training set overlaps with test set, OR test set was used to tune hyperparameters, OR pretrained model was trained on downstream benchmark.
**Verifier**: `verify_reproducibility.py` (Stage C) + manual code review.
**Severity**: P0.
**Handling**: No Stage-A coverage; Stage C clones repo and traces data flow. Narrative-only reviewer cannot catch.

## U4. Wrong baseline / unfair comparison

**Trigger**: Baseline is an outdated version, uses different data preprocessing, or has pathologically bad hyperparameters.
**Verifier**: LLM methodology reviewer + `verify_reproducibility.py` (Stage C).
**Severity**: major → P0 if claim is "SOTA".
**Handling**: Stage A relies on existing `academic-paper-reviewer` methodology_reviewer_agent.

## U5. Scope overclaim (abstract universe > methods universe)

**Trigger**: Abstract asserts universal applicability while methods / proofs only cover a restricted case.
**Verifier**: `verify_internal_consistency.py` (**Stage A ✓**).
**Severity**: P0 if gap is "universal" vs "specific-subclass"; major otherwise.
**Handling**: Scope diff score > threshold → P0. Emit exact claim excerpts from abstract and methods.

## U6. Figure misleading (y-axis truncation, axis inconsistency)

**Trigger**: Y-axis range starts above zero for a ratio-scale metric; inconsistent axes across comparison figures; log/linear switched without annotation.
**Verifier**: `verify_figure_integrity.py` (Stage B).
**Severity**: major (camera-ready can sometimes fix by redrawing; but if the exaggeration is load-bearing for the claim, upgrades to P0).
**Handling**: Stage A has no coverage.

## U7. Ethics / IRB / dual-use omission

**Trigger**: Human-subject study without IRB statement; animal study without ethics approval; dual-use research without impact statement.
**Verifier**: Structured checklist + human attestation (Stage C).
**Severity**: P0.
**Handling**: Stage A has no coverage beyond `paper-audit` heuristic.

## U8. Statistical assumption violation

**Trigger**: t-test on non-normal data, ANOVA without homogeneity check, unreported multiple-comparison correction, power undersized.
**Verifier**: `verify_stats_assumptions.py` (Stage B) + `statistical-analysis` skill.
**Severity**: major → P0 if the violated assumption drives the conclusion.
**Handling**: Stage A covers this only via existing `paper-audit` Taxonomy #10.

## U9. Reproducibility impossibility

**Trigger**: Code not released, OR released but does not run, OR runs but doesn't reproduce headline numbers.
**Verifier**: `verify_reproducibility.py` (Stage C).
**Severity**: major → P0 if paper claims "code released" and the claim is false.
**Handling**: Stage A can only check URL syntactically (not invoked in this release).

## U10. Internal contradiction (abstract ≠ conclusion, intro ≠ results)

**Trigger**: Abstract says X, conclusion says not-X. Intro promises to show Y; results section doesn't address Y.
**Verifier**: `verify_internal_consistency.py` (**Stage A ✓**, partial — covers scope diff; full contradiction detection pending).
**Severity**: P0.
**Handling**: Stage A covers scope diffs; adjacent section contradictions pending Stage B.

## U11. Novelty miss (prior work unacknowledged)

**Trigger**: Paper claims "first to do X" but prior work (often non-English, pre-2020, or in adjacent field) already did X.
**Verifier**: `verify_novelty_broad.py` (Stage C). Stage A: `convergent-peer-review` Stage 2 partial coverage (English + post-2020 only).
**Severity**: P0.
**Handling**: Stage A relies on existing `convergent-peer-review` novelty grounding.

## U12. Self-citation / citation-ring bias

**Trigger**: >30% of cited works are by the author or a small collaborator ring; key independent work omitted.
**Verifier**: `citation-management` + manual audit.
**Severity**: major.
**Handling**: Stage A has limited coverage via `citation-management` existing tools.

## U13. P-hacking / garden of forking paths

**Trigger**: Multiple statistical tests reported without pre-registration or correction; "exploratory" framing of confirmatory claims.
**Verifier**: `verify_stats_assumptions.py` (Stage B).
**Severity**: major → P0 if headline claim rests on one lucky p < 0.05.
**Handling**: Stage A no coverage.

## U14. Cherry-picked seeds / best-run reporting

**Trigger**: Reported metric is max over seeds, not mean; seed count < 3; high variance hidden.
**Verifier**: `verify_reproducibility.py` (Stage C) + `ai-validation-blindspots` Class 9.
**Severity**: major.
**Handling**: Stage A no coverage.

## U15. Dual-use / harm disclosure missing

**Trigger**: Research with clear dual-use risk (bio, cyber, chemistry) lacks explicit harm mitigation section.
**Verifier**: Checklist + human attestation (Stage C).
**Severity**: P0 for venues requiring impact statements.
**Handling**: Stage A no coverage.

---

## Stage A + B coverage matrix

| Class | Covered | Verifier | Stage |
|---|---|---|---|
| U1 Silent math/sign | ✓ | verify_math_sympy | B |
| U2 Fake citation | ✓ | verify_citations_full | A |
| U5 Scope overclaim | ✓ | verify_internal_consistency | A |
| U10 Internal contradiction | ✓ | verify_internal_contradiction + verify_internal_consistency | B (full) + A (scope subset) |
| Round-regression (meta) | ✓ | verify_round_regression | B |
| U3 Data leakage | partial | ai-validation-blindspots Class 9.4 | via R4 adversarial probe |
| U4 Weak baseline | partial | ai-validation-blindspots Class 9.2 | via R4 adversarial probe |
| U6 Figure misleading | partial | ai-validation-blindspots Class 11 | via R4 adversarial probe |
| U7 Ethics / IRB | partial | paper-audit heuristic | via R1 review |
| U8 Stats assumption | partial | ai-validation-blindspots Class 9.3, 9.5 | via R4 adversarial probe |
| U9 Reproducibility | partial | ai-validation-blindspots Class 10 | via R4 adversarial probe |
| U11 Novelty miss | partial | convergent-peer-review | via R1 review |
| U12 Self-citation ring | partial | citation-management | via R1 review |
| U13 P-hacking | partial | ai-validation-blindspots Class 9.3 | via R4 adversarial probe |
| U14 Cherry-picked seeds | partial | ai-validation-blindspots Class 9.1, 9.7 | via R4 adversarial probe |
| U15 Dual-use disclosure | partial | paper-audit checklist | via R1 review |

Stage B goal: block ≥4 canonical counterexample papers (fake citation, scope overclaim, silent math error, abstract↔conclusion contradiction) from reaching STOP_ACCEPT, and catch round-over-round regressions introduced by the fix phase.

Stage C roadmap (not yet implemented): upgrade the "partial via R4 probe" rows to script-evidence verifiers — `verify_reproducibility`, `verify_novelty_broad`, `verify_figure_integrity`, `verify_stats_assumptions`.

See `PIPELINE_5ROUND.md` for how these plug into the ≤5-round submission gate.

---

## Mapping to paper-audit Taxonomy

Stage A extensions to `paper-audit/references/DEEP_REVIEW_CRITERIA.md`:

| New Taxonomy | Covers Unfixable Class | Verifier |
|---|---|---|
| #17 Citation content match | U2, U12 | verify_citations_full |
| #18 Internal consistency / scope | U5, U10 | verify_internal_consistency |

Future (Stage B/C):
- #19 Statistical assumption → U8, U13
- #20 Figure integrity → U6
- #21 Reproducibility execution → U9, U14
- #22 Novelty broad scan → U11
- #23 Ethics / IRB gate → U7, U15
