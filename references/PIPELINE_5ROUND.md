# Five-Round AI Review Pipeline

The ≤5-round protocol combining `review-verification-harness` (script evidence) + `ai-validation-blindspots` (adversarial probes) + `paper-audit` (review orchestrator) so that a paper that passes does so with acceptable rebuttal risk.

**Design objective**: by R5 the paper should have no P0 findings that would make rebuttal impossible, and high coverage against the 15 unfixable-error classes in `ERROR_TAXONOMY_UNFIXABLE.md`.

**Hard stop**: if R5 still shows P0 → paper is not ready to submit. A 6th round will not solve the underlying problem.

---

## Workspace conventions

Canonical per-round layout:

```
workspace/r<N>/
├── tex/
│   └── paper.tex          # snapshot of manuscript at start of round N
├── claims/                # from extract_claims.py (preprocessor)
│   ├── citation_claims.jsonl
│   ├── numerical_claims.jsonl
│   ├── scope_claims.jsonl
│   └── extract_summary.json
├── comments/              # from paper-audit deep review
│   └── *.json             # one per reviewer lane
├── verifier/              # from the 4 content verifiers
│   ├── citations.json
│   ├── scope.json
│   ├── contradiction.json
│   ├── math.json
│   └── (regression.json in R3+ only)
├── fix_log.json           # user-maintained; populated between R2 and R3
└── consolidated.json      # from consolidate_review_findings.py
```

Source for each directory:

| Directory | Populated by |
|---|---|
| `tex/` | user's manuscript snapshot (copied in; never edited in-round) |
| `claims/` | `extract_claims.py` (preprocessor; deterministic, no LLM) |
| `comments/` | `paper-audit` deep review (LLM reviewer lanes) |
| `verifier/` | the 4 content verifiers (`verify_citations_full`, `verify_internal_consistency`, `verify_internal_contradiction`, `verify_math_sympy`) + `verify_round_regression` in R3+ |
| `fix_log.json` | **user-maintained between R2 and R3** (see fix_log schema below) |
| `consolidated.json` | `paper-audit/scripts/consolidate_review_findings.py` merges all three of comments/ + verifier/ into the final issue bundle |

**One-line rule**: always create a fresh `workspace/rN/` per round, never overwrite. The consolidator globs `verifier/*.json` and will mix reports across rounds if asked to re-read a dirty workspace.

---

## fix_log.json schema

Between R2 edits and R3 regression sweep, the user must record one entry per P0/major finding from R1 that they acted on (or declined to act on). Without this log, R3 cannot tell a legitimate scope reduction from deletion-evasion.

```json
[
  {
    "finding_key": "<root_cause_key from prior round verifier target>",
    "disposition": "fixed | rejected | dropped",
    "rationale": "<short human explanation>",
    "touched_file": "<optional path>",
    "touched_span": "<optional line range>"
  }
]
```

Disposition semantics:

- **fixed** — the underlying defect was repaired (citation replaced with a real one; equation sign corrected; scope claim narrowed). R3 expects the corresponding claim to still exist in the manuscript with the defect removed.
- **rejected** — we believe the finding is a false positive (verifier was wrong) and are not acting. R3 keeps the finding visible but does not flag it as regression.
- **dropped** — we removed the claim entirely from the manuscript. **This disposition is what prevents `verify_round_regression` from flagging the removal as deletion-evasion.** Use when a P0 claim cannot be fixed and must be cut rather than repaired.

`finding_key` must match the original round's `root_cause_key` (see the consolidator's `original_root_cause_key` field for regression-verifier-derived findings — the consolidator exposes the original key precisely so fix_log matching works).

---

## Round overview

| Round | Skill | Output | Gate |
|---|---|---|---|
| R1 | `extract_claims` + all harness verifiers + `paper-audit` deep review | initial `findings.json` | — |
| R2 | `paper-audit` fixes P0/major + targeted diff-mode verifier re-run | `paper_v2.tex` + updated verifier JSONs | — |
| R3 | `verify_round_regression` + harness verifiers on diff | regression report | no new P0 vs R1 |
| R4 | `ai-validation-blindspots` hostile probes | `threat_report.json` | no catastrophic verdict |
| R5 | camera-ready gate — all verifiers green; every `unverifiable` has human override | `SUBMISSION_READY.md` | zero P0 |

Rounds can loop — if R3 turns up a regression, restart at R2 for the affected findings. **Total fresh runs capped at 5**; loops on the same content don't count as a new round but must converge in ≤2 loops or the paper is not ready.

---

## R1 — Extract + broad scan

**Forward-reference chain** — R1 produces three input streams that the consolidator then merges:

1. `paper-audit` deep review → `workspace/r1/comments/*.json` (LLM reviewer lanes)
2. `extract_claims.py` preprocessor → `workspace/r1/claims/*.jsonl` (citation / numerical / scope claim records)
3. The 4 content verifiers (`verify_citations_full`, `verify_internal_consistency`, `verify_internal_contradiction`, `verify_math_sympy`) → `workspace/r1/verifier/*.json`

The consolidator `paper-audit/scripts/consolidate_review_findings.py` reads all three and merges into `workspace/r1/consolidated.json`. It auto-upgrades any verifier `unverifiable`/`failed` target to `gate_blocker`, and silently drops `extract_claims` `verified` targets (they are preprocessor output, not paper defects).

```bash
# 1. Preprocess: automatic claim extraction (feeds verify_citations_full)
python ~/.claude/skills/review-verification-harness/scripts/extract_claims.py \
  --tex paper.tex --bib paper.bib --out-dir workspace/r1/claims/

# 2. Deep review (paper-audit primary)
# Uses paper-audit skill — kicks off reviewer agents + collects LLM findings
# Output: workspace/r1/comments/*.json

# 3. All harness verifiers in parallel
python ~/.claude/skills/review-verification-harness/scripts/verify_citations_full.py \
  --claims workspace/r1/claims/citation_claims.jsonl \
  --out workspace/r1/verifier/citations.json &

python ~/.claude/skills/review-verification-harness/scripts/verify_internal_consistency.py \
  --tex paper.tex --out workspace/r1/verifier/scope.json &

python ~/.claude/skills/review-verification-harness/scripts/verify_internal_contradiction.py \
  --tex paper.tex --out workspace/r1/verifier/contradiction.json &

python ~/.claude/skills/review-verification-harness/scripts/verify_math_sympy.py \
  --tex paper.tex --out workspace/r1/verifier/math.json &

wait

# 4. Consolidate comments/ + claims/ + verifier/ → consolidated.json
python ~/.claude/skills/paper-audit/scripts/consolidate_review_findings.py \
  workspace/r1/ > workspace/r1/consolidated.json
```

Alternatively, one command runs steps 1 + 3 (preprocessor + 4 verifiers) with a fresh workspace:

```bash
bash ~/.claude/skills/review-verification-harness/scripts/run_round.sh \
  1 paper.tex --bib paper.bib --workspace workspace --fast
```

Exit R1 when: `consolidated.json` is written and lists the P0 / major / minor findings grouped for triage.

---

## R2 — Targeted fix

For each P0 / major finding the user or `paper-audit` skill applies fixes. After each batch of edits, re-run the verifiers whose targets changed (diff-mode: feed only touched sections / equations / citations), not the whole suite.

Critical: every fix must be accompanied by an entry in `workspace/r2/fix_log.json`:
```json
[{"finding_key": "citation-unresolvable-ghost2024", "disposition": "fixed|rejected|dropped", "rationale": "removed citation after confirming fabricated; no dependent claim remained"}]
```
Without this log, R3 cannot tell deletion-evasion from legitimate scope reduction.

---

## R3 — Regression sweep ⚠ **mandatory**

```bash
python ~/.claude/skills/review-verification-harness/scripts/verify_round_regression.py \
  --prev-tex paper.tex \
  --curr-tex paper_v2.tex \
  --prev-verifier-dir workspace/r1/verifier/ \
  --curr-verifier-dir workspace/r2/verifier/ \
  --prev-claims workspace/r1/claims/ \
  --curr-claims workspace/r2/claims/ \
  --fix-log workspace/r2/fix_log.json \
  --out workspace/r3/regression.json
```

R3 catches:
- Previously-verified target now failed (R2 fix broke something else)
- Numeric/quantifier silently changed without fix-log entry
- P0 claim removed rather than fixed (deletion-evasion)
- New citations or universal-scope phrases introduced without verifier coverage

If R3 status is `failed` or `unverifiable` → loop back to R2 for those targets. Max 2 loops; otherwise abort.

---

## R4 — Adversarial probes

Invoke `ai-validation-blindspots` skill (full Phase B + B' + optional D cross-model diff):

```bash
# From the paper-audit workspace or directly:
# Skill invocation inside Claude Code: /ai-validation-blindspots paper_v2.tex --venue PRL --upstream sympy,multi-round,novelty
```

The 15 hostile agents (cold_reader, verb_auditor, strong_word_auditor, hostile_persona×N, empirical_rigor_auditor, reproducibility_auditor, figure_integrity_auditor, abstract_only_probe, methods_only_probe, figures_only_probe, etc.) feed finds into `threat_report.json`. Verifier evidence from R1–R3 can be attached as ammunition in the `hostile_persona` agent prompt.

Gate: `threat_report.summary.verdict` must be `proceed` or `proceed_with_minor_fixes`. Any `catastrophic`/`reject_risk` → cycle back to R2.

---

## R5 — Camera-ready gate

Final check before submission:
1. All verifier outputs under `workspace/r4/verifier/` have top-level `status=verified`.
2. Any `unverifiable` target in any verifier JSON has a corresponding entry in `workspace/r5/human_overrides.json` with `rationale` explaining why the user accepts the residual risk.
3. `ai-validation-blindspots` Phase C human checklist items are marked done (not auto-done).
4. Write `SUBMISSION_READY.md` aggregating:
   - Coverage matrix (which of the 15 unfixable classes were checked)
   - Total cost USD across all LLM-invoking verifiers
   - Residual risks accepted + rationale

**Submission green-light rule**: if step 1–3 pass AND no P0 findings outstanding → ready to submit. Otherwise STOP — either do R6 (which is implicitly the same as R2) or acknowledge the paper isn't ready.

---

## Anti-patterns

- **Running the same round twice without change** → waste; if no new information, the round can't improve the finding set.
- **Skipping R3** → you will ship regression bugs. This is the single most load-bearing round.
- **Letting `unverifiable` findings pass** without explicit override → violates harness design; downstream readers (reviewers) will flag what the harness flagged.
- **Running R4 before R2 is complete** → Phase D cross-model diff signal is diluted when the paper still has basic P0 issues.
- **Treating `threat_report` as a TODO list** — it's a threat map; some threats are accepted risk. Only P0-equivalent items require action.
- **More than 2 R2↔R3 loops on the same finding** — indicates the issue is not fixable by narrow edit. Escalate: rescope the paper's claims.

---

## Coverage matrix

| Unfixable class (see ERROR_TAXONOMY_UNFIXABLE.md) | Covered by |
|---|---|
| U1 silent math/sign | `verify_math_sympy` (Stage B) |
| U2 fake citation | `verify_citations_full` (Stage A) |
| U3 data leakage | `ai-validation-blindspots` Class 9.4 (R4) |
| U4 weak baseline | `ai-validation-blindspots` Class 9.2 (R4) |
| U5 scope overclaim | `verify_internal_consistency` (Stage A) |
| U6 figure misleading | `ai-validation-blindspots` Class 11 (R4) |
| U7 ethics / IRB | `paper-audit` heuristic (R1) |
| U8 stats assumption | `ai-validation-blindspots` Class 9.3, 9.5 (R4) |
| U9 reproducibility | `ai-validation-blindspots` Class 10 (R4) |
| U10 internal contradiction | `verify_internal_contradiction` (Stage B) |
| U11 novelty miss | `convergent-peer-review` (R1) |
| U12 self-citation ring | `citation-management` (R1, partial) |
| U13 p-hacking | `ai-validation-blindspots` Class 9.3 (R4) |
| U14 cherry-picked seeds | `ai-validation-blindspots` Class 9.1, 9.7 (R4) |
| U15 dual-use disclosure | `paper-audit` checklist (R1) |

Stage B verifiers (`verify_math_sympy`, `verify_internal_contradiction`, `verify_round_regression`) close the biggest previous gaps. Stage C (`verify_reproducibility`, `verify_novelty_broad`, `verify_figure_integrity`, `verify_stats_assumptions`) would move more classes from agent-narrative to script-evidence, but Stage A+B + ai-validation-blindspots is already sufficient for a ≤5-round submission gate.
