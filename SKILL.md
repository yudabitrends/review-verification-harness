---
name: review-verification-harness
description: Script-backed verification primitives for academic paper review. Converts "LLM-thinks-it's-right" review comments into evidence-grounded verdicts by resolving DOIs, fetching abstracts, running sympy on equations, cross-section contradiction detection, and round-over-round regression checking. Invoked by `paper-audit`, `convergent-peer-review`, and `academic-pipeline` to back up CRITICAL findings with machine-verifiable evidence. Pairs with `ai-validation-blindspots` (R4 adversarial probes) to form the ≤5-round submission gate.
type: verification-harness
argument-hint: "[<paper.tex>] [--bib <paper.bib>] [--workspace <dir>] [--fast]"
version: 0.4-cc-bridge
---

# review-verification-harness

## Why

The review pipeline historically treats reviewer opinions as ground truth. A CRITICAL finding from a reviewer ("this citation doesn't support the claim", "this equation has a sign error", "abstract overclaims") is trusted, not verified. This harness converts those judgments into script-backed evidence, and distinguishes three verdict states:

- **verified** — a script or API call confirms the finding is real
- **unverifiable** — evidence-gathering was inconclusive
- **failed** — the claimed finding was disproven

**Design rule:** `unverifiable` is never equivalent to `pass`. If a verifier cannot check, the paper is not ready to ship — BUT since v0.3-stageB2 every `unverifiable` target carries an `evidence.unverifiable_kind` subtype:

- `env` — host misconfiguration (missing API key, unreachable network, missing dep). Consumer routes to **`setup_needed`**, not `gate_blocker`. Fix the host, re-run.
- `tool` — the verifier's tool doesn't cover this target (sympy parser gap, custom macro, regex limit). Consumer routes to **`human_review_recommended`**, not `gate_blocker`. A human should eyeball the target.
- `evidence` — we ran the check and the evidence was genuinely inconclusive (DOI resolved but abstract empty, LLM judge low-confidence, seeds produced NaN). Consumer routes to **`gate_blocker`** — this is the original P0 upgrade case.

This taxonomy fixes the false-P0 drowning failure mode in the ESORICS run: 80 of 81 unverifiable targets were `env`/`tool`; only 1 was genuine `evidence`.

## Stage A + B verifiers (this release)

Eight scripts under `scripts/`, each consuming a manuscript (or pre-extracted claims) and emitting JSON per `references/VERIFIER_CONTRACT.md`.

### Preprocessor
- **`extract_claims.py`** — automated claim extraction. LaTeX + BibTeX → `citation_claims.jsonl` + `numerical_claims.jsonl` + `scope_claims.jsonl`. Deterministic, no LLM. Removes the "we only verify what the human remembered" failure mode.

### Content verifiers
- **`verify_citations_full.py`** *(Stage A)* — DOI/arXiv/Semantic-Scholar resolution → adversarial two-pass LLM judge → flags fake or misattributed citations (class **U2**). Strict author-surname cross-check; second judge explicitly attacks the attribution.
- **`verify_internal_consistency.py`** *(Stage A)* — regex-based abstract↔methods scope-diff; flags universality asymmetry + restriction leakage (class **U5**).
- **`verify_internal_contradiction.py`** *(Stage B)* — LLM-grounded `(subject, predicate, scope)` triple extraction from abstract/intro/methods/results/conclusion; pairwise contradiction detection with devil's-advocate second pass (class **U10** full).
- **`verify_math_sympy.py`** *(Stage B)* — parses displayed equations, runs symbolic + 3-seed numerical check of `LHS = RHS`; sign-flip heuristic (class **U1**). Catches silent algebra errors LLM reviewers miss.

### Round-safety verifier
- **`verify_round_regression.py`** *(Stage B)* — compares round N-1 vs round N. Catches:
  - Previously-verified targets now failed (R2 fix broke another claim)
  - Silent numeric/quantifier drift in claim text without fix-log entry
  - Deletion-evasion — P0 claim removed instead of fixed
  - New citations / universal quantifiers introduced without verifier coverage

## Output contract

Every verifier emits JSON conforming to `references/VERIFIER_CONTRACT.md`. Downstream consumer `paper-audit/scripts/consolidate_review_findings.py` reads the JSONs and upgrades `unverifiable` findings to `gate_blocker`.

## How to invoke

**Recommended entry point:** use the orchestrator script. It runs preprocessor + all 4 content verifiers with a fresh per-round workspace and a tidy summary. Regression verification between rounds is a separate invocation (see below).

```bash
# One-shot: run round R1 end-to-end
bash ~/.claude/skills/review-verification-harness/scripts/run_round.sh \
  1 paper.tex --bib paper.bib --workspace workspace --fast
```

`bash run_round.sh --help` prints full usage. Individual scripts below remain available for custom workflows.

## Using Claude Max (no API key)

v0.4-cc-bridge adds an orchestrator that runs the pipeline inside a Claude Code session without `ANTHROPIC_API_KEY`. LLM judgments are deferred to subagents dispatched by the session itself (Claude Max subscription).

Two-step flow:

```bash
# 1) Emit phase — runs preflight + extract_claims + scope + all 3 LLM verifiers
#    in `--judge-backend batch-emit` mode. Exits 7 if any LLM tasks pending.
python3 ~/.claude/skills/review-verification-harness/scripts/cc_run_round.py \
    paper.tex --bib paper.bib --workspace workspace --round 1 [--fast]

# 2) A Claude Code session reads scripts/cc_dispatch_template.md and dispatches
#    subagents (≤10 tasks per batch) to produce
#    workspace/r1/judge_results/<verifier>.results.jsonl.

# 3) Finalize phase — ingests results, writes final verifier JSONs.
python3 scripts/cc_run_round.py --phase finalize --workspace workspace --round 1
```

If `finalize` exits 7, a second wave was emitted (typical for `verify_internal_contradiction`'s compare step). Dispatch the new task file the same way and re-invoke `finalize` once more.

```bash
# Preprocess: extract claims from paper (no LLM needed)
python ~/.claude/skills/review-verification-harness/scripts/extract_claims.py \
  --tex paper.tex --bib paper.bib --out-dir workspace/claims/

# Content verifiers (run in parallel)
python ~/.claude/skills/review-verification-harness/scripts/verify_citations_full.py \
  --claims workspace/claims/citation_claims.jsonl \
  --out workspace/verifier/citations.json

python ~/.claude/skills/review-verification-harness/scripts/verify_internal_consistency.py \
  --tex paper.tex --out workspace/verifier/scope.json

python ~/.claude/skills/review-verification-harness/scripts/verify_internal_contradiction.py \
  --tex paper.tex --out workspace/verifier/contradiction.json

python ~/.claude/skills/review-verification-harness/scripts/verify_math_sympy.py \
  --tex paper.tex --out workspace/verifier/math.json

# Round-regression (between rounds)
python ~/.claude/skills/review-verification-harness/scripts/verify_round_regression.py \
  --prev-tex paper_v1.tex --curr-tex paper_v2.tex \
  --prev-verifier-dir workspace/r1/verifier/ --curr-verifier-dir workspace/r2/verifier/ \
  --prev-claims workspace/r1/claims/ --curr-claims workspace/r2/claims/ \
  --fix-log workspace/r2/fix_log.json \
  --out workspace/r2/verifier/regression.json
```

All output files drop into `<workspace>/verifier/`; `paper-audit`'s consolidator auto-picks them up.

## The ≤5-round submission pipeline

This harness is designed around a 5-round gate that guarantees no paper ships with a fatal, rebuttal-blocking error. See `references/PIPELINE_5ROUND.md` for the full runbook.

| Round | Action | Skill |
|---|---|---|
| R1 | Extract claims + run all verifiers + paper-audit deep review | this harness + `paper-audit` |
| R2 | Fix P0/major findings (with fix_log.json); diff-mode re-verify | this harness + user/agent edits |
| R3 | **Regression sweep** (mandatory) — catches R2-induced breakage | `verify_round_regression.py` |
| R4 | Adversarial probes (15 hostile agents) | `ai-validation-blindspots` |
| R5 | Camera-ready gate: all verified OR explicit human override | this harness |

**Why 5 rounds?** R3 is the structural safety net you don't have without it — most "AI-reviewed paper still gets rejected" failures come from R2-fix regressions that no one checks for. R4 adds human-reviewer-style adversarial pressure after the script-evidence rounds. If R5 still has P0 findings → the paper is not ready; no 6th round will save it.

## Roadmap — Stage C (not in this release)

Stage C moves the classes currently covered only by `ai-validation-blindspots` narrative probes into script-evidence:
- `verify_reproducibility.py` — sandbox clone + pip install + headline script (U9, U14)
- `verify_stats_assumptions.py` — Shapiro/Levene/power on reported numbers (U8, U13)
- `verify_figure_integrity.py` — axis truncation, axis-consistency scan (U6)
- `verify_novelty_broad.py` — multi-language + pre-2020 literature scan (U11)

See `references/ERROR_TAXONOMY_UNFIXABLE.md` for the full 15-class coverage matrix.

## Relationship to `ai-validation-blindspots`

Two complementary layers:

| Layer | Skill | When | Output |
|---|---|---|---|
| **Script evidence** | `review-verification-harness` (this) | every round | JSON verdicts with DOI/sympy/diff evidence |
| **Adversarial narrative** | `ai-validation-blindspots` | R4, pre-submission | `threat_report.json` with 15 hostile-agent findings |

Blindspots skill's Classes 9–11 (empirical rigor / reproducibility / figure integrity) cover the rows that harness Stages A+B don't yet script-verify. A paper that passes both layers has concrete evidence for the check-able classes and adversarial pressure-tested the rest.

## Environment requirements

Stage B2 preflight (`scripts/_preflight.py`) checks these before any verifier runs. `run_round.sh` invokes preflight by default; `--skip-preflight` bypasses.

| Dependency | Needed by | Kind if missing | Impact |
|---|---|---|---|
| `ANTHROPIC_API_KEY` env var | `verify_citations_full`, `verify_internal_contradiction`, LLM math fallback | **env** | Top-level `unverifiable`; consumer routes to `setup_needed` (not gate_blocker). |
| `anthropic` Python SDK | same as above | **env** | Same — fixable with `pip install anthropic`. |
| `sympy` | `verify_math_sympy` | **env** | Top-level `unverifiable`. Install with `pip install sympy`. |
| `antlr4-python3-runtime` | `verify_math_sympy` LaTeX parser | **tool** | Per-equation `unverifiable_kind=tool`. Install with `pip install antlr4-python3-runtime`. |
| CrossRef reachability | `verify_citations_full` DOI resolve | **env** | Network-blocked runs downgrade unresolvable citations to `unverifiable_kind=env`. |
| arXiv reachability | `verify_citations_full` arXiv fallback | **env** | Same; Semantic Scholar may still cover. |

Run `python3 scripts/_preflight.py` to get a human-readable diagnostic and a JSON report. Exit code 0 = ready, 1 = degraded (safe to run, expect env-tagged unverifiable), 2 = setup_incomplete (don't run).

## Philosophy

Verifiers **fail loud, not fail quiet**. When in doubt, return `unverifiable`. Never return `verified` based only on LLM reasoning without external evidence. Coverage-for-certainty is the right trade: flagging 10 findings as unverifiable (human must check) beats falsely certifying 1 as verified.

Stage B2 clarifies: `unverifiable ≠ pass` still holds, but the three-state taxonomy means `unverifiable_kind=env|tool` are NOT paper defects — they are setup/coverage gaps. Only `unverifiable_kind=evidence` maps to a gate-blocking P0. This is the fix for the false-P0 drowning failure mode the ESORICS conference-paper run exposed.

## Failure modes

Known conditions under which verifier outputs are noisy, degraded, or misleading. These are expected behaviors, not bugs — but they require human interpretation rather than blind trust:

- **Missing `ANTHROPIC_API_KEY`** — two paths: (a) Outside Claude Code, `verify_citations_full` and `verify_internal_contradiction` emit `unverifiable` P0 at the top level. Interpret as "needs human review", not as an actual paper defect. (b) Inside Claude Code Max, use the CC-bridge flow: `scripts/cc_run_round.py` defers LLM work to dispatched subagents via `--judge-backend batch-emit` / `batch-ingest`. Results are equivalent to the API-key path; only the dispatcher changes.
- **Missing `--bib`** — `extract_claims` will still emit citation records but with empty reference metadata; `verify_citations_full` then emits `unverifiable` P0 for every citation (no DOI/author/year to resolve against). Supply a `.bib` file or expect uniformly noisy citation output that is not actionable.
- **sympy not installed** — `verify_math_sympy` emits top-level `unverifiable` P0 gracefully (does not crash). Install with `pip install sympy antlr4-python3-runtime` to enable real symbolic/numerical equation checking; otherwise treat math targets as human-review-required.
- **Large citation count (>100)** — citation verifier spends ~2× the typical per-paper LLM cost (two-pass judge, ~1 call per citation × ~2 judges). Use `--fast` to halve LLM calls by skipping the second-opinion pass. Accept a small hit in precision for a ~50% cost reduction.
- **Previous-round workspace collision** — re-running R1 into a directory that already holds verifier JSON will silently mix the two rounds' reports in the consolidator (it globs `*.json`). **Always use a fresh `workspace/rN/` per round.** The orchestrator `run_round.sh` enforces this and refuses to overwrite without `--force`.

## Cost

Stage A+B per full-paper run: roughly $3–8 (citation LLM judge + contradiction judge dominates; sympy is free; round regression is free). `--fast` mode on citation + contradiction verifiers skips second-opinion calls for budget-sensitive runs — expect roughly a **~40% cost reduction** for those two verifiers combined, at the price of losing adversarial second-pass disagreement signal on individual targets.
