# review-verification-harness

Script-backed verification primitives for academic paper review. Converts "LLM-thinks-it's-right" review comments into evidence-grounded verdicts by resolving DOIs, fetching abstracts, running sympy on equations, cross-section contradiction detection, and round-over-round regression checking.

Pairs with [`ai-validation-blindspots`](https://github.com/yudabitrends/ai-validation-blindspots) (R4 adversarial probes) and `paper-audit` to form a ≤5-round submission gate.

## Three-state verdict design

- **verified** — a script or API call confirms the finding is real
- **unverifiable** — evidence-gathering was inconclusive; **treat as P0**
- **failed** — the claimed finding was disproven

**Design rule:** `unverifiable` is never equivalent to `pass`. If a verifier cannot check, the paper is not ready to ship.

## Stage A + B verifiers

| Script | Stage | Coverage |
|---|---|---|
| `extract_claims.py` | preprocessor | LaTeX+BibTeX → citation / numerical / scope claims JSONL |
| `verify_citations_full.py` | A | U2 fake / misattributed citation (DOI+arXiv+S2 + adversarial 2-pass LLM judge) |
| `verify_internal_consistency.py` | A | U5 scope overclaim (regex-based abstract↔methods scope-diff) |
| `verify_internal_contradiction.py` | B | U10 abstract↔conclusion semantic contradiction (LLM triple extraction) |
| `verify_math_sympy.py` | B | U1 silent math / sign error (sympy symbolic + 3-seed numerical) |
| `verify_round_regression.py` | B | round-over-round safety: regression, drift, deletion-evasion, new-unchecked |

## Quick start

```bash
# Preprocess + 4 content verifiers in one shot
bash scripts/run_round.sh 1 paper.tex \
  --bib paper.bib --workspace workspace --fast
```

See `SKILL.md` for the full pipeline and `references/PIPELINE_5ROUND.md` for the ≤5-round submission protocol runbook.

## Output contract

Every verifier emits JSON conforming to [`references/VERIFIER_CONTRACT.md`](references/VERIFIER_CONTRACT.md). Downstream consumers (e.g. `paper-audit`) read the JSONs and upgrade `unverifiable` findings to `gate_blocker`.

## Tests

```bash
python3 tests/test_stage_a_smoke.py   # 7/7
python3 tests/test_stage_b_smoke.py   # 19/19
```

## Installation as a Claude Code skill

Drop the directory into `~/.claude/skills/` and invoke via the `review-verification-harness` skill from Claude Code, or call individual scripts directly.

## License

MIT (add a LICENSE file before publishing).
