# Verifier I/O Contract

All verifier scripts under `~/.claude/skills/review-verification-harness/scripts/` (and the parallel `aps-verification-harness`) emit a single JSON document per run, conforming to this schema.

## Top-level schema

```json
{
  "verifier_id": "string",
  "verifier_version": "semver string",
  "status": "verified | unverifiable | failed",
  "severity_suggestion": "P0 | major | moderate | minor",
  "summary": "one-line human-readable",
  "targets": [ <TargetResult> ],
  "metadata": {
    "generated_at": "ISO8601",
    "cost_usd": 0.0,
    "model": "optional model id",
    "cached": false,
    "inputs": { "...": "echo of key inputs for reproducibility" }
  },
  "errors": [ "optional list of recoverable errors encountered" ]
}
```

## TargetResult

Each verifier runs over multiple targets (citations, section pairs, equations, ...). One entry per target:

```json
{
  "locator": "human-pointer (e.g. 'cite:smith2024', 'pair:abstract-methods', 'eq:3')",
  "status": "verified | unverifiable | failed",
  "evidence": {
    "quote": "exact text from paper that raised the check",
    "external_url": "optional DOI/arXiv/URL proving or disproving",
    "judge_notes": "LLM judge explanation if used",
    "judge_confidence": "high|medium|low",
    "second_opinion_agreed": true
  },
  "severity_suggestion": "P0 | major | moderate | minor",
  "root_cause_key": "stable slug (e.g. 'citation-content-mismatch-smith2024')"
}
```

## Status semantics (strict)

- **verified** — external evidence decisively confirms the finding. Downstream must treat as actionable defect.
- **unverifiable** — evidence gathering was inconclusive (API unreachable, abstract missing, ambiguous LLM judge, paper paywalled without abstract). **Downstream MUST upgrade to P0** and require human attention. Never silently pass.
- **failed** — the claimed finding was checked and disproven (false alarm). Downstream can drop the finding.

## Severity suggestion mapping

Verifier returns a `severity_suggestion` per target, but the **final severity is decided by the consumer** (`paper-audit`, `convergent-peer-review`). Recommended mapping:

| Verifier outcome | Suggested severity |
|---|---|
| status=verified + content-level error (wrong claim attribution, scope exceeds methods) | `P0` |
| status=verified + moderate issue (ambiguity, minor scope drift) | `major` |
| status=unverifiable | `P0` (upgrade) — consumer may override with `--allow-unverifiable` flag only with user acknowledgment |
| status=failed | drop or `minor` if informational |

## Consumer responsibilities

Any skill reading verifier JSON must:

1. Respect the `status` field. Never downgrade `unverifiable` without explicit override.
2. Preserve `evidence.external_url` and `judge_quotes` in the final report so a human can audit the verifier's judgment.
3. Emit its own audit trail showing `verifier_id + version` alongside each finding upgrade/downgrade.

## Error handling

Verifier scripts should **never crash** when a single target fails. They must catch per-target exceptions, emit `status=unverifiable` with the error in `evidence.judge_notes`, and continue. An empty `targets` list with a top-level error is only valid when the verifier's input was invalid.

## Caching

Verifiers that make external API calls (CrossRef, arXiv, Semantic Scholar, OpenAlex) should cache results under `~/.claude/cache/review-verification-harness/<verifier_id>/<hash>.json` with a 30-day TTL. The `metadata.cached` flag should reflect cache hit state.

## Example: verify_citations_full output

```json
{
  "verifier_id": "verify_citations_full",
  "verifier_version": "0.1",
  "status": "failed",
  "severity_suggestion": "P0",
  "summary": "2 of 7 citations had unsupported or unverifiable content-claim match",
  "targets": [
    {
      "locator": "cite:cardy2024",
      "status": "unverifiable",
      "evidence": {
        "quote": "As shown by Cardy et al. (2024), entropy production satisfies X.",
        "external_url": null,
        "judge_notes": "Could not resolve DOI or find paper matching 'Cardy 2024 entropy production'. Likely fabricated.",
        "judge_confidence": "high",
        "second_opinion_agreed": true
      },
      "severity_suggestion": "P0",
      "root_cause_key": "citation-unresolvable-cardy2024"
    },
    {
      "locator": "cite:smith2020",
      "status": "verified",
      "evidence": {
        "quote": "Smith (2020) shows linear scaling for n<1000.",
        "external_url": "https://doi.org/10.1234/smith.2020",
        "judge_notes": "Abstract confirms linear-scaling claim, same regime.",
        "judge_confidence": "high",
        "second_opinion_agreed": true
      },
      "severity_suggestion": "minor",
      "root_cause_key": "citation-match-smith2020"
    }
  ],
  "metadata": {
    "generated_at": "2026-04-21T10:30:00Z",
    "cost_usd": 3.21,
    "model": "anthropic:claude-sonnet-4-6",
    "cached": false,
    "inputs": { "claims_file": "workspace/citation_claims.jsonl", "n_targets": 7 }
  }
}
```
