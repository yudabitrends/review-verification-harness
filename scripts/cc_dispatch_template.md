# Claude Code dispatch template — review-verification-harness CC-bridge

You are dispatching LLM judgment tasks for `review-verification-harness` v0.4.
The orchestrator `cc_run_round.py --phase emit` has produced one or more
`*.jsonl` task files under `<workspace>/r<N>/judge_tasks/`. For each task
file, invoke subagents to process tasks in batches of up to 10.

## Task schema

Each line in a task JSONL file is a JSON object with these fields:

- `task_id` — unique identifier you **MUST preserve** verbatim in your result.
- `verifier` — which verifier emitted the task (informational).
- `target_locator` — the locator of the pending target being judged.
- `kind` — one of `primary_judge`, `devils_advocate`, `triple_extract`,
  `compare`, `math_llm_fallback`.
- `system` — the subagent's system prompt.
- `user` — the subagent's user message.
- `max_tokens`, `temperature`, `expected_format` — inference parameters.

## Per-batch dispatch

Group up to 10 tasks per subagent invocation (via the Task/Agent tool). Each
subagent's sole job is to return a JSON array of results, one per task:

```json
[
  {
    "task_id": "...",
    "ok": true,
    "body": "... verbatim response body ...",
    "tokens_in": 0,
    "tokens_out": 0,
    "error": null
  },
  ...
]
```

Write these results to `<workspace>/r<N>/judge_results/<verifier>.results.jsonl`
(one JSON object per line). Expected filenames:

- `verify_citations_full.results.jsonl`
- `verify_internal_contradiction.results.jsonl`
- `verify_math_sympy.results.jsonl`

When every task file has a corresponding results file, run:

```
python3 scripts/cc_run_round.py --phase finalize \
    --workspace <workspace> --round <N>
```

If `finalize` exits 7, a second wave has been emitted (typical for the
contradiction verifier's compare step). Check
`<workspace>/r<N>/cc_bridge_summary.json` under `second_wave[]` for new task
files, dispatch them the same way, write new results, re-invoke `finalize`.

## Conventions

- **Batching**: 10 tasks per subagent keeps context small; larger batches risk
  truncation. Only raise the limit if you have strong reason.
- **Preserve `task_id` verbatim**: the orchestrator matches results by exact
  string. Paraphrased ids will be dropped silently.
- **Validate JSON before writing**: any subagent response that is not valid
  JSON must be written with `ok: false` and `error: "non-JSON response"`.
  Don't drop the line — the orchestrator needs every `task_id` present.
- **Expected output format**: `expected_format` is advisory. For
  `primary_judge`, `devils_advocate`, `math_llm_fallback`, the inner response
  is a JSON object (verdict/confidence/reason). For `triple_extract` and
  `compare`, it is a JSON array.
- **Model choice**: no constraint here. Claude Max subagents inherit the
  current session's model; for faster batches, prefer sonnet-tier.

## End state

When `cc_run_round.py --phase finalize` exits 0, the round's verifier JSONs
are complete. Downstream consumers (paper-audit's `consolidate_review_findings`)
can read them directly.
