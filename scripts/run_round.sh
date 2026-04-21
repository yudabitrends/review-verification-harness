#!/usr/bin/env bash
# run_round.sh — orchestrate one round of the review-verification harness.
#
# Usage:
#   bash run_round.sh <round_number> <paper.tex> \
#                    [--bib paper.bib] [--workspace workspace] \
#                    [--fast] [--force] [--help]
#
# Behavior:
#   1. Creates workspace/r<N>/{tex,claims,verifier}/ (fresh). If the round
#      directory already exists, refuses to overwrite unless --force is given.
#   2. Copies <paper.tex> → workspace/r<N>/tex/paper.tex (snapshot).
#   3. Runs extract_claims.py → workspace/r<N>/claims/.
#   4. Runs the 4 content verifiers in parallel (internal_consistency,
#      internal_contradiction, math_sympy, citations_full) → workspace/r<N>/verifier/.
#      Skips citations_full if no --bib was supplied.
#   5. Prints per-verifier status + a top-level summary.
#   6. Exits 0 only if every verifier that ran produced a readable JSON report
#      (even `unverifiable` counts as "ran successfully"). Exits non-zero on
#      script/JSON failure.
#
# Regression:
#   Regression between rounds is a separate invocation. After R2 edits, run
#   verify_round_regression.py directly (see SKILL.md) — do not invoke this
#   orchestrator for R3 regression sweep.

set -u
set -o pipefail

HARNESS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPTS_DIR="$HARNESS_DIR"

show_help() {
    cat <<'EOF'
run_round.sh — one-shot harness round orchestrator

USAGE:
  bash run_round.sh <round_number> <paper.tex> \
                    [--bib <paper.bib>] [--workspace <dir>] \
                    [--fast] [--force]

POSITIONAL:
  <round_number>   integer round id; produces workspace/r<N>/
  <paper.tex>      LaTeX source to analyze

OPTIONS:
  --bib <file>     BibTeX file; without this, verify_citations_full is skipped
                   (with a warning) because citations cannot be resolved.
  --workspace <d>  Parent workspace directory (default: workspace)
  --fast           Pass through to citation + contradiction verifiers —
                   skips adversarial second-opinion pass (~40% cheaper).
  --force          Allow overwriting an existing workspace/r<N>/ directory.
                   WITHOUT this flag, the script refuses to proceed if the
                   round workspace already exists (prevents dirty mixes).

OUTPUTS:
  workspace/r<N>/tex/paper.tex
  workspace/r<N>/claims/{citation,numerical,scope}_claims.jsonl
  workspace/r<N>/claims/extract_summary.json
  workspace/r<N>/verifier/citations.json   (if --bib supplied)
  workspace/r<N>/verifier/scope.json
  workspace/r<N>/verifier/contradiction.json
  workspace/r<N>/verifier/math.json

EXIT STATUS:
  0  every verifier that ran produced a readable JSON report (any status)
  1  one or more verifiers crashed or produced no output

REGRESSION (note):
  This orchestrator does NOT run verify_round_regression.py. That's a
  separate workflow that compares round N-1 vs N and runs between rounds.
  See SKILL.md and references/PIPELINE_5ROUND.md (R3 section).

EXAMPLES:
  bash run_round.sh 1 paper.tex --bib paper.bib
  bash run_round.sh 2 paper_v2.tex --bib paper.bib --fast
  bash run_round.sh 1 paper.tex --workspace /tmp/ws --force
EOF
}

# --- Argument parsing ---------------------------------------------------------

if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

# --help can appear anywhere; check first.
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        show_help
        exit 0
    fi
done

if [[ $# -lt 2 ]]; then
    echo "error: need at least <round_number> and <paper.tex>" >&2
    echo "run 'bash run_round.sh --help' for usage." >&2
    exit 2
fi

ROUND="$1"; shift
TEX="$1"; shift

BIB=""
WORKSPACE="workspace"
FAST_FLAG=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bib)
            BIB="${2:-}"; shift 2 ;;
        --workspace)
            WORKSPACE="${2:-}"; shift 2 ;;
        --fast)
            FAST_FLAG="--fast"; shift ;;
        --force)
            FORCE=1; shift ;;
        *)
            echo "error: unknown option '$1'" >&2
            echo "run 'bash run_round.sh --help' for usage." >&2
            exit 2 ;;
    esac
done

# --- Validation --------------------------------------------------------------

if ! [[ "$ROUND" =~ ^[0-9]+$ ]]; then
    echo "error: <round_number> must be a non-negative integer, got '$ROUND'" >&2
    exit 2
fi

if [[ ! -f "$TEX" ]]; then
    echo "error: paper.tex not found: $TEX" >&2
    exit 2
fi

if [[ -n "$BIB" && ! -f "$BIB" ]]; then
    echo "error: --bib file not found: $BIB" >&2
    exit 2
fi

ROUND_DIR="$WORKSPACE/r${ROUND}"
if [[ -e "$ROUND_DIR" ]]; then
    if [[ $FORCE -ne 1 ]]; then
        echo "warning: round workspace already exists: $ROUND_DIR" >&2
        echo "         re-running into an existing round directory will mix" >&2
        echo "         verifier JSONs across rounds in the consolidator." >&2
        echo "         re-run with --force to overwrite, or use a fresh workspace." >&2
        exit 3
    fi
    echo "warning: --force supplied; removing existing $ROUND_DIR"
    rm -rf "$ROUND_DIR"
fi

mkdir -p "$ROUND_DIR/tex" "$ROUND_DIR/claims" "$ROUND_DIR/verifier"

# --- 2. Snapshot tex ---------------------------------------------------------

cp "$TEX" "$ROUND_DIR/tex/paper.tex"
TEX_SNAPSHOT="$ROUND_DIR/tex/paper.tex"
echo "[run_round] snapshot: $TEX_SNAPSHOT"

# --- 3. Preprocessor: extract_claims ----------------------------------------

echo "[run_round] running extract_claims.py ..."
EXTRACT_ARGS=(--tex "$TEX_SNAPSHOT" --out-dir "$ROUND_DIR/claims")
if [[ -n "$BIB" ]]; then
    EXTRACT_ARGS+=(--bib "$BIB")
fi
set +e
python3 "$SCRIPTS_DIR/extract_claims.py" "${EXTRACT_ARGS[@]}"
EXTRACT_RC=$?
set -e
if [[ $EXTRACT_RC -ne 0 && $EXTRACT_RC -ne 1 ]]; then
    # rc=1 from extract_claims means "no claims extracted" (status=unverifiable);
    # that is still a readable report. Anything else means a real crash.
    echo "error: extract_claims.py crashed with rc=$EXTRACT_RC" >&2
    exit 1
fi

# --- 4. Content verifiers in parallel ---------------------------------------

LOG_DIR="$ROUND_DIR/verifier/_logs"
mkdir -p "$LOG_DIR"

declare -a PIDS=()
declare -a NAMES=()
declare -a REPORTS=()

run_bg() {
    local name="$1"; shift
    local report="$1"; shift
    echo "[run_round] launching: $name -> $report"
    ( "$@" ) >"$LOG_DIR/${name}.stdout" 2>"$LOG_DIR/${name}.stderr" &
    PIDS+=("$!")
    NAMES+=("$name")
    REPORTS+=("$report")
}

# verify_internal_consistency
run_bg "internal_consistency" "$ROUND_DIR/verifier/scope.json" \
    python3 "$SCRIPTS_DIR/verify_internal_consistency.py" \
    --tex "$TEX_SNAPSHOT" --out "$ROUND_DIR/verifier/scope.json"

# verify_internal_contradiction (pass --fast if requested)
CONTRA_ARGS=(--tex "$TEX_SNAPSHOT" --out "$ROUND_DIR/verifier/contradiction.json")
if [[ -n "$FAST_FLAG" ]]; then CONTRA_ARGS+=("$FAST_FLAG"); fi
run_bg "internal_contradiction" "$ROUND_DIR/verifier/contradiction.json" \
    python3 "$SCRIPTS_DIR/verify_internal_contradiction.py" "${CONTRA_ARGS[@]}"

# verify_math_sympy
run_bg "math_sympy" "$ROUND_DIR/verifier/math.json" \
    python3 "$SCRIPTS_DIR/verify_math_sympy.py" \
    --tex "$TEX_SNAPSHOT" --out "$ROUND_DIR/verifier/math.json"

# verify_citations_full — only if --bib was supplied
if [[ -n "$BIB" ]]; then
    CITE_CLAIMS="$ROUND_DIR/claims/citation_claims.jsonl"
    if [[ -f "$CITE_CLAIMS" ]]; then
        CITE_ARGS=(--claims "$CITE_CLAIMS" --out "$ROUND_DIR/verifier/citations.json")
        if [[ -n "$FAST_FLAG" ]]; then CITE_ARGS+=("$FAST_FLAG"); fi
        run_bg "citations_full" "$ROUND_DIR/verifier/citations.json" \
            python3 "$SCRIPTS_DIR/verify_citations_full.py" "${CITE_ARGS[@]}"
    else
        echo "[run_round] warning: $CITE_CLAIMS missing; skipping citations_full"
    fi
else
    echo "[run_round] warning: no --bib supplied; skipping verify_citations_full"
    echo "            (citation resolution requires BibTeX metadata)"
fi

# --- Wait for all verifiers --------------------------------------------------

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    report="${REPORTS[$i]}"
    set +e
    wait "$pid"
    rc=$?
    set -e
    if [[ $rc -ne 0 && $rc -ne 1 ]]; then
        # rc=1 from verifiers commonly means "ran, status != verified". The
        # contract requires the JSON to exist regardless; enforce that below.
        echo "[run_round] WARN $name exited with rc=$rc (logs in $LOG_DIR)" >&2
    fi
    if [[ ! -s "$report" ]]; then
        echo "[run_round] ERROR $name did not produce $report" >&2
        FAILED=1
        continue
    fi
    if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$report" >/dev/null 2>&1; then
        echo "[run_round] ERROR $name produced invalid JSON at $report" >&2
        FAILED=1
        continue
    fi
    echo "[run_round] ok   $name -> $report"
done

# --- Per-verifier + top-level summary ---------------------------------------

echo ""
echo "=== Round $ROUND summary ==="
python3 - "$ROUND_DIR" <<'PY'
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
verifier_dir = root / "verifier"
claims_dir = root / "claims"

totals = {"verified": 0, "unverifiable": 0, "failed": 0}

def summarize_report(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  - {path.name}: UNREADABLE ({e})")
        return
    top_status = payload.get("status", "?")
    vid = payload.get("verifier_id", "?")
    targets = payload.get("targets") or []
    counts = {"verified": 0, "unverifiable": 0, "failed": 0}
    for t in targets:
        s = t.get("status")
        if s in counts:
            counts[s] += 1
            totals[s] += 1
    print(f"  - {path.name} ({vid}): top={top_status} "
          f"targets={len(targets)} "
          f"[verified={counts['verified']}, "
          f"unverifiable={counts['unverifiable']}, "
          f"failed={counts['failed']}]")

print("verifier/:")
if verifier_dir.is_dir():
    reports = sorted(p for p in verifier_dir.glob("*.json"))
    if not reports:
        print("  (no verifier reports)")
    for p in reports:
        summarize_report(p)

print("claims/:")
summary_path = claims_dir / "extract_summary.json"
if summary_path.exists():
    summarize_report(summary_path)
else:
    print("  (no extract_summary.json)")

print()
print(f"TOTAL across verifier/ targets: "
      f"verified={totals['verified']}, "
      f"unverifiable={totals['unverifiable']}, "
      f"failed={totals['failed']}")
PY

echo ""
if [[ $FAILED -ne 0 ]]; then
    echo "[run_round] FAILED: one or more verifiers did not produce a readable report."
    exit 1
fi

echo "[run_round] OK: all verifiers produced readable reports."
echo "[run_round] next step: consolidate + (between rounds) verify_round_regression.py"
echo "[run_round]   python3 ~/.claude/skills/paper-audit/scripts/consolidate_review_findings.py $ROUND_DIR"
exit 0
