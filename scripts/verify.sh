#!/usr/bin/env bash
# Regression gate for Phase 6: an optimisation must not change results.
#  1. Python reference cross-check (Phase 4): 50 random queries must match.
#  2. Full evaluation (Phase 5): the run file must be byte-identical to the
#     committed results/bm25_run.trec, which implies identical nDCG@10.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "== Phase 4 cross-check"
"$PY" scripts/reference_bm25.py | tail -2

echo "== Phase 5 evaluation"
build/apps/bm25_eval --out "$tmp/run.trec" --metrics-out "$tmp/metrics.tsv" | grep -E "nDCG|Recall"
if cmp -s "$tmp/run.trec" results/bm25_run.trec; then
    echo "run file: identical to results/bm25_run.trec"
else
    echo "run file: DIFFERS from results/bm25_run.trec" >&2
    exit 1
fi
