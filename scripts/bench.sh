#!/usr/bin/env bash
# Runs `bm25_eval --bench` several times, pinned to one CPU core, so the
# numbers in results/perf.md can be reproduced. Each run already does a 15 s
# warm-up, 5 repetitions, and discards the first repetition.
#
# Usage: scripts/bench.sh [trials=3] [extra bm25_eval args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIALS="${1:-3}"
shift || true
for t in $(seq 1 "$TRIALS"); do
    echo "== trial $t"
    taskset -c 4 "$ROOT/build/apps/bm25_eval" --data "$ROOT/data/scifact" --bench "$@" |
        grep -E "build|p50|p95|p99|mean"
done
