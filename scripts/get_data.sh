#!/usr/bin/env bash
# Downloads BEIR SciFact into data/scifact/. Safe to run more than once:
# if the dataset is already present, it does nothing.
set -euo pipefail

URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
TARGET="$DATA_DIR/scifact"

required=(corpus.jsonl queries.jsonl qrels/test.tsv)

have_all() {
    for f in "${required[@]}"; do
        [[ -s "$TARGET/$f" ]] || return 1
    done
    return 0
}

if have_all; then
    echo "SciFact already present in $TARGET, nothing to do."
    exit 0
fi

mkdir -p "$DATA_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $URL"
curl -fsSL --retry 5 -o "$tmp/scifact.zip" "$URL"

# Unzip into a temp dir first, then move into place, so an interrupted run
# never leaves a half-extracted dataset that have_all() would accept.
unzip -q "$tmp/scifact.zip" -d "$tmp/extract"
rm -rf "$TARGET"
mv "$tmp/extract/scifact" "$TARGET"

if ! have_all; then
    echo "error: download finished but expected files are missing" >&2
    exit 1
fi
echo "SciFact ready in $TARGET"
