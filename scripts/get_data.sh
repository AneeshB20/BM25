#!/usr/bin/env bash
# Downloads BEIR SciFact into data/scifact/, and the official Porter stemmer
# test vocabulary into data/porter/. Safe to run more than once: anything
# already present is left alone.
set -euo pipefail

URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
PORTER_URL="https://tartarus.org/martin/PorterStemmer"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
TARGET="$DATA_DIR/scifact"
PORTER_DIR="$DATA_DIR/porter"

required=(corpus.jsonl queries.jsonl qrels/test.tsv)

have_all() {
    for f in "${required[@]}"; do
        [[ -s "$TARGET/$f" ]] || return 1
    done
    return 0
}

mkdir -p "$DATA_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Porter test vocabulary (optional: the full-vocabulary unit test skips without it).
if [[ -s "$PORTER_DIR/voc.txt" && -s "$PORTER_DIR/output.txt" ]]; then
    echo "Porter vocabulary already present in $PORTER_DIR."
else
    echo "Downloading Porter test vocabulary"
    if curl -fsSL --retry 3 -o "$tmp/voc.txt" "$PORTER_URL/voc.txt" &&
       curl -fsSL --retry 3 -o "$tmp/output.txt" "$PORTER_URL/output.txt"; then
        mkdir -p "$PORTER_DIR"
        mv "$tmp/voc.txt" "$tmp/output.txt" "$PORTER_DIR/"
    else
        echo "warning: Porter vocabulary download failed; that test will be skipped" >&2
    fi
fi

if have_all; then
    echo "SciFact already present in $TARGET, nothing to do."
    exit 0
fi

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
