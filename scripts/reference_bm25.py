#!/usr/bin/env python3
"""Independent plain-Python BM25, used to cross-check the C++ engine.

It does NOT tokenize. It reads the tokens the C++ tokenizer exported
(bm25_index --export-tokens), so any disagreement is in indexing or scoring,
not in tokenization. It builds its own postings and statistics from those
tokens, scores with exactly the CLAUDE.md formula, then runs the C++
bm25_search binary on the same queries and compares the top-k lists.

Usage:
  build/apps/bm25_index --export-tokens data/scifact/tokens
  python3 scripts/reference_bm25.py --tokens data/scifact/tokens \
      --search build/apps/bm25_search --data data/scifact
"""

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict


def read_tokens(path):
    """Returns [(id, [tokens])] in file order."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            ident, _, toks = line.partition("\t")
            rows.append((ident, toks.split()))
    return rows


class ReferenceBm25:
    def __init__(self, docs, k1, b):
        self.k1 = k1
        self.b = b
        self.doc_ids = [d for d, _ in docs]
        self.doc_len = [len(t) for _, t in docs]
        self.n_docs = len(docs)
        self.avgdl = sum(self.doc_len) / self.n_docs
        # term -> list of (doc_index, tf), doc_index ascending
        self.postings = defaultdict(list)
        for i, (_, toks) in enumerate(docs):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))

    def idf(self, n):
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def search(self, query_tokens, k):
        # Repeated query terms count once; sum in first-occurrence order.
        terms = list(dict.fromkeys(query_tokens))
        k1, b, avgdl = self.k1, self.b, self.avgdl
        scores = {}
        for term in terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(len(plist))
            for doc, tf in plist:
                dl = self.doc_len[doc]
                s = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
                scores[doc] = scores.get(doc, 0.0) + s
        # Full sort is fine here: this is the slow, obviously-correct version.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.doc_ids[d], s) for d, s in ranked]


def load_query_text(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[obj["_id"]] = obj["text"]
    return out


def run_cpp(search_bin, data_dir, queries, k, k1, b):
    """Runs bm25_search --batch and parses its TREC output into {qid: [(doc, score)]}."""
    stdin = "".join(f"{qid}\t{text}\n" for qid, text in queries)
    proc = subprocess.run(
        [search_bin, "--batch", "--data", data_dir, "-k", str(k), "--k1", str(k1), "--b", str(b)],
        input=stdin, capture_output=True, text=True, check=True)
    out = defaultdict(list)
    for line in proc.stdout.splitlines():
        qid, _, doc, _rank, score, _tag = line.split()
        out[qid].append((doc, float(score)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="data/scifact/tokens")
    ap.add_argument("--data", default="data/scifact")
    ap.add_argument("--search", default="build/apps/bm25_search")
    ap.add_argument("--num-queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    docs = read_tokens(f"{args.tokens}/corpus_tokens.tsv")
    query_tokens = dict(read_tokens(f"{args.tokens}/query_tokens.tsv"))
    ref = ReferenceBm25(docs, args.k1, args.b)
    print(f"reference index: {ref.n_docs} docs, {len(ref.postings)} terms, avgdl {ref.avgdl:.4f}")

    qids = sorted(query_tokens, key=int)
    sample = random.Random(args.seed).sample(qids, min(args.num_queries, len(qids)))
    texts = load_query_text(f"{args.data}/queries.jsonl")
    cpp = run_cpp(args.search, args.data, [(q, texts[q]) for q in sample], args.k, args.k1, args.b)

    matched = 0
    max_diff = 0.0
    for qid in sample:
        want = ref.search(query_tokens[qid], args.k)
        got = cpp.get(qid, [])
        same_docs = [d for d, _ in want] == [d for d, _ in got]
        diff = max((abs(a[1] - b[1]) for a, b in zip(want, got)), default=0.0)
        max_diff = max(max_diff, diff)
        if same_docs and diff <= args.tol:
            matched += 1
        else:
            print(f"MISMATCH query {qid}: same_docs={same_docs} max_score_diff={diff:.3e}")
            print(f"  python: {want}")
            print(f"  c++:    {got}")

    print(f"queries compared: {len(sample)} (seed {args.seed}, top-{args.k}, k1={args.k1}, b={args.b})")
    print(f"max |score diff|: {max_diff:.3e} (tolerance {args.tol:g})")
    print(f"matched: {matched}/{len(sample)}")
    return 0 if matched == len(sample) else 1


if __name__ == "__main__":
    sys.exit(main())
