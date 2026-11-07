#!/usr/bin/env python3
"""Evaluates TREC run files with pytrec_eval (the trec_eval C code) and,
optionally, cross-checks the C++ metrics written by bm25_eval.

Usage:
  python3 scripts/eval.py results/bm25_run.trec \
      --qrels data/scifact/qrels/test.tsv --cpp-metrics results/bm25_metrics.tsv

Means are taken over every query in the qrels. A query missing from the run
counts as 0. (pytrec_eval on its own silently skips such queries, which would
inflate the mean.)
"""

import argparse
import sys

import pytrec_eval

MEASURES = {"ndcg_cut_10", "recall_100"}


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.split()
            if i == 0 and parts[0] == "query-id":
                continue
            qid, docid, grade = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[docid] = grade
    return qrels


def load_run(path):
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _q0, docid, _rank, score, _tag = line.split()
            run.setdefault(qid, {})[docid] = float(score)
    return run


def evaluate(qrels, run):
    """Returns ({qid: {measure: value}}, {measure: mean over all qrels queries})."""
    per_query = pytrec_eval.RelevanceEvaluator(qrels, MEASURES).evaluate(run)
    means = {}
    for m in MEASURES:
        means[m] = sum(per_query.get(q, {}).get(m, 0.0) for q in qrels) / len(qrels)
    return per_query, means


def load_cpp_metrics(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().split()
        for line in f:
            parts = line.split()
            out[parts[0]] = {header[i]: float(parts[i]) for i in range(1, len(header))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="TREC run files")
    ap.add_argument("--qrels", default="data/scifact/qrels/test.tsv")
    ap.add_argument("--cpp-metrics", help="per-query TSV from bm25_eval, compared to the first run")
    ap.add_argument("--tol", type=float, default=0.001)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    status = 0
    first_per_query = None
    first_means = None
    for path in args.runs:
        run = load_run(path)
        per_query, means = evaluate(qrels, run)
        missing = len([q for q in qrels if q not in run])
        print(f"{path}: queries={len(qrels)} missing_from_run={missing} "
              f"nDCG@10={means['ndcg_cut_10']:.4f} Recall@100={means['recall_100']:.4f}")
        if first_per_query is None:
            first_per_query, first_means = per_query, means

    if args.cpp_metrics:
        cpp = load_cpp_metrics(args.cpp_metrics)
        worst = {m: 0.0 for m in MEASURES}
        for qid in qrels:
            for m in MEASURES:
                py = first_per_query.get(qid, {}).get(m, 0.0)
                worst[m] = max(worst[m], abs(py - cpp[qid][m]))
        print("C++ vs pytrec_eval:")
        for m in sorted(MEASURES):
            d_mean = abs(first_means[m] - cpp["all"][m])
            ok = d_mean <= args.tol
            status |= 0 if ok else 1
            print(f"  {m:12s} C++ {cpp['all'][m]:.6f}  pytrec_eval {first_means[m]:.6f}  "
                  f"|diff of means| {d_mean:.2e}  max per-query |diff| {worst[m]:.2e}  "
                  f"{'OK' if ok else 'FAIL'} (tol {args.tol})")
    return status


if __name__ == "__main__":
    sys.exit(main())
