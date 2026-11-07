#!/usr/bin/env python3
"""Dense retrieval baseline and BM25 + dense hybrid on BEIR SciFact.

1. Encodes the corpus (title + " " + text, as BM25 indexes it) and the judged
   test queries with sentence-transformers all-MiniLM-L6-v2, normalised, and
   retrieves the top-100 by cosine similarity (a dot product of unit vectors).
   Writes results/dense_run.trec.
2. Fuses results/bm25_run.trec and the dense run with Reciprocal Rank Fusion,
   score(d) = sum over runs of 1 / (60 + rank_run(d)), and writes
   results/hybrid_run.trec.
3. Evaluates all three runs with scripts/eval.py (pytrec_eval), measures
   per-query latency, and writes results/comparison.md with the table and three
   example queries (BM25 wins, dense wins, hybrid beats both).

Usage: .venv/bin/python scripts/dense_baseline.py
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as trec_eval  # noqa: E402  (scripts/eval.py)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
DEPTH = 100


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_run(path, ranked, tag):
    """ranked: {qid: [(docid, score), ...]} best first."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, hits in ranked.items():
            for rank, (doc, score) in enumerate(hits, start=1):
                f.write(f"{qid} Q0 {doc} {rank} {score:.9f} {tag}\n")


def read_ranked(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _q0, doc, rank, score, _tag = line.split()
            out.setdefault(qid, []).append((doc, float(score), int(rank)))
    for hits in out.values():
        hits.sort(key=lambda h: h[2])
    return {q: [(d, s) for d, s, _ in hits] for q, hits in out.items()}


def top_k(scores, k):
    """Indices of the k largest scores, best first (argpartition, then sort k)."""
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def rrf(runs, k=RRF_K, depth=DEPTH):
    fused = {}
    for qid in set().union(*runs):
        acc = {}
        for run in runs:
            for rank, (doc, _score) in enumerate(run.get(qid, []), start=1):
                acc[doc] = acc.get(doc, 0.0) + 1.0 / (k + rank)
        fused[qid] = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
    return fused


def percentile(sorted_vals, p):
    """Nearest-rank percentile, the same definition bm25_eval uses."""
    import math
    rank = max(1, math.ceil(p / 100 * len(sorted_vals)))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def bm25_latency(eval_bin, data_dir):
    """Runs the C++ benchmark (top-100, to match the dense depth) and parses it."""
    out = subprocess.run(["taskset", "-c", "4", eval_bin, "--data", data_dir, "--bench", "-k", "100"],
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    for line in out.splitlines():
        if line.startswith("query p50:"):
            vals["p50"] = float(line.split()[2])
        elif line.startswith("query p95:"):
            vals["p95"] = float(line.split()[2])
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/scifact")
    ap.add_argument("--cache", default="data/dense")
    ap.add_argument("--bm25-run", default="results/bm25_run.trec")
    ap.add_argument("--bm25-eval", default="build/apps/bm25_eval")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    corpus = load_jsonl(f"{args.data}/corpus.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    doc_text = [(d.get("title") or "") + " " + (d.get("text") or "") for d in corpus]
    titles = {d["_id"]: d.get("title") or "" for d in corpus}
    qrels = trec_eval.load_qrels(f"{args.data}/qrels/test.tsv")
    queries = [q for q in load_jsonl(f"{args.data}/queries.jsonl") if q["_id"] in qrels]
    print(f"corpus {len(corpus)} docs, {len(queries)} judged test queries", flush=True)

    model = SentenceTransformer(MODEL, device="cpu")
    print(f"model {MODEL}, max_seq_length {model.max_seq_length}, torch {torch.__version__}", flush=True)

    # --- 1. Corpus embeddings (cached: encoding is the slow part) -----------
    emb_path = f"{args.cache}/corpus_minilm.npy"
    if os.path.exists(emb_path):
        doc_emb = np.load(emb_path)
        encode_s = None
        print(f"loaded cached corpus embeddings {doc_emb.shape}", flush=True)
    else:
        t0 = time.perf_counter()
        doc_emb = model.encode(doc_text, batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        encode_s = time.perf_counter() - t0
        np.save(emb_path, doc_emb)
        print(f"encoded corpus {doc_emb.shape} in {encode_s:.1f} s ({torch.get_num_threads()} threads)",
              flush=True)

    # --- Dense retrieval, one query at a time, single thread, timed ---------
    torch.set_num_threads(1)
    for q in queries[:20]:  # warm-up
        model.encode(q["text"], normalize_embeddings=True, convert_to_numpy=True)
    dense, lat_ms, enc_ms = {}, [], []
    for q in queries:
        t0 = time.perf_counter()
        qv = model.encode(q["text"], normalize_embeddings=True, convert_to_numpy=True)
        t1 = time.perf_counter()
        scores = doc_emb @ qv
        idx = top_k(scores, DEPTH)
        t2 = time.perf_counter()
        dense[q["_id"]] = [(doc_ids[i], float(scores[i])) for i in idx]
        lat_ms.append((t2 - t0) * 1000)
        enc_ms.append((t1 - t0) * 1000)
    lat_ms.sort()
    write_run(f"{args.out_dir}/dense_run.trec", dense, "minilm")

    # --- 2. Hybrid with RRF ------------------------------------------------
    bm25 = read_ranked(args.bm25_run)
    t0 = time.perf_counter()
    hybrid = rrf([bm25, dense])
    fuse_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    write_run(f"{args.out_dir}/hybrid_run.trec", hybrid, "rrf60")

    # --- 3. Evaluate ---------------------------------------------------------
    results = {}
    for name, path in [("bm25", args.bm25_run), ("dense", f"{args.out_dir}/dense_run.trec"),
                       ("hybrid", f"{args.out_dir}/hybrid_run.trec")]:
        per_q, means = trec_eval.evaluate(qrels, trec_eval.load_run(path))
        results[name] = (per_q, means)
        print(f"{name:7s} nDCG@10 {means['ndcg_cut_10']:.4f}  Recall@100 {means['recall_100']:.4f}")

    bm = bm25_latency(args.bm25_eval, args.data)
    d50, d95 = percentile(lat_ms, 50), percentile(lat_ms, 95)
    print(f"latency p50/p95: bm25 {bm['p50']:.1f}/{bm['p95']:.1f} us, dense {d50:.2f}/{d95:.2f} ms, "
          f"fusion {fuse_ms * 1000:.1f} us/query")

    # --- Example queries ----------------------------------------------------
    def nd(name, qid):
        return results[name][0].get(qid, {}).get("ndcg_cut_10", 0.0)

    qtext = {q["_id"]: q["text"] for q in queries}
    qids = [q["_id"] for q in queries]
    bm25_wins = max(qids, key=lambda q: (nd("bm25", q) - nd("dense", q), -int(q)))
    dense_wins = max(qids, key=lambda q: (nd("dense", q) - nd("bm25", q), -int(q)))
    hybrid_fixes = max(qids, key=lambda q: (nd("hybrid", q) - max(nd("bm25", q), nd("dense", q)), -int(q)))

    def show(qid, label):
        lines = [f"### {label}: query {qid}", "", f"> {qtext[qid]}", "",
                 f"nDCG@10: BM25 {nd('bm25', qid):.3f} · dense {nd('dense', qid):.3f} · "
                 f"hybrid {nd('hybrid', qid):.3f}. Relevant docs: "
                 + ", ".join(f"`{d}`" for d, g in qrels[qid].items() if g > 0), ""]
        for name, run in [("BM25", bm25), ("Dense", dense), ("Hybrid", hybrid)]:
            lines.append(f"**{name} top 5:**")
            lines.append("")
            first = None
            for rank, (doc, _s) in enumerate(run[qid], start=1):
                if qrels[qid].get(doc, 0) > 0:
                    first = rank
                    break
            for rank, (doc, score) in enumerate(run[qid][:5], start=1):
                mark = " ✅ relevant" if qrels[qid].get(doc, 0) > 0 else ""
                title = titles[doc][:100].replace("|", "/")
                lines.append(f"{rank}. `{doc}` ({score:.4f}) {title}{mark}")
            lines.append("")
            lines.append(f"First relevant result at rank: {first if first else 'not in top 100'}")
            lines.append("")
        return "\n".join(lines)

    m = {k: v[1] for k, v in results.items()}
    md = [
        "# BM25 vs dense vs hybrid on BEIR SciFact",
        "",
        "Generated by `.venv/bin/python scripts/dense_baseline.py`. All three runs use depth 100",
        f"and are scored with pytrec_eval over the {len(queries)} judged test queries.",
        "",
        "| Method | nDCG@10 | Recall@100 | Query latency p50 (p95) |",
        "|---|---:|---:|---:|",
        f"| BM25 (this engine, C++) | {m['bm25']['ndcg_cut_10']:.4f} | {m['bm25']['recall_100']:.4f} | "
        f"{bm['p50']:.1f} µs ({bm['p95']:.1f} µs) |",
        f"| Dense: all-MiniLM-L6-v2, exact cosine | {m['dense']['ndcg_cut_10']:.4f} | "
        f"{m['dense']['recall_100']:.4f} | {d50:.2f} ms ({d95:.2f} ms) |",
        f"| Hybrid: RRF (k = {RRF_K}) of BM25 + dense | {m['hybrid']['ndcg_cut_10']:.4f} | "
        f"{m['hybrid']['recall_100']:.4f} | ≈ {d50:.2f} ms + {bm['p50']:.1f} µs + {fuse_ms * 1000:.0f} µs fusion |",
        "",
        "Latency notes:",
        "",
        "- All single-threaded on the same CPU. BM25 latency comes from `bm25_eval --bench -k 100`",
        "  run by this script (top-100, to match the dense depth), pinned to one core. Dense latency",
        "  is measured per query with batch size 1 and `torch.set_num_threads(1)`: query encoding",
        f"  (p50 {percentile(sorted(enc_ms), 50):.2f} ms) plus a dot product with the 5,183 × 384 corpus",
        "  matrix and top-100 selection. Corpus encoding is offline and not included.",
        "- The hybrid row is the sum of its parts (BM25 + dense + RRF fusion measured over all",
        "  queries). Running BM25 and dense in parallel would make it roughly the dense latency.",
        "- Dense latency is dominated by the transformer forward pass, not the vector search.",
        "",
    ]
    if encode_s is not None:
        md += [f"Corpus encoding took {encode_s:.1f} s on CPU with {os.cpu_count()} logical cores "
               "(one-off, cached in data/dense/).", ""]
    md += ["## Example queries", "",
           "Picked automatically: the largest per-query nDCG@10 gap in each direction, and the",
           "query where the hybrid beats the better of the two by the most.", "",
           show(bm25_wins, "BM25 wins"), show(dense_wins, "Dense wins"), show(hybrid_fixes, "Hybrid beats both")]
    with open(f"{args.out_dir}/comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out_dir}/comparison.md")


if __name__ == "__main__":
    main()
