#pragma once

#include <cstddef>
#include <cstdint>

namespace bm25 {

struct Bm25Params {
    double k1 = 0.9;  // term-frequency saturation (Anserini BEIR default)
    double b = 0.4;   // length-normalisation strength (Anserini BEIR default)
};

// The BM25 formula (Lucene variant), exactly as in CLAUDE.md:
//   idf(t)  = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
//   score  += idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
// It knows nothing about the index: callers pass in the statistics.
class Bm25Scorer {
public:
    Bm25Scorer(std::size_t num_docs, double avgdl, Bm25Params params = {});

    // Inverse document frequency of a term that occurs in `doc_freq` documents.
    // Always > 0 for doc_freq <= N (the "1 +" inside the log guarantees it).
    double idf(std::uint32_t doc_freq) const;

    // Contribution of one query term to one document's score.
    double term_score(double idf, std::uint32_t tf, std::uint32_t doc_len) const;

    // The document-dependent part of the denominator, k1 * (1 - b + b * dl / avgdl).
    // It does not depend on the query, so a Searcher computes it once per document.
    double length_norm(std::uint32_t doc_len) const;

    // term_score with length_norm(doc_len) already computed. Produces exactly
    // the same double as term_score (same operations in the same order).
    double term_score_normed(double idf, std::uint32_t tf, double length_norm) const {
        const double f = static_cast<double>(tf);
        return idf * f * (params_.k1 + 1.0) / (f + length_norm);
    }

    const Bm25Params& params() const { return params_; }

private:
    double num_docs_;
    double avgdl_;
    Bm25Params params_;
};

}  // namespace bm25
