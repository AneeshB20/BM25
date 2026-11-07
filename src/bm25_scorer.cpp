#include "bm25/bm25_scorer.hpp"

#include <cmath>

namespace bm25 {

Bm25Scorer::Bm25Scorer(std::size_t num_docs, double avgdl, Bm25Params params)
    : num_docs_(static_cast<double>(num_docs)), avgdl_(avgdl), params_(params) {}

double Bm25Scorer::idf(std::uint32_t doc_freq) const {
    const double n = static_cast<double>(doc_freq);
    return std::log(1.0 + (num_docs_ - n + 0.5) / (n + 0.5));
}

double Bm25Scorer::term_score(double idf, std::uint32_t tf, std::uint32_t doc_len) const {
    const double k1 = params_.k1;
    const double b = params_.b;
    const double f = static_cast<double>(tf);
    const double dl = static_cast<double>(doc_len);
    // avgdl is 0 only for a corpus with no tokens at all, where no posting
    // exists and this is never called; guard anyway rather than divide by 0.
    const double length_norm = avgdl_ > 0.0 ? (1.0 - b + b * dl / avgdl_) : (1.0 - b);
    // Same operation order as the CLAUDE.md formula and scripts/reference_bm25.py,
    // so C++ and Python produce bit-identical doubles.
    return idf * f * (k1 + 1.0) / (f + k1 * length_norm);
}

}  // namespace bm25
