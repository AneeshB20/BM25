#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "bm25/bm25_scorer.hpp"
#include "bm25/inverted_index.hpp"
#include "bm25/tokenizer.hpp"

namespace bm25 {

struct SearchResult {
    std::uint32_t doc_id;
    double score;
};

// Runs ranked retrieval over an InvertedIndex.
// The tokenizer MUST have the same options as the one the index was built
// with, otherwise query terms will not match index terms.
class Searcher {
public:
    Searcher(const InvertedIndex& index, const Tokenizer& tokenizer, Bm25Params params = {});

    // Top-k documents by BM25 score, best first. Equal scores are ordered by
    // ascending doc id so results are deterministic. Returns fewer than k
    // results if fewer documents match, and nothing for k == 0 or a query
    // with no indexed terms. A term repeated in the query counts once.
    std::vector<SearchResult> search(std::string_view query, std::size_t k) const;

    const InvertedIndex& index() const { return index_; }

private:
    const InvertedIndex& index_;
    const Tokenizer& tokenizer_;
    Bm25Scorer scorer_;
    // scorer_.length_norm(doc_length(d)) for every document d, computed once in
    // the constructor so the scoring loop does no per-posting division.
    std::vector<double> length_norms_;
};

}  // namespace bm25
