#include "bm25/searcher.hpp"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>

namespace bm25 {

namespace {

// Strict "a ranks above b": higher score, then lower doc id.
struct RanksHigher {
    bool operator()(const SearchResult& a, const SearchResult& b) const {
        if (a.score != b.score) return a.score > b.score;
        return a.doc_id < b.doc_id;
    }
};

}  // namespace

Searcher::Searcher(const InvertedIndex& index, const Tokenizer& tokenizer, Bm25Params params)
    : index_(index), tokenizer_(tokenizer), scorer_(index.num_docs(), index.avgdl(), params) {
    length_norms_.reserve(index_.num_docs());
    for (std::uint32_t d = 0; d < index_.num_docs(); ++d) {
        length_norms_.push_back(scorer_.length_norm(index_.doc_length(d)));
    }
}

std::vector<SearchResult> Searcher::search(std::string_view query, std::size_t k) const {
    if (k == 0) {
        return {};
    }

    // 1. Tokenize exactly as documents were, then drop repeated terms while
    //    keeping first-occurrence order (the order scores are summed in).
    std::vector<std::string> terms;
    {
        std::unordered_set<std::string> seen;
        for (std::string& t : tokenizer_.tokenize(query)) {
            if (seen.insert(t).second) {
                terms.push_back(std::move(t));
            }
        }
    }

    // 2. Term-at-a-time scoring: walk each posting list and add the term's
    //    contribution into a dense per-document accumulator (one double per
    //    document, indexed by doc id), remembering which documents were
    //    touched so step 3 only visits those. Every matching document gets a
    //    strictly positive score (idf > 0, tf >= 1), so "== 0.0" means "not
    //    touched yet".
    std::vector<double> acc(index_.num_docs(), 0.0);
    std::vector<std::uint32_t> touched;
    for (const std::string& term : terms) {
        const std::vector<Posting>& plist = index_.postings(term);
        if (plist.empty()) {
            continue;
        }
        const double idf = scorer_.idf(static_cast<std::uint32_t>(plist.size()));
        for (const Posting& p : plist) {
            double& slot = acc[p.doc_id];
            if (slot == 0.0) {
                touched.push_back(p.doc_id);
            }
            slot += scorer_.term_score_normed(idf, p.tf, length_norms_[p.doc_id]);
        }
    }

    // 3. Top-k with a size-k min-heap. With RanksHigher as the comparator,
    //    std::priority_queue keeps the *worst* kept result on top, so each
    //    candidate is compared only against the current k-th best.
    //    O(M log k) for M matching documents, instead of O(M log M) for a sort.
    std::priority_queue<SearchResult, std::vector<SearchResult>, RanksHigher> heap;
    const RanksHigher ranks_higher;
    for (const std::uint32_t doc_id : touched) {
        const SearchResult cand{doc_id, acc[doc_id]};
        if (heap.size() < k) {
            heap.push(cand);
        } else if (ranks_higher(cand, heap.top())) {
            heap.pop();
            heap.push(cand);
        }
    }

    // 4. The heap pops worst-first; reverse into best-first order.
    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

}  // namespace bm25
