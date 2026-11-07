#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace bm25 {

// Ranking-quality metrics for one query, computed the way trec_eval does.
//   ranked: retrieved doc ids, best first.
//   judged: doc id -> relevance grade for this query (grade <= 0 = not relevant).
class Metrics {
public:
    // Normalised Discounted Cumulative Gain at cutoff k, with linear gain
    // (gain = grade) and a log2(rank + 1) discount:
    //   DCG@k  = sum_{i=1..k} grade(doc_i) / log2(i + 1)
    //   nDCG@k = DCG@k / IDCG@k, IDCG = DCG of the ideal ordering of all judged
    //            relevant docs. Returns 0 if the query has no relevant docs.
    static double ndcg_at_k(const std::vector<std::string>& ranked,
                            const std::map<std::string, int>& judged, std::size_t k);

    // Fraction of the query's relevant docs that appear in the top k.
    // Returns 0 if the query has no relevant docs.
    static double recall_at_k(const std::vector<std::string>& ranked,
                              const std::map<std::string, int>& judged, std::size_t k);
};

}  // namespace bm25
