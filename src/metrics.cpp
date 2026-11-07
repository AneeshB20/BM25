#include "bm25/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace bm25 {

namespace {

int grade_of(const std::map<std::string, int>& judged, const std::string& doc) {
    auto it = judged.find(doc);
    return it == judged.end() ? 0 : it->second;
}

}  // namespace

double Metrics::ndcg_at_k(const std::vector<std::string>& ranked,
                          const std::map<std::string, int>& judged, std::size_t k) {
    double dcg = 0.0;
    const std::size_t depth = std::min(k, ranked.size());
    for (std::size_t i = 0; i < depth; ++i) {
        const int g = grade_of(judged, ranked[i]);
        if (g > 0) {
            // rank r = i + 1, discount log2(r + 1) = log2(i + 2)
            dcg += static_cast<double>(g) / std::log2(static_cast<double>(i) + 2.0);
        }
    }

    std::vector<int> ideal;
    for (const auto& [doc, g] : judged) {
        if (g > 0) ideal.push_back(g);
    }
    std::sort(ideal.begin(), ideal.end(), std::greater<int>());
    double idcg = 0.0;
    for (std::size_t i = 0; i < std::min(k, ideal.size()); ++i) {
        idcg += static_cast<double>(ideal[i]) / std::log2(static_cast<double>(i) + 2.0);
    }
    return idcg > 0.0 ? dcg / idcg : 0.0;
}

double Metrics::recall_at_k(const std::vector<std::string>& ranked,
                            const std::map<std::string, int>& judged, std::size_t k) {
    std::size_t relevant = 0;
    for (const auto& [doc, g] : judged) {
        if (g > 0) ++relevant;
    }
    if (relevant == 0) {
        return 0.0;
    }
    std::size_t found = 0;
    const std::size_t depth = std::min(k, ranked.size());
    for (std::size_t i = 0; i < depth; ++i) {
        if (grade_of(judged, ranked[i]) > 0) ++found;
    }
    return static_cast<double>(found) / static_cast<double>(relevant);
}

}  // namespace bm25
