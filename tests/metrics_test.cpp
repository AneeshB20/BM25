#include "bm25/metrics.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using bm25::Metrics;
using Ranked = std::vector<std::string>;
using Judged = std::map<std::string, int>;

constexpr double kTol = 1e-9;

TEST(Metrics, PerfectRankingHasNdcgOne) {
    const Judged j{{"a", 1}, {"b", 1}};
    EXPECT_NEAR(Metrics::ndcg_at_k({"a", "b", "x"}, j, 10), 1.0, kTol);
}

TEST(Metrics, NdcgByHand) {
    // Relevant: a, b. Ranking: x a y b.
    //   DCG  = 1/log2(3) + 1/log2(5) = 0.630929754 + 0.430676558 = 1.061606312
    //   IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.630929754           = 1.630929754
    //   nDCG = 0.650921...
    const Judged j{{"a", 1}, {"b", 1}};
    const double expected = (1 / std::log2(3.0) + 1 / std::log2(5.0)) / (1 + 1 / std::log2(3.0));
    EXPECT_NEAR(Metrics::ndcg_at_k({"x", "a", "y", "b"}, j, 10), expected, kTol);
    EXPECT_NEAR(expected, 0.650921, 1e-6);
}

TEST(Metrics, NdcgCutoffIgnoresLowerRanks) {
    // With k = 2 only "x a" counts: DCG = 1/log2(3); IDCG@2 = 1 + 1/log2(3).
    const Judged j{{"a", 1}, {"b", 1}};
    const double expected = (1 / std::log2(3.0)) / (1 + 1 / std::log2(3.0));
    EXPECT_NEAR(Metrics::ndcg_at_k({"x", "a", "y", "b"}, j, 2), expected, kTol);
}

TEST(Metrics, NdcgGradedLinearGain) {
    // Grades a=2, b=1. Ranking b a: DCG = 1 + 2/log2(3); ideal a b: IDCG = 2 + 1/log2(3).
    const Judged j{{"a", 2}, {"b", 1}};
    const double expected = (1 + 2 / std::log2(3.0)) / (2 + 1 / std::log2(3.0));
    EXPECT_NEAR(Metrics::ndcg_at_k({"b", "a"}, j, 10), expected, kTol);
}

TEST(Metrics, IdealUsesAllRelevantNotJustRetrieved) {
    // Three relevant docs, only one retrieved at rank 1.
    //   DCG = 1; IDCG = 1 + 1/log2(3) + 1/log2(4) = 2.130929754
    const Judged j{{"a", 1}, {"b", 1}, {"c", 1}};
    const double expected = 1.0 / (1 + 1 / std::log2(3.0) + 0.5);
    EXPECT_NEAR(Metrics::ndcg_at_k({"a"}, j, 10), expected, kTol);
}

TEST(Metrics, ZeroGradeIsNotRelevant) {
    const Judged j{{"a", 0}, {"b", 1}};
    EXPECT_NEAR(Metrics::ndcg_at_k({"a"}, j, 10), 0.0, kTol);
    EXPECT_NEAR(Metrics::recall_at_k({"a", "b"}, j, 10), 1.0, kTol);
}

TEST(Metrics, NoRelevantDocsGivesZero) {
    EXPECT_EQ(Metrics::ndcg_at_k({"a"}, Judged{}, 10), 0.0);
    EXPECT_EQ(Metrics::recall_at_k({"a"}, Judged{}, 10), 0.0);
}

TEST(Metrics, EmptyRankingGivesZero) {
    const Judged j{{"a", 1}};
    EXPECT_EQ(Metrics::ndcg_at_k({}, j, 10), 0.0);
    EXPECT_EQ(Metrics::recall_at_k({}, j, 10), 0.0);
}

TEST(Metrics, RecallByHand) {
    // Relevant a b c d; top-3 = a x c -> 2/4.
    const Judged j{{"a", 1}, {"b", 1}, {"c", 1}, {"d", 1}};
    const Ranked r{"a", "x", "c", "b"};
    EXPECT_NEAR(Metrics::recall_at_k(r, j, 3), 0.5, kTol);
    EXPECT_NEAR(Metrics::recall_at_k(r, j, 4), 0.75, kTol);
    EXPECT_NEAR(Metrics::recall_at_k(r, j, 100), 0.75, kTol);
}
