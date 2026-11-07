#include "bm25/bm25_scorer.hpp"

#include <cmath>

#include <gtest/gtest.h>

using bm25::Bm25Params;
using bm25::Bm25Scorer;

// Statistics of the 4-document test corpus without stopword removal
// (see inverted_index_test.cpp): N = 4, avgdl = 14 / 4 = 3.5.
// k1 = 0.9, b = 0.4 throughout unless stated.
constexpr double kTol = 1e-6;

TEST(Bm25Scorer, Idf) {
    const Bm25Scorer s(4, 3.5);
    // n = 2: ln(1 + (4 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 2.5/2.5) = ln 2
    EXPECT_NEAR(s.idf(2), 0.693147180560, kTol);
    // n = 1: ln(1 + 3.5 / 1.5) = ln(10/3)
    EXPECT_NEAR(s.idf(1), 1.203972804326, kTol);
    // n = 4 (in every doc): ln(1 + 0.5 / 4.5) = ln(10/9), small but still positive
    EXPECT_NEAR(s.idf(4), 0.105360515658, kTol);
    EXPECT_GT(s.idf(4), 0.0);
}

TEST(Bm25Scorer, TermScoreByHand) {
    const Bm25Scorer s(4, 3.5);
    const double ln2 = std::log(2.0);
    // "cat" in doc 0: tf = 1, dl = 6
    //   1 - b + b*dl/avgdl = 0.6 + 0.4*6/3.5 = 9/7
    //   denominator = 1 + 0.9 * 9/7 = 151/70
    //   numerator   = 1 * 1.9
    //   score = ln2 * 1.9 * 70/151 = ln2 * 133/151 = 0.610520364334
    EXPECT_NEAR(s.term_score(ln2, 1, 6), 0.610520364334, kTol);
    // "cat" in doc 2: tf = 2, dl = 3
    //   0.6 + 0.4*3/3.5 = 33/35;  denominator = 2 + 0.9*33/35 = 99.7/35
    //   score = ln2 * 3.8 * 35/99.7 = ln2 * 133/99.7 = 0.924659729333
    EXPECT_NEAR(s.term_score(ln2, 2, 3), 0.924659729333, kTol);
    // "bird" in doc 3: idf = ln(10/3), tf = 1, dl = 2
    //   0.6 + 0.4*2/3.5 = 29/35;  denominator = 1 + 0.9*29/35 = 61.1/35
    //   score = ln(10/3) * 66.5/61.1 = 1.310379566083
    EXPECT_NEAR(s.term_score(std::log(10.0 / 3.0), 1, 2), 1.310379566083, kTol);
}

TEST(Bm25Scorer, TfSaturates) {
    // As tf grows the score approaches idf * (k1 + 1) but never exceeds it.
    const Bm25Scorer s(4, 3.5);
    const double s1 = s.term_score(1.0, 1, 4);
    const double s10 = s.term_score(1.0, 10, 4);
    const double s1000 = s.term_score(1.0, 1000, 4);
    EXPECT_LT(s1, s10);
    EXPECT_LT(s10, s1000);
    EXPECT_LT(s1000, 1.9);
    EXPECT_GT(s1000, 1.89);
}

TEST(Bm25Scorer, BZeroIgnoresLength) {
    const Bm25Scorer s(4, 3.5, Bm25Params{0.9, 0.0});
    EXPECT_DOUBLE_EQ(s.term_score(1.0, 1, 1), s.term_score(1.0, 1, 100));
}

TEST(Bm25Scorer, LongerDocsScoreLowerWhenBPositive) {
    const Bm25Scorer s(4, 3.5);
    EXPECT_GT(s.term_score(1.0, 1, 2), s.term_score(1.0, 1, 20));
}

TEST(Bm25Scorer, KeepsParams) {
    const Bm25Scorer s(10, 5.0, Bm25Params{1.2, 0.75});
    EXPECT_DOUBLE_EQ(s.params().k1, 1.2);
    EXPECT_DOUBLE_EQ(s.params().b, 0.75);
}
