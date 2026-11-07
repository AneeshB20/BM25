#include "bm25/searcher.hpp"

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

using bm25::Document;
using bm25::InvertedIndex;
using bm25::SearchResult;
using bm25::Searcher;
using bm25::Tokenizer;
using bm25::TokenizerOptions;

namespace {

constexpr double kTol = 1e-6;

// Same corpus as inverted_index_test.cpp, indexed without stopwords or stemming.
//   doc 0: the cat sat on the mat   (6 tokens)
//   doc 1: the dog sat              (3 tokens)
//   doc 2: cat cat dog              (3 tokens)
//   doc 3: a bird                   (2 tokens)
// N = 4, avgdl = 3.5, k1 = 0.9, b = 0.4.
class SearcherTest : public ::testing::Test {
protected:
    SearcherTest() : tokenizer_(TokenizerOptions{false, false}) {
        index_.build({{"d0", "The cat sat on the mat."},
                      {"d1", "The dog sat"},
                      {"d2", "cat, cat, dog!"},
                      {"d3", "A bird"}},
                     tokenizer_);
    }

    std::vector<SearchResult> search(const char* q, std::size_t k = 10) const {
        return Searcher(index_, tokenizer_).search(q, k);
    }

    Tokenizer tokenizer_;
    InvertedIndex index_;
};

}  // namespace

TEST_F(SearcherTest, SingleTermByHand) {
    // "cat": n = 2 so idf = ln 2.
    //   doc 2: tf 2, dl 3 -> ln2 * 133/99.7 = 0.924659729333
    //   doc 0: tf 1, dl 6 -> ln2 * 133/151  = 0.610520364334
    const auto r = search("cat");
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0].doc_id, 2u);
    EXPECT_NEAR(r[0].score, 0.924659729333, kTol);
    EXPECT_EQ(r[1].doc_id, 0u);
    EXPECT_NEAR(r[1].score, 0.610520364334, kTol);
}

TEST_F(SearcherTest, TwoTermsSumByHand) {
    // "cat dog": both have n = 2, idf = ln 2.
    //   "dog" in doc 1 or doc 2: tf 1, dl 3 -> denominator 1 + 0.9*33/35 = 64.7/35
    //       -> ln2 * 66.5/64.7 = 0.712431027933
    //   doc 2: cat 0.924659729333 + dog 0.712431027933 = 1.637090757266
    //   doc 1: dog only = 0.712431027933
    //   doc 0: cat only = 0.610520364334
    const auto r = search("cat dog");
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0].doc_id, 2u);
    EXPECT_NEAR(r[0].score, 1.637090757266, kTol);
    EXPECT_EQ(r[1].doc_id, 1u);
    EXPECT_NEAR(r[1].score, 0.712431027933, kTol);
    EXPECT_EQ(r[2].doc_id, 0u);
    EXPECT_NEAR(r[2].score, 0.610520364334, kTol);
}

TEST_F(SearcherTest, RareTermByHand) {
    // "bird": n = 1, idf = ln(10/3); doc 3: tf 1, dl 2 -> 1.310379566083
    const auto r = search("bird");
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].doc_id, 3u);
    EXPECT_NEAR(r[0].score, 1.310379566083, kTol);
}

TEST_F(SearcherTest, HigherTfButLongerDoc) {
    // "the": doc 0 has tf 2 but dl 6; doc 1 has tf 1 and dl 3.
    //   doc 0: 0.6 + 0.4*6/3.5 = 9/7; denominator 2 + 0.9*9/7 = 22.1/7
    //          -> ln2 * 3.8*7/22.1 = ln2 * 26.6/22.1 = 0.834285746737
    //   doc 1: ln2 * 66.5/64.7 = 0.712431027933
    const auto r = search("the");
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0].doc_id, 0u);
    EXPECT_NEAR(r[0].score, 0.834285746737, kTol);
    EXPECT_EQ(r[1].doc_id, 1u);
    EXPECT_NEAR(r[1].score, 0.712431027933, kTol);
}

TEST_F(SearcherTest, RepeatedQueryTermCountsOnce) {
    const auto once = search("cat dog");
    const auto repeated = search("cat CAT dog cat");
    ASSERT_EQ(once.size(), repeated.size());
    for (std::size_t i = 0; i < once.size(); ++i) {
        EXPECT_EQ(once[i].doc_id, repeated[i].doc_id);
        EXPECT_DOUBLE_EQ(once[i].score, repeated[i].score);
    }
}

TEST_F(SearcherTest, KLimitsResults) {
    const auto r = search("cat dog", 1);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].doc_id, 2u);
}

TEST_F(SearcherTest, KLargerThanMatches) {
    EXPECT_EQ(search("cat dog", 100).size(), 3u);
}

TEST_F(SearcherTest, UnknownTermsReturnEmpty) {
    EXPECT_TRUE(search("unicorn zebra").empty());
}

TEST_F(SearcherTest, UnknownTermsAreIgnoredAlongsideKnownOnes) {
    const auto r = search("unicorn bird");
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].doc_id, 3u);
}

TEST_F(SearcherTest, EmptyQueryReturnsEmpty) {
    EXPECT_TRUE(search("").empty());
    EXPECT_TRUE(search("  ?!  ").empty());
}

TEST_F(SearcherTest, KZeroReturnsEmpty) {
    EXPECT_TRUE(search("cat", 0).empty());
}

TEST_F(SearcherTest, ResultsAreSortedDescending) {
    const auto r = search("the cat sat on a mat dog bird");
    for (std::size_t i = 1; i < r.size(); ++i) {
        EXPECT_GE(r[i - 1].score, r[i].score);
    }
}

TEST(Searcher, TiesBrokenByLowerDocId) {
    // Identical documents get identical scores; order must still be stable.
    const Tokenizer tok(TokenizerOptions{false, false});
    InvertedIndex idx;
    idx.build({{"a", "x y"}, {"b", "z"}, {"c", "x y"}, {"d", "x y"}}, tok);
    const auto r = Searcher(idx, tok).search("x", 2);
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0].doc_id, 0u);
    EXPECT_EQ(r[1].doc_id, 2u);
    EXPECT_DOUBLE_EQ(r[0].score, r[1].score);
}

TEST(Searcher, OnlyStopwordQueryReturnsEmpty) {
    const Tokenizer tok;  // stopwords on
    InvertedIndex idx;
    idx.build({{"a", "the cat"}}, tok);
    EXPECT_TRUE(Searcher(idx, tok).search("the of and", 10).empty());
}

TEST(Searcher, StemmedQueryMatchesStemmedDocs) {
    const Tokenizer tok;
    InvertedIndex idx;
    idx.build({{"a", "Infections were observed"}, {"b", "Bone density"}}, tok);
    const auto r = Searcher(idx, tok).search("infected", 10);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(idx.external_id(r[0].doc_id), "a");
}

TEST(Searcher, RejectsInvalidParams) {
    const Tokenizer tok;
    InvertedIndex idx;
    idx.build({{"a", "cat"}}, tok);
    EXPECT_THROW(Searcher(idx, tok, bm25::Bm25Params{-1.0, 0.4}), std::invalid_argument);
    EXPECT_THROW(Searcher(idx, tok, bm25::Bm25Params{0.9, 2.0}), std::invalid_argument);
    EXPECT_THROW(Searcher(idx, tok, bm25::Bm25Params{0.9, -0.1}), std::invalid_argument);
    EXPECT_NO_THROW(Searcher(idx, tok, bm25::Bm25Params{0.0, 0.0}));
    EXPECT_NO_THROW(Searcher(idx, tok, bm25::Bm25Params{2.0, 1.0}));
}
