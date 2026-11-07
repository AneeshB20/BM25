#include "bm25/inverted_index.hpp"

#include <vector>

#include <gtest/gtest.h>

using bm25::Document;
using bm25::InvertedIndex;
using bm25::Posting;
using bm25::Tokenizer;
using bm25::TokenizerOptions;
using Postings = std::vector<Posting>;

namespace {

// The hand-made corpus used throughout the index and scoring tests.
//   doc 0: the cat sat on the mat   (6 tokens)
//   doc 1: the dog sat              (3 tokens)
//   doc 2: cat cat dog              (3 tokens)
//   doc 3: a bird                   (2 tokens)
std::vector<Document> four_docs() {
    return {
        {"d0", "The cat sat on the mat."},
        {"d1", "The dog sat"},
        {"d2", "cat, cat, dog!"},
        {"d3", "A bird"},
    };
}

InvertedIndex build_raw() {
    InvertedIndex idx;
    idx.build(four_docs(), Tokenizer(TokenizerOptions{false, false}));
    return idx;
}

}  // namespace

TEST(InvertedIndex, ExactPostingsWithoutStopwords) {
    const InvertedIndex idx = build_raw();
    EXPECT_EQ(idx.postings("the"), (Postings{{0, 2}, {1, 1}}));
    EXPECT_EQ(idx.postings("cat"), (Postings{{0, 1}, {2, 2}}));
    EXPECT_EQ(idx.postings("sat"), (Postings{{0, 1}, {1, 1}}));
    EXPECT_EQ(idx.postings("dog"), (Postings{{1, 1}, {2, 1}}));
    EXPECT_EQ(idx.postings("on"), (Postings{{0, 1}}));
    EXPECT_EQ(idx.postings("mat"), (Postings{{0, 1}}));
    EXPECT_EQ(idx.postings("a"), (Postings{{3, 1}}));
    EXPECT_EQ(idx.postings("bird"), (Postings{{3, 1}}));
}

TEST(InvertedIndex, DocFrequencies) {
    const InvertedIndex idx = build_raw();
    EXPECT_EQ(idx.doc_freq("the"), 2u);
    EXPECT_EQ(idx.doc_freq("cat"), 2u);
    EXPECT_EQ(idx.doc_freq("mat"), 1u);
    EXPECT_EQ(idx.doc_freq("bird"), 1u);
    EXPECT_EQ(idx.doc_freq("unicorn"), 0u);
}

TEST(InvertedIndex, DocLengthsAndAvgdl) {
    const InvertedIndex idx = build_raw();
    ASSERT_EQ(idx.num_docs(), 4u);
    EXPECT_EQ(idx.doc_length(0), 6u);
    EXPECT_EQ(idx.doc_length(1), 3u);
    EXPECT_EQ(idx.doc_length(2), 3u);
    EXPECT_EQ(idx.doc_length(3), 2u);
    EXPECT_DOUBLE_EQ(idx.avgdl(), 14.0 / 4.0);
}

TEST(InvertedIndex, StatsCountVocabAndPostings) {
    const bm25::IndexStats s = build_raw().stats();
    EXPECT_EQ(s.num_docs, 4u);
    EXPECT_EQ(s.vocab_size, 8u);       // the cat sat on mat dog a bird
    EXPECT_EQ(s.total_postings, 12u);  // 2+2+2+2+1+1+1+1
    EXPECT_EQ(s.total_tokens, 14u);
    EXPECT_GT(s.memory_mb, 0.0);
}

TEST(InvertedIndex, StopwordsAreNotIndexedOrCounted) {
    // Defaults remove "the", "on", "a" (and stem, which changes nothing here).
    //   doc 0: cat sat mat (3)   doc 1: dog sat (2)   doc 2: cat cat dog (3)   doc 3: bird (1)
    InvertedIndex idx;
    idx.build(four_docs(), Tokenizer());
    EXPECT_TRUE(idx.postings("the").empty());
    EXPECT_TRUE(idx.postings("a").empty());
    EXPECT_EQ(idx.doc_length(0), 3u);
    EXPECT_EQ(idx.doc_length(3), 1u);
    EXPECT_DOUBLE_EQ(idx.avgdl(), 9.0 / 4.0);
    EXPECT_EQ(idx.stats().vocab_size, 5u);
    EXPECT_EQ(idx.stats().total_postings, 8u);
}

TEST(InvertedIndex, PostingListsAreSortedByDocId) {
    const InvertedIndex idx = build_raw();
    for (const char* term : {"the", "cat", "sat", "dog"}) {
        const Postings& p = idx.postings(term);
        for (std::size_t i = 1; i < p.size(); ++i) {
            EXPECT_LT(p[i - 1].doc_id, p[i].doc_id) << term;
        }
    }
}

TEST(InvertedIndex, KeepsExternalIds) {
    const InvertedIndex idx = build_raw();
    EXPECT_EQ(idx.external_id(0), "d0");
    EXPECT_EQ(idx.external_id(3), "d3");
}

TEST(InvertedIndex, EmptyDocumentHasLengthZero) {
    InvertedIndex idx;
    idx.build({{"x", "cat"}, {"y", "... !!"}}, Tokenizer(TokenizerOptions{false, false}));
    EXPECT_EQ(idx.doc_length(1), 0u);
    EXPECT_DOUBLE_EQ(idx.avgdl(), 0.5);
}

TEST(InvertedIndex, EmptyCorpus) {
    InvertedIndex idx;
    idx.build({}, Tokenizer());
    EXPECT_EQ(idx.num_docs(), 0u);
    EXPECT_DOUBLE_EQ(idx.avgdl(), 0.0);
    EXPECT_TRUE(idx.postings("cat").empty());
}

TEST(InvertedIndex, RebuildReplacesContents) {
    InvertedIndex idx = build_raw();
    idx.build({{"z", "zebra"}}, Tokenizer(TokenizerOptions{false, false}));
    EXPECT_EQ(idx.num_docs(), 1u);
    EXPECT_TRUE(idx.postings("cat").empty());
    EXPECT_EQ(idx.postings("zebra"), (Postings{{0, 1}}));
}
