#include "bm25/tokenizer.hpp"

#include <string>
#include <vector>

#include <gtest/gtest.h>

using bm25::Tokenizer;
using bm25::TokenizerOptions;
using Tokens = std::vector<std::string>;

namespace {

// Plain splitting only: no stopwords, no stemming.
Tokenizer raw() {
    return Tokenizer(TokenizerOptions{false, false});
}

}  // namespace

TEST(Tokenizer, SplitsOnPunctuation) {
    EXPECT_EQ(raw().tokenize("Hello, world! foo-bar_baz (qux)."),
              (Tokens{"hello", "world", "foo", "bar", "baz", "qux"}));
}

TEST(Tokenizer, KeepsDigitsAndSplitsDecimals) {
    EXPECT_EQ(raw().tokenize("COVID-19 rose 3.5x in 2020s"),
              (Tokens{"covid", "19", "rose", "3", "5x", "in", "2020s"}));
}

TEST(Tokenizer, LowercasesMixedCase) {
    EXPECT_EQ(raw().tokenize("BrCA1 MixedCase TNF"), (Tokens{"brca1", "mixedcase", "tnf"}));
}

TEST(Tokenizer, EmptyStringGivesNoTokens) {
    EXPECT_TRUE(raw().tokenize("").empty());
    EXPECT_TRUE(Tokenizer().tokenize("").empty());
}

TEST(Tokenizer, OnlySeparatorsGivesNoTokens) {
    EXPECT_TRUE(raw().tokenize("  ,.;!? -- \t\n ").empty());
}

TEST(Tokenizer, NonAsciiBytesAreSeparators) {
    // "naïve β-cell": the UTF-8 bytes of ï and β are not [a-z0-9].
    EXPECT_EQ(raw().tokenize("na\xC3\xAFve \xCE\xB2-cell"), (Tokens{"na", "ve", "cell"}));
}

TEST(Tokenizer, OnlyStopwordsGivesNoTokens) {
    const Tokenizer t(TokenizerOptions{true, false});
    EXPECT_TRUE(t.tokenize("The and of IT is, to THIS").empty());
}

TEST(Tokenizer, StopwordRemovalKeepsOtherWords) {
    const Tokenizer t(TokenizerOptions{true, false});
    EXPECT_EQ(t.tokenize("The role of the immune system"),
              (Tokens{"role", "immune", "system"}));
}

TEST(Tokenizer, StemmingFlagStemsTokens) {
    const Tokenizer t(TokenizerOptions{false, true});
    EXPECT_EQ(t.tokenize("Running relational ponies"), (Tokens{"run", "relat", "poni"}));
}

TEST(Tokenizer, StopwordsCheckedBeforeStemming) {
    // "this" is a stopword; its stem "thi" is not. Removal must look at the
    // unstemmed form, otherwise "this" would survive as "thi".
    const Tokenizer t;  // defaults: stopwords + stemming
    EXPECT_EQ(t.tokenize("this patients"), (Tokens{"patient"}));
}

TEST(Tokenizer, DefaultsEnableBothOptions) {
    const Tokenizer t;
    EXPECT_TRUE(t.options().remove_stopwords);
    EXPECT_TRUE(t.options().stem);
    EXPECT_EQ(t.tokenize("The effects of vitamin D on bones"),
              (Tokens{"effect", "vitamin", "d", "bone"}));
}

TEST(Tokenizer, StopwordListIsLuceneEnglish) {
    EXPECT_EQ(Tokenizer::stopword_count(), 33u);
    for (const char* w : {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
                          "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
                          "that", "the", "their", "then", "there", "these", "they", "this", "to",
                          "was", "will", "with"}) {
        EXPECT_TRUE(Tokenizer::is_stopword(w)) << w;
    }
    EXPECT_FALSE(Tokenizer::is_stopword("vitamin"));
    EXPECT_FALSE(Tokenizer::is_stopword("we"));  // not in the Lucene list
}
