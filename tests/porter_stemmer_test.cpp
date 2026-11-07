#include "bm25/porter_stemmer.hpp"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using bm25::PorterStemmer;

namespace {

struct Pair {
    const char* word;
    const char* stem;
};

void expect_stems(const std::vector<Pair>& pairs) {
    const PorterStemmer s;
    for (const Pair& p : pairs) {
        EXPECT_EQ(s.stem(p.word), p.stem) << "word: " << p.word;
    }
}

}  // namespace

// Examples from Porter's 1980 paper, run through the full algorithm.
TEST(PorterStemmer, PaperExamples) {
    expect_stems({
        {"caresses", "caress"},
        {"ponies", "poni"},
        {"relational", "relat"},
        {"conditional", "condit"},
        {"generalization", "gener"},
        {"hopefulness", "hope"},
    });
}

// Pairs taken from the official test vocabulary
// (tartarus.org/martin/PorterStemmer/voc.txt and output.txt), grouped by the
// step they mainly exercise.
TEST(PorterStemmer, Step1Plurals) {
    expect_stems({
        {"blesses", "bless"},
        {"ties", "ti"},
        {"abilities", "abil"},
        {"cats", "cat"},
    });
}

TEST(PorterStemmer, Step1bEdIng) {
    expect_stems({
        {"feed", "feed"},
        {"agreed", "agre"},
        {"bled", "bled"},
        {"sing", "sing"},
        {"troubled", "troubl"},
        {"tanned", "tan"},
        {"falling", "fall"},
        {"hissing", "hiss"},
        {"failing", "fail"},
        {"abetting", "abet"},
        {"controlling", "control"},
    });
}

TEST(PorterStemmer, Step1cY) {
    expect_stems({
        {"happy", "happi"},
        {"sky", "sky"},
        {"knightly", "knightli"},
    });
}

TEST(PorterStemmer, Step2) {
    expect_stems({
        {"rational", "ration"},
        {"traditional", "tradit"},
        {"temporizer", "tempor"},
        {"possibly", "possibl"},
        {"accommodation", "accommod"},
        {"arbitrator", "arbitr"},
        {"forgiveness", "forgiv"},
        {"fearfulness", "fear"},
        {"covetousness", "covet"},
    });
}

TEST(PorterStemmer, Step3) {
    expect_stems({
        {"certificate", "certif"},
        {"comparative", "compar"},
        {"naturalize", "natur"},
        {"alphabetical", "alphabet"},
        {"hopeful", "hope"},
        {"goodness", "good"},
    });
}

TEST(PorterStemmer, Step4) {
    expect_stems({
        {"allowance", "allow"},
        {"inference", "infer"},
        {"defensible", "defens"},
        {"dependent", "depend"},
        {"adoption", "adopt"},
        {"abatement", "abat"},
        {"barbarism", "barbar"},
        {"abortive", "abort"},
        {"anatomize", "anatom"},
        {"abundant", "abund"},
        {"absolver", "absolv"},
        {"accidental", "accident"},
        {"universal", "univers"},
        {"university", "univers"},
        {"generous", "gener"},
    });
}

TEST(PorterStemmer, Step5) {
    expect_stems({
        {"rate", "rate"},
        {"cease", "ceas"},
        {"roll", "roll"},
        {"appall", "appal"},
        {"audible", "audibl"},
        {"absence", "absenc"},
    });
}

TEST(PorterStemmer, ShortWordsUnchanged) {
    expect_stems({{"", ""}, {"a", "a"}, {"is", "is"}, {"as", "as"}});
}

// Runs the whole official vocabulary (23,531 words) when get_data.sh has
// downloaded it into data/porter/. Skipped otherwise.
TEST(PorterStemmer, OfficialVocabulary) {
    const std::string dir = std::string(BM25_SOURCE_DIR) + "/data/porter/";
    std::ifstream voc(dir + "voc.txt");
    std::ifstream out(dir + "output.txt");
    if (!voc || !out) {
        GTEST_SKIP() << "data/porter/voc.txt not found; run scripts/get_data.sh";
    }
    const PorterStemmer s;
    std::string word, expected;
    std::size_t total = 0, mismatches = 0;
    while (std::getline(voc, word) && std::getline(out, expected)) {
        ++total;
        if (s.stem(word) != expected) {
            if (++mismatches <= 10) {
                ADD_FAILURE() << word << " -> " << s.stem(word) << ", expected " << expected;
            }
        }
    }
    EXPECT_EQ(total, 23531u);
    EXPECT_EQ(mismatches, 0u);
}
