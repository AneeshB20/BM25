#include "bm25/tokenizer.hpp"

#include <algorithm>
#include <array>

namespace bm25 {

namespace {

// Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, which Anserini uses by
// default. Kept sorted so lookup can be a binary search.
constexpr std::array<std::string_view, 33> kStopwords = {
    "a",    "an",   "and",   "are",   "as",    "at",   "be",    "but",  "by",
    "for",  "if",   "in",    "into",  "is",    "it",   "no",    "not",  "of",
    "on",   "or",   "such",  "that",  "the",   "their", "then", "there", "these",
    "they", "this", "to",    "was",   "will",  "with",
};

// ASCII-only checks on purpose: std::isalnum/std::tolower depend on the C
// locale and are undefined for negative char values (bytes >= 0x80).
bool is_ascii_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

}  // namespace

Tokenizer::Tokenizer(TokenizerOptions options) : options_(options) {}

bool Tokenizer::is_stopword(std::string_view word) {
    return std::binary_search(kStopwords.begin(), kStopwords.end(), word);
}

std::size_t Tokenizer::stopword_count() {
    return kStopwords.size();
}

std::vector<std::string> Tokenizer::tokenize(std::string_view text) const {
    std::vector<std::string> tokens;
    std::string current;

    auto flush = [&]() {
        if (current.empty()) {
            return;
        }
        if (!(options_.remove_stopwords && is_stopword(current))) {
            tokens.push_back(options_.stem ? stemmer_.stem(current) : current);
        }
        current.clear();
    };

    for (char c : text) {
        if (is_ascii_alnum(c)) {
            current.push_back(ascii_lower(c));
        } else {
            flush();
        }
    }
    flush();
    return tokens;
}

}  // namespace bm25
