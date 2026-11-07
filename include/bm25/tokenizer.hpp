#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "bm25/porter_stemmer.hpp"

namespace bm25 {

struct TokenizerOptions {
    bool remove_stopwords = true;  // drop words in the Lucene English stopword list
    bool stem = true;              // apply the Porter stemmer to each surviving token
};

// Turns raw text into index terms:
//   1. lowercase ASCII letters
//   2. split on every character that is not [a-z0-9] (so non-ASCII bytes split too)
//   3. drop empty tokens
//   4. optionally drop stopwords (checked on the unstemmed token, as Lucene does)
//   5. optionally Porter-stem
// The same Tokenizer must be used for documents and queries.
class Tokenizer {
public:
    explicit Tokenizer(TokenizerOptions options = {});

    std::vector<std::string> tokenize(std::string_view text) const;

    const TokenizerOptions& options() const { return options_; }

    // True if `word` (already lowercase) is in the 33-word Lucene English stopword list.
    static bool is_stopword(std::string_view word);
    static std::size_t stopword_count();

private:
    TokenizerOptions options_;
    PorterStemmer stemmer_;
};

}  // namespace bm25
