#include "bm25/tokenizer.hpp"

namespace bm25 {

// Stub: the tests are written first and must fail against this.
Tokenizer::Tokenizer(TokenizerOptions options) : options_(options) {}

std::vector<std::string> Tokenizer::tokenize(std::string_view) const {
    return {};
}

bool Tokenizer::is_stopword(std::string_view) {
    return false;
}

std::size_t Tokenizer::stopword_count() {
    return 0;
}

}  // namespace bm25
