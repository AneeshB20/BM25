#include "bm25/porter_stemmer.hpp"

namespace bm25 {

// Stub: the tests are written first and must fail against this.
std::string PorterStemmer::stem(std::string_view word) const {
    return std::string(word);
}

}  // namespace bm25
