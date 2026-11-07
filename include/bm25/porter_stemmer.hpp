#pragma once

#include <string>
#include <string_view>

namespace bm25 {

// The Porter (1980) suffix-stripping stemmer, following Martin Porter's
// reference C implementation (the version Lucene's PorterStemFilter uses and
// the one the official voc.txt/output.txt test vocabulary was generated with).
// Steps: 1a, 1b, 1c, 2, 3, 4, 5a, 5b.
//
// Input is expected to be lowercase ASCII; words of length <= 2 are returned
// unchanged.
class PorterStemmer {
public:
    std::string stem(std::string_view word) const;
};

}  // namespace bm25
