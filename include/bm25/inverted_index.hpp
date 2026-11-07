#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm25/corpus.hpp"
#include "bm25/tokenizer.hpp"

namespace bm25 {

// One entry of a posting list: term occurs `tf` times in document `doc_id`.
struct Posting {
    std::uint32_t doc_id;
    std::uint32_t tf;

    bool operator==(const Posting& o) const { return doc_id == o.doc_id && tf == o.tf; }
};

struct IndexStats {
    std::size_t num_docs = 0;
    std::size_t vocab_size = 0;      // number of distinct terms
    std::size_t total_postings = 0;  // sum of all posting-list lengths
    std::size_t total_tokens = 0;    // sum of all document lengths
    double memory_mb = 0.0;          // approximate heap footprint, see stats()
};

// Term -> posting list, plus per-document lengths. Doc ids are the positions
// of the documents passed to build(), and every posting list is sorted by
// ascending doc id (a consequence of building in doc-id order).
//
// Terms are interned: a hash map gives each distinct term a dense integer id
// (in order of first appearance), and posting lists live in a vector indexed
// by that id. Building then hashes each token's string once; everything after
// that works on integers.
class InvertedIndex {
public:
    // Replaces any previous contents. One pass over the documents.
    void build(const std::vector<Document>& docs, const Tokenizer& tokenizer);

    // Posting list of `term` (already tokenized form). Empty if unknown.
    const std::vector<Posting>& postings(const std::string& term) const;

    // Number of documents containing `term` (0 if unknown).
    std::uint32_t doc_freq(const std::string& term) const;

    // Length of document `doc_id` in tokens, after stopword removal.
    std::uint32_t doc_length(std::uint32_t doc_id) const { return doc_lengths_[doc_id]; }

    // External (corpus) id of an internal doc id.
    const std::string& external_id(std::uint32_t doc_id) const { return external_ids_[doc_id]; }

    std::size_t num_docs() const { return doc_lengths_.size(); }
    double avgdl() const { return avgdl_; }

    IndexStats stats() const;

private:
    std::unordered_map<std::string, std::uint32_t> term_ids_;  // term -> term id
    std::vector<std::vector<Posting>> postings_;              // indexed by term id
    std::vector<std::uint32_t> doc_lengths_;
    std::vector<std::string> external_ids_;
    double avgdl_ = 0.0;
};

}  // namespace bm25
