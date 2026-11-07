#include "bm25/inverted_index.hpp"

#include <limits>
#include <stdexcept>

namespace bm25 {

void InvertedIndex::build(const std::vector<Document>& docs, const Tokenizer& tokenizer) {
    if (docs.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("too many documents for 32-bit doc ids");
    }
    postings_.clear();
    doc_lengths_.assign(docs.size(), 0);
    external_ids_.clear();
    external_ids_.reserve(docs.size());

    std::unordered_map<std::string, std::uint32_t> tf;  // reused per document
    std::uint64_t total_tokens = 0;

    for (std::uint32_t doc_id = 0; doc_id < docs.size(); ++doc_id) {
        external_ids_.push_back(docs[doc_id].id);

        const std::vector<std::string> tokens = tokenizer.tokenize(docs[doc_id].text);
        doc_lengths_[doc_id] = static_cast<std::uint32_t>(tokens.size());
        total_tokens += tokens.size();

        tf.clear();
        for (const std::string& t : tokens) {
            ++tf[t];
        }
        // Documents are visited in increasing doc_id order, so appending keeps
        // every posting list sorted without an explicit sort.
        for (const auto& [term, count] : tf) {
            postings_[term].push_back(Posting{doc_id, count});
        }
    }

    avgdl_ = docs.empty() ? 0.0 : static_cast<double>(total_tokens) / static_cast<double>(docs.size());
}

const std::vector<Posting>& InvertedIndex::postings(const std::string& term) const {
    static const std::vector<Posting> kEmpty;
    auto it = postings_.find(term);
    return it == postings_.end() ? kEmpty : it->second;
}

std::uint32_t InvertedIndex::doc_freq(const std::string& term) const {
    return static_cast<std::uint32_t>(postings(term).size());
}

IndexStats InvertedIndex::stats() const {
    IndexStats s;
    s.num_docs = num_docs();
    s.vocab_size = postings_.size();

    // Approximate heap footprint. Counts what the containers own:
    //  - per hash-map node: key string object + vector object + next pointer
    //    + cached hash (libstdc++ stores both for std::string keys),
    //  - key characters that do not fit in the small-string buffer,
    //  - posting-list capacity,
    //  - the bucket array,
    //  - doc lengths and external id strings.
    // Allocator headers and padding are ignored, so the true figure is somewhat higher.
    std::size_t bytes = 0;
    const std::size_t node_bytes =
        sizeof(void*) + sizeof(std::string) + sizeof(std::vector<Posting>) + sizeof(std::size_t);
    const std::size_t sso_capacity = std::string().capacity();
    for (const auto& [term, list] : postings_) {
        s.total_postings += list.size();
        bytes += node_bytes;
        if (term.capacity() > sso_capacity) {
            bytes += term.capacity() + 1;
        }
        bytes += list.capacity() * sizeof(Posting);
    }
    bytes += postings_.bucket_count() * sizeof(void*);
    bytes += doc_lengths_.capacity() * sizeof(std::uint32_t);
    bytes += external_ids_.capacity() * sizeof(std::string);
    for (const std::string& id : external_ids_) {
        if (id.capacity() > sso_capacity) {
            bytes += id.capacity() + 1;
        }
    }
    for (std::uint32_t len : doc_lengths_) {
        s.total_tokens += len;
    }
    s.memory_mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    return s;
}

}  // namespace bm25
