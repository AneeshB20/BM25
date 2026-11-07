#include "bm25/inverted_index.hpp"

#include <limits>
#include <stdexcept>

namespace bm25 {

void InvertedIndex::build(const std::vector<Document>& docs, const Tokenizer& tokenizer) {
    if (docs.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("too many documents for 32-bit doc ids");
    }
    term_ids_.clear();
    postings_.clear();
    doc_lengths_.assign(docs.size(), 0);
    external_ids_.clear();
    external_ids_.reserve(docs.size());

    // Per-document term counts, indexed by term id. `doc_terms` lists the ids
    // seen in the current document so only those slots are read and reset.
    std::vector<std::uint32_t> tf_by_term;
    std::vector<std::uint32_t> doc_terms;
    std::uint64_t total_tokens = 0;

    for (std::uint32_t doc_id = 0; doc_id < docs.size(); ++doc_id) {
        external_ids_.push_back(docs[doc_id].id);

        const std::vector<std::string> tokens = tokenizer.tokenize(docs[doc_id].text);
        doc_lengths_[doc_id] = static_cast<std::uint32_t>(tokens.size());
        total_tokens += tokens.size();

        doc_terms.clear();
        for (const std::string& t : tokens) {
            // One string hash per token: find the term's id, or assign the next one.
            const auto [it, inserted] =
                term_ids_.try_emplace(t, static_cast<std::uint32_t>(postings_.size()));
            const std::uint32_t id = it->second;
            if (inserted) {
                postings_.emplace_back();
                tf_by_term.push_back(0);
            }
            if (tf_by_term[id]++ == 0) {
                doc_terms.push_back(id);
            }
        }
        // Documents are visited in increasing doc_id order, so appending keeps
        // every posting list sorted without an explicit sort.
        for (const std::uint32_t id : doc_terms) {
            postings_[id].push_back(Posting{doc_id, tf_by_term[id]});
            tf_by_term[id] = 0;
        }
    }

    avgdl_ = docs.empty() ? 0.0 : static_cast<double>(total_tokens) / static_cast<double>(docs.size());
}

const std::vector<Posting>& InvertedIndex::postings(const std::string& term) const {
    static const std::vector<Posting> kEmpty;
    auto it = term_ids_.find(term);
    return it == term_ids_.end() ? kEmpty : postings_[it->second];
}

std::uint32_t InvertedIndex::doc_freq(const std::string& term) const {
    return static_cast<std::uint32_t>(postings(term).size());
}

IndexStats InvertedIndex::stats() const {
    IndexStats s;
    s.num_docs = num_docs();
    s.vocab_size = postings_.size();

    // Approximate heap footprint. Counts what the containers own:
    //  - per dictionary node: key string object + 4-byte id (padded to 8)
    //    + next pointer + cached hash (libstdc++ caches it for string keys),
    //  - key characters that do not fit in the small-string buffer,
    //  - the dictionary's bucket array,
    //  - one vector object per term plus its posting capacity,
    //  - doc lengths and external id strings.
    // Allocator headers and padding are ignored, so the true figure is somewhat higher.
    std::size_t bytes = 0;
    const std::size_t node_bytes = sizeof(void*) + sizeof(std::string) + 8 + sizeof(std::size_t);
    const std::size_t sso_capacity = std::string().capacity();
    for (const auto& entry : term_ids_) {
        const std::string& term = entry.first;
        bytes += node_bytes;
        if (term.capacity() > sso_capacity) {
            bytes += term.capacity() + 1;
        }
    }
    bytes += term_ids_.bucket_count() * sizeof(void*);
    bytes += postings_.capacity() * sizeof(std::vector<Posting>);
    for (const std::vector<Posting>& list : postings_) {
        s.total_postings += list.size();
        bytes += list.capacity() * sizeof(Posting);
    }
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
