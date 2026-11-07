#include "bm25/inverted_index.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace bm25 {

void InvertedIndex::build(const std::vector<Document>& docs, const Tokenizer& tokenizer) {
    if (docs.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("too many documents for 32-bit doc ids");
    }
    tokenizer_options_ = tokenizer.options();
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

// ---------------------------------------------------------------------------
// Binary save / load
// ---------------------------------------------------------------------------

namespace {

constexpr char kMagic[8] = {'B', 'M', '2', '5', 'I', 'D', 'X', '\0'};
constexpr std::uint32_t kVersion = 1;

// Thin helpers over std::ofstream/std::ifstream that write and read
// fixed-size values and length-prefixed strings, checking every read.
class Writer {
public:
    explicit Writer(const std::string& path) : out_(path, std::ios::binary), path_(path) {
        if (!out_) throw std::runtime_error("cannot write " + path);
    }
    template <typename T>
    void value(const T& v) {
        out_.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    template <typename T>
    void array(const std::vector<T>& v) {
        out_.write(reinterpret_cast<const char*>(v.data()),
                   static_cast<std::streamsize>(v.size() * sizeof(T)));
    }
    void string(const std::string& s) {
        value(static_cast<std::uint32_t>(s.size()));
        out_.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
    void finish() {
        out_.flush();
        if (!out_) throw std::runtime_error("error while writing " + path_);
    }

private:
    std::ofstream out_;
    std::string path_;
};

class Reader {
public:
    explicit Reader(const std::string& path) : in_(path, std::ios::binary), path_(path) {
        if (!in_) throw std::runtime_error("cannot open " + path);
    }
    void bytes(char* dst, std::size_t n) {
        in_.read(dst, static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in_.gcount()) != n) {
            throw std::runtime_error(path_ + ": truncated index file");
        }
    }
    template <typename T>
    T value() {
        T v;
        bytes(reinterpret_cast<char*>(&v), sizeof(T));
        return v;
    }
    template <typename T>
    void array(std::vector<T>& v, std::size_t n) {
        v.resize(n);
        bytes(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    }
    std::string string() {
        const auto n = value<std::uint32_t>();
        std::string s(n, '\0');
        bytes(s.data(), n);
        return s;
    }
    bool at_end() { return in_.peek() == std::ifstream::traits_type::eof(); }
    const std::string& path() const { return path_; }

private:
    std::ifstream in_;
    std::string path_;
};

}  // namespace

void InvertedIndex::save(const std::string& path) const {
    Writer w(path);
    w.value(kMagic);
    w.value(kVersion);
    w.value(static_cast<std::uint8_t>(tokenizer_options_.stem));
    w.value(static_cast<std::uint8_t>(tokenizer_options_.remove_stopwords));
    w.value(static_cast<std::uint64_t>(num_docs()));
    w.value(avgdl_);
    for (const std::string& id : external_ids_) {
        w.string(id);
    }
    w.array(doc_lengths_);

    // Term strings in id order, so ids survive the round trip unchanged.
    std::vector<const std::string*> terms(postings_.size());
    for (const auto& [term, id] : term_ids_) {
        terms[id] = &term;
    }
    w.value(static_cast<std::uint64_t>(postings_.size()));
    for (std::size_t id = 0; id < postings_.size(); ++id) {
        w.string(*terms[id]);
        w.value(static_cast<std::uint64_t>(postings_[id].size()));
        w.array(postings_[id]);
    }
    w.finish();
}

InvertedIndex InvertedIndex::load(const std::string& path) {
    Reader r(path);
    char magic[sizeof kMagic];
    r.bytes(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof kMagic) != 0) {
        throw std::runtime_error(path + ": not a BM25 index file");
    }
    if (const auto version = r.value<std::uint32_t>(); version != kVersion) {
        throw std::runtime_error(path + ": unsupported index version " + std::to_string(version));
    }

    InvertedIndex idx;
    idx.tokenizer_options_.stem = r.value<std::uint8_t>() != 0;
    idx.tokenizer_options_.remove_stopwords = r.value<std::uint8_t>() != 0;
    const auto num_docs = r.value<std::uint64_t>();
    if (num_docs > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(path + ": corrupt document count");
    }
    idx.avgdl_ = r.value<double>();
    idx.external_ids_.reserve(num_docs);
    for (std::uint64_t d = 0; d < num_docs; ++d) {
        idx.external_ids_.push_back(r.string());
    }
    r.array(idx.doc_lengths_, num_docs);
    {
        // avgdl must be exactly what build() computes from the doc lengths.
        std::uint64_t total = 0;
        for (std::uint32_t len : idx.doc_lengths_) total += len;
        const double expected =
            num_docs == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(num_docs);
        if (idx.avgdl_ != expected) {
            throw std::runtime_error(path + ": avgdl does not match document lengths");
        }
    }

    const auto vocab = r.value<std::uint64_t>();
    idx.term_ids_.reserve(vocab);
    idx.postings_.resize(vocab);
    for (std::uint64_t id = 0; id < vocab; ++id) {
        std::string term = r.string();
        const auto count = r.value<std::uint64_t>();
        if (count > num_docs) {
            throw std::runtime_error(path + ": corrupt posting count");
        }
        r.array(idx.postings_[id], count);
        // The searcher indexes arrays by doc_id and relies on every posting
        // contributing a positive score, so reject anything a build() could
        // not have produced: doc ids in range and strictly ascending, tf >= 1.
        const std::vector<Posting>& plist = idx.postings_[id];
        for (std::size_t i = 0; i < plist.size(); ++i) {
            if (plist[i].doc_id >= num_docs || plist[i].tf == 0 ||
                (i > 0 && plist[i].doc_id <= plist[i - 1].doc_id)) {
                throw std::runtime_error(path + ": corrupt posting list for term '" + term + "'");
            }
        }
        if (!idx.term_ids_.emplace(std::move(term), static_cast<std::uint32_t>(id)).second) {
            throw std::runtime_error(path + ": duplicate term");
        }
    }
    if (!r.at_end()) {
        throw std::runtime_error(path + ": trailing bytes after index");
    }
    return idx;
}

}  // namespace bm25
