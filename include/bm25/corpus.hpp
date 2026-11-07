#pragma once

#include <map>
#include <string>
#include <vector>

namespace bm25 {

// One retrievable document. `text` is the BEIR title and body joined by a space.
struct Document {
    std::string id = {};     // external id from corpus.jsonl ("_id")
    std::string text = {};   // title + " " + text: what gets indexed
    std::string title = {};  // title alone, for display (optional in tests)
};

struct Query {
    std::string id;    // external id from queries.jsonl ("_id")
    std::string text;
};

// Relevance judgements: query id -> (doc id -> relevance grade).
// std::map keeps iteration order deterministic, which keeps output files stable.
using Qrels = std::map<std::string, std::map<std::string, int>>;

// Loads a BEIR-format dataset directory (corpus.jsonl, queries.jsonl,
// qrels/<split>.tsv). Internal doc ids are positions in documents(), so they
// are dense integers 0..N-1 in file order.
class Corpus {
public:
    // Throws std::runtime_error if a file is missing or malformed.
    static Corpus load(const std::string& dir, const std::string& split = "test");

    const std::vector<Document>& documents() const { return documents_; }
    const std::vector<Query>& queries() const { return queries_; }
    const Qrels& qrels() const { return qrels_; }

    // Queries that have at least one judgement in the loaded split,
    // in queries.jsonl order.
    std::vector<Query> judged_queries() const;

    // Individual loaders, public so they can be unit-tested on small files.
    static std::vector<Document> load_documents(const std::string& path);
    static std::vector<Query> load_queries(const std::string& path);
    static Qrels load_qrels(const std::string& path);

private:
    std::vector<Document> documents_;
    std::vector<Query> queries_;
    Qrels qrels_;
};

}  // namespace bm25
