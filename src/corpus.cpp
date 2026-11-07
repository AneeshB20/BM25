#include "bm25/corpus.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace bm25 {

namespace {

std::ifstream open_or_throw(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    return in;
}

// Reads a string field, treating a missing or null field as "".
std::string string_field(const nlohmann::json& obj, const char* key) {
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) {
        return "";
    }
    return it->get<std::string>();
}

}  // namespace

std::vector<Document> Corpus::load_documents(const std::string& path) {
    std::ifstream in = open_or_throw(path);
    std::vector<Document> docs;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        nlohmann::json obj;
        try {
            obj = nlohmann::json::parse(line);
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": " + e.what());
        }
        Document doc;
        doc.id = string_field(obj, "_id");
        doc.title = string_field(obj, "title");
        doc.text = doc.title + " " + string_field(obj, "text");
        docs.push_back(std::move(doc));
    }
    return docs;
}

std::vector<Query> Corpus::load_queries(const std::string& path) {
    std::ifstream in = open_or_throw(path);
    std::vector<Query> queries;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        nlohmann::json obj;
        try {
            obj = nlohmann::json::parse(line);
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": " + e.what());
        }
        queries.push_back(Query{string_field(obj, "_id"), string_field(obj, "text")});
    }
    return queries;
}

Qrels Corpus::load_qrels(const std::string& path) {
    std::ifstream in = open_or_throw(path);
    Qrels qrels;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (first) {
            first = false;
            // BEIR qrels start with a "query-id\tcorpus-id\tscore" header.
            if (line.rfind("query-id", 0) == 0) {
                continue;
            }
        }
        std::istringstream fields(line);
        std::string qid, docid;
        int grade = 0;
        if (!(fields >> qid >> docid >> grade)) {
            throw std::runtime_error(path + ": malformed qrels line: " + line);
        }
        qrels[qid][docid] = grade;
    }
    return qrels;
}

Corpus Corpus::load(const std::string& dir, const std::string& split) {
    Corpus c;
    c.documents_ = load_documents(dir + "/corpus.jsonl");
    c.queries_ = load_queries(dir + "/queries.jsonl");
    c.qrels_ = load_qrels(dir + "/qrels/" + split + ".tsv");
    return c;
}

std::vector<Query> Corpus::judged_queries() const {
    std::vector<Query> out;
    for (const Query& q : queries_) {
        if (qrels_.count(q.id) > 0) {
            out.push_back(q);
        }
    }
    return out;
}

}  // namespace bm25
