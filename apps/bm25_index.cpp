// Builds the inverted index over a BEIR dataset and prints build statistics.
//
// Usage: bm25_index [--data DIR] [--no-stem] [--no-stopwords] [--export-tokens DIR]
//                   [--save FILE]
//
// --save FILE writes the index in binary form (see InvertedIndex::save).
//
// --export-tokens DIR writes the tokenizer output so scripts/reference_bm25.py
// can score exactly the same terms:
//   DIR/corpus_tokens.tsv   "doc_id<TAB>tok tok ..." per document, in doc-id order
//   DIR/query_tokens.tsv    "query_id<TAB>tok tok ..." per judged test query

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bm25/corpus.hpp"
#include "bm25/inverted_index.hpp"
#include "bm25/tokenizer.hpp"

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: bm25_index [--data DIR] [--no-stem] [--no-stopwords] [--export-tokens DIR]\n"
                 "                  [--save FILE]\n");
}

void write_tokens_line(std::ofstream& out, const std::string& id,
                       const std::vector<std::string>& tokens) {
    out << id << '\t';
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) out << ' ';
        out << tokens[i];
    }
    out << '\n';
}

void export_tokens(const bm25::Corpus& corpus, const bm25::Tokenizer& tokenizer,
                   const std::string& dir) {
    std::ofstream docs(dir + "/corpus_tokens.tsv");
    std::ofstream queries(dir + "/query_tokens.tsv");
    if (!docs || !queries) {
        throw std::runtime_error("cannot write token files in " + dir + " (does it exist?)");
    }
    for (const bm25::Document& d : corpus.documents()) {
        write_tokens_line(docs, d.id, tokenizer.tokenize(d.text));
    }
    for (const bm25::Query& q : corpus.judged_queries()) {
        write_tokens_line(queries, q.id, tokenizer.tokenize(q.text));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = "data/scifact";
    std::string export_dir;
    std::string save_path;
    bm25::TokenizerOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--no-stem") == 0) {
            opts.stem = false;
        } else if (std::strcmp(argv[i], "--no-stopwords") == 0) {
            opts.remove_stopwords = false;
        } else if (std::strcmp(argv[i], "--export-tokens") == 0 && i + 1 < argc) {
            export_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }

    try {
        using Clock = std::chrono::steady_clock;
        auto ms_since = [](Clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        };

        const auto t_load = Clock::now();
        const bm25::Corpus corpus = bm25::Corpus::load(data_dir);
        const double load_ms = ms_since(t_load);

        const bm25::Tokenizer tokenizer(opts);
        bm25::InvertedIndex index;
        const auto t_build = Clock::now();
        index.build(corpus.documents(), tokenizer);
        const double build_ms = ms_since(t_build);

        const bm25::IndexStats s = index.stats();
        std::printf("tokenizer:        stopwords=%s stem=%s\n", opts.remove_stopwords ? "on" : "off",
                    opts.stem ? "on" : "off");
        std::printf("corpus load:      %.1f ms\n", load_ms);
        std::printf("index build:      %.1f ms (tokenize + invert)\n", build_ms);
        std::printf("documents:        %zu\n", s.num_docs);
        std::printf("vocabulary size:  %zu\n", s.vocab_size);
        std::printf("total postings:   %zu\n", s.total_postings);
        std::printf("total tokens:     %zu\n", s.total_tokens);
        std::printf("avgdl (tokens):   %.4f\n", index.avgdl());
        std::printf("memory (approx):  %.2f MB\n", s.memory_mb);

        if (!save_path.empty()) {
            const auto t_save = Clock::now();
            index.save(save_path);
            const double save_ms = ms_since(t_save);
            const double mb = static_cast<double>(std::filesystem::file_size(save_path)) / (1024.0 * 1024.0);
            std::printf("saved index:      %s (%.2f MB, %.1f ms)\n", save_path.c_str(), mb, save_ms);
        }
        if (!export_dir.empty()) {
            export_tokens(corpus, tokenizer, export_dir);
            std::printf("tokens exported:  %s/{corpus,query}_tokens.tsv\n", export_dir.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
