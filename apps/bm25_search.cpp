// Searches a BEIR dataset with BM25.
//
// Interactive (default):
//   bm25_search [--index FILE] [--data DIR] [-k N] [--k1 X] [--b X]
//   Reads one query per line from stdin; prints rank, doc id, score and the
//   first 100 characters of the title. With --index, the saved index (and its
//   tokenizer options) is loaded instead of being rebuilt; the corpus in --data
//   is still read for the titles.
//
// Batch (used by scripts/reference_bm25.py):
//   bm25_search --batch [--index FILE] [--data DIR] [-k N] [--k1 X] [--b X]
//               [--no-stem] [--no-stopwords]
//   Reads "query_id<TAB>query text" lines and writes a TREC run:
//   "qid Q0 docid rank score bm25" with 9 decimal places.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

#include "bm25/corpus.hpp"
#include "bm25/inverted_index.hpp"
#include "bm25/searcher.hpp"
#include "bm25/tokenizer.hpp"

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: bm25_search [--index FILE] [--data DIR] [-k N] [--k1 X] [--b X]\n"
                 "                   [--no-stem] [--no-stopwords] [--batch]\n"
                 "  interactive: one query per line on stdin\n"
                 "  --batch:     'qid<TAB>text' lines on stdin, TREC run on stdout\n");
}

// First `max_bytes` bytes of `s`, cut back so a UTF-8 character is never split.
std::string truncate_utf8(const std::string& s, std::size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    std::size_t end = max_bytes;
    while (end > 0 && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) {
        --end;  // s[end] is a continuation byte: back off to the character start
    }
    return s.substr(0, end) + "...";
}

void run_batch(const bm25::Searcher& searcher, std::size_t k) {
    const bm25::InvertedIndex& index = searcher.index();
    std::string line;
    while (std::getline(std::cin, line)) {
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            std::fprintf(stderr, "skipping line without a tab: %s\n", line.c_str());
            continue;
        }
        const std::string qid = line.substr(0, tab);
        const auto results = searcher.search(std::string_view(line).substr(tab + 1), k);
        for (std::size_t r = 0; r < results.size(); ++r) {
            std::printf("%s Q0 %s %zu %.9f bm25\n", qid.c_str(),
                        index.external_id(results[r].doc_id).c_str(), r + 1, results[r].score);
        }
    }
}

void run_interactive(const bm25::Searcher& searcher, std::size_t k,
                     const std::unordered_map<std::string, std::string>& titles) {
    const bm25::InvertedIndex& index = searcher.index();
    const bool tty = isatty(STDIN_FILENO) != 0;
    std::string line;
    while (true) {
        if (tty) {
            std::fprintf(stderr, "query> ");
            std::fflush(stderr);
        }
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        const auto results = searcher.search(line, k);
        std::printf("query: %s\n", line.c_str());
        if (results.empty()) {
            std::printf("  (no matching documents)\n\n");
            continue;
        }
        std::printf("%4s  %-10s  %8s  %s\n", "rank", "doc id", "score", "title");
        for (std::size_t r = 0; r < results.size(); ++r) {
            const std::string& id = index.external_id(results[r].doc_id);
            auto it = titles.find(id);
            const std::string title = it == titles.end() ? "" : truncate_utf8(it->second, 100);
            std::printf("%4zu  %-10s  %8.4f  %s\n", r + 1, id.c_str(), results[r].score, title.c_str());
        }
        std::printf("\n");
        std::fflush(stdout);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = "data/scifact";
    std::string index_path;
    std::size_t k = 10;
    bool batch = false;
    bm25::TokenizerOptions opts;
    bm25::Bm25Params params;

    for (int i = 1; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (std::strcmp(argv[i], "--batch") == 0) {
            batch = true;
        } else if (std::strcmp(argv[i], "--index") == 0 && has_value) {
            index_path = argv[++i];
        } else if (std::strcmp(argv[i], "--data") == 0 && has_value) {
            data_dir = argv[++i];
        } else if (std::strcmp(argv[i], "-k") == 0 && has_value) {
            k = std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--k1") == 0 && has_value) {
            params.k1 = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--b") == 0 && has_value) {
            params.b = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--no-stem") == 0) {
            opts.stem = false;
        } else if (std::strcmp(argv[i], "--no-stopwords") == 0) {
            opts.remove_stopwords = false;
        } else {
            usage();
            return 2;
        }
    }

    try {
        bm25::InvertedIndex index;
        if (!index_path.empty()) {
            index = bm25::InvertedIndex::load(index_path);
            opts = index.tokenizer_options();  // queries must be analysed like the index
        }
        const bm25::Tokenizer tokenizer(opts);

        // The corpus is needed to build the index, or (interactive) for titles.
        std::unordered_map<std::string, std::string> titles;
        if (index_path.empty() || !batch) {
            const auto docs = bm25::Corpus::load_documents(data_dir + "/corpus.jsonl");
            if (index_path.empty()) {
                index.build(docs, tokenizer);
            }
            if (!batch) {
                titles.reserve(docs.size());
                for (const bm25::Document& d : docs) titles.emplace(d.id, d.title);
            }
        }

        const bm25::Searcher searcher(index, tokenizer, params);
        if (batch) {
            run_batch(searcher, k);
        } else {
            run_interactive(searcher, k, titles);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
