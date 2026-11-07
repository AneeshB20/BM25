// Searches a BEIR dataset with BM25.
//
// Batch mode (used by the Python cross-check and evaluation scripts):
//   bm25_search --batch [--data DIR] [-k N] [--k1 X] [--b X] [--no-stem] [--no-stopwords]
//   reads "query_id<TAB>query text" lines from stdin and writes a TREC run:
//   "qid Q0 docid rank score bm25" with 9 decimal places.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "bm25/corpus.hpp"
#include "bm25/inverted_index.hpp"
#include "bm25/searcher.hpp"
#include "bm25/tokenizer.hpp"

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: bm25_search --batch [--data DIR] [-k N] [--k1 X] [--b X]\n"
                 "                   [--no-stem] [--no-stopwords]\n"
                 "  reads 'qid<TAB>text' lines on stdin, writes a TREC run on stdout\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = "data/scifact";
    std::size_t k = 10;
    bool batch = false;
    bm25::TokenizerOptions opts;
    bm25::Bm25Params params;

    for (int i = 1; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (std::strcmp(argv[i], "--batch") == 0) {
            batch = true;
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
    if (!batch) {
        usage();
        return 2;
    }

    try {
        const bm25::Corpus corpus = bm25::Corpus::load(data_dir);
        const bm25::Tokenizer tokenizer(opts);
        bm25::InvertedIndex index;
        index.build(corpus.documents(), tokenizer);
        const bm25::Searcher searcher(index, tokenizer, params);

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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
