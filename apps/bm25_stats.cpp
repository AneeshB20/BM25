// Prints basic statistics about a BEIR dataset, as a sanity check that the
// loader reads the files correctly.
//
// Usage: bm25_stats [data_dir]   (default: data/scifact)

#include <cstdio>
#include <exception>
#include <string>

#include "bm25/corpus.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data/scifact";
    try {
        const bm25::Corpus corpus = bm25::Corpus::load(dir);

        std::size_t total_chars = 0;
        for (const bm25::Document& d : corpus.documents()) {
            total_chars += d.text.size();
        }
        const std::size_t n_docs = corpus.documents().size();
        const double avg_chars =
            n_docs == 0 ? 0.0 : static_cast<double>(total_chars) / static_cast<double>(n_docs);

        std::printf("dataset:                    %s\n", dir.c_str());
        std::printf("documents:                  %zu\n", n_docs);
        std::printf("queries (all):              %zu\n", corpus.queries().size());
        std::printf("test queries with qrels:    %zu\n", corpus.judged_queries().size());
        std::printf("avg doc length (chars):     %.2f\n", avg_chars);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
