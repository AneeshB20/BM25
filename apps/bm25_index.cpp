// Builds the inverted index over a BEIR dataset and prints build statistics.
//
// Usage: bm25_index [--data DIR] [--no-stem] [--no-stopwords]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "bm25/corpus.hpp"
#include "bm25/inverted_index.hpp"
#include "bm25/tokenizer.hpp"

namespace {

void usage() {
    std::fprintf(stderr, "usage: bm25_index [--data DIR] [--no-stem] [--no-stopwords]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = "data/scifact";
    bm25::TokenizerOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
