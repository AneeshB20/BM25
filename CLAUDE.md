# BM25 Search Engine

A from-scratch lexical search engine in C++17: tokenizer, inverted index and
BM25 ranking, evaluated on the BEIR SciFact benchmark and compared against
dense embedding retrieval. This is a learning and portfolio project: the owner
must be able to explain every line in an interview.

## Hard rules
- No search or IR libraries in C++ (no Lucene ports, no Xapian, no Tantivy
  bindings). The standard library is fine. nlohmann/json is allowed for JSONL
  parsing. GoogleTest is allowed for tests.
- Python is used ONLY for evaluation helpers and the dense-retrieval
  comparison. Never for the engine itself.
- Prefer clarity over cleverness. No template metaprogramming or premature
  optimisation. Optimise only in the performance phase, and measure before and
  after.
- Every number written in README or NOTES must come from a command that was
  actually run. Never estimate or invent a metric.

## Build and test
- Configure: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
- Build:     cmake --build build -j
- Test:      ctest --test-dir build --output-on-failure
- Data:      ./scripts/get_data.sh   (downloads BEIR SciFact into data/scifact/)

## Layout
- include/bm25/   public headers (one class per header)
- src/            implementations
- apps/           command-line tools (bm25_index, bm25_search, bm25_eval)
- tests/          GoogleTest unit tests
- scripts/        data download, Python reference scorer, dense baseline, eval
- data/           downloaded datasets (git-ignored)
- results/        run files and metric tables (committed)

## Design
- Classes: Tokenizer, PorterStemmer, InvertedIndex, Bm25Scorer, Searcher.
  Each has a single job and its own tests.
- BM25 formula (Lucene variant; use exactly this everywhere, including the
  Python reference):
    idf(t)  = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score  += idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
  Defaults: k1 = 0.9, b = 0.4 (Anserini's BEIR defaults).
- Postings: unordered_map<string, vector<Posting{doc_id, tf}>>, with doc ids
  ascending inside each list. Doc lengths are stored in a vector indexed by
  doc id.
- Top-k uses a size-k min-heap, never a full sort.

## Workflow rules
- At the end of every phase, append a section to NOTES.md written for the owner:
  what was built, why each design choice was made, the time and space
  complexity of the main operations, and 3 likely interview questions with
  short answers.
- Show evidence when you finish: paste the test output and command output.
  Do not just say "done".
- Stop at the end of each phase and wait. Do not start the next phase.
- Commit messages: feat:, test:, fix:, perf:, docs:. One concern per commit.
