# Build notes

Written for the project owner. Each phase adds a section: what was built, why,
complexity, and likely interview questions. Every number here comes from a
command that was run; the command is given next to it.

---

## Phase 0: Plan

### Toolchain (checked 2026-11-07)

| Tool | Version | Note |
|---|---|---|
| g++ | 13.3.0 (Ubuntu 24.04) | full C++17 support |
| cmake | 4.4.4 | not installed system-wide and `sudo` needs a password, so it was installed with `pip install cmake` into the project venv `.venv/`. Run `export PATH="$PWD/.venv/bin:$PATH"` before the cmake commands in CLAUDE.md. |
| python3 | 3.12.3 | evaluation helpers run from `.venv/` |
| make | 4.3 | CMake generator |

### Planned file tree

```
CLAUDE.md                     project rules for Claude Code
NOTES.md                      this file
README.md                     summary, results, how to run (Phase 8)
CMakeLists.txt                top-level build: library, apps, tests, FetchContent deps
.gitignore                    build/, data/, .venv/, index binaries
include/bm25/corpus.hpp       Document, Query, Qrels types and the Corpus loader
include/bm25/tokenizer.hpp    Tokenizer: lowercase, split, optional stopwords and stemming
include/bm25/porter_stemmer.hpp  PorterStemmer: Porter (1980) algorithm
include/bm25/inverted_index.hpp  Posting, InvertedIndex: term -> postings, doc lengths
include/bm25/bm25_scorer.hpp  Bm25Params, Bm25Scorer: idf and per-term score
include/bm25/searcher.hpp     SearchResult, Searcher: query -> top-k via min-heap
include/bm25/metrics.hpp      nDCG@k and Recall@k (Phase 5)
src/*.cpp                     one implementation file per header
apps/bm25_stats.cpp           dataset sanity check (Phase 1)
apps/bm25_index.cpp           build index, print stats, export tokens, save index
apps/bm25_search.cpp          batch (TREC run) and interactive search
apps/bm25_eval.cpp            run all test queries, write run file, metrics, benchmark
tests/*_test.cpp              one GoogleTest file per class
scripts/get_data.sh           download BEIR SciFact (idempotent)
scripts/requirements.txt      Python deps for evaluation helpers
scripts/reference_bm25.py     independent Python BM25 for cross-checking (Phase 4)
scripts/eval.py               pytrec_eval cross-check of C++ metrics (Phase 5)
scripts/dense_baseline.py     MiniLM dense retrieval + RRF hybrid (Phase 7)
results/                      run files, ablations.md, perf.md, comparison.md
```

### Class interfaces

```cpp
struct TokenizerOptions { bool remove_stopwords = true; bool stem = true; };
class Tokenizer {
  explicit Tokenizer(TokenizerOptions opts = {});
  std::vector<std::string> tokenize(std::string_view text) const;
  const TokenizerOptions& options() const;
  static bool is_stopword(std::string_view word);
};

class PorterStemmer {
  std::string stem(std::string_view word) const;   // expects lowercase a-z
};

struct Posting { std::uint32_t doc_id; std::uint32_t tf; };
struct IndexStats { std::size_t vocab_size, total_postings; double memory_mb; };
class InvertedIndex {
  void build(const std::vector<Document>& docs, const Tokenizer& tok);
  const std::vector<Posting>& postings(const std::string& term) const; // empty if unknown
  std::uint32_t doc_freq(const std::string& term) const;
  std::uint32_t doc_length(std::uint32_t doc_id) const;
  std::size_t num_docs() const;
  double avgdl() const;
  IndexStats stats() const;
};

struct Bm25Params { double k1 = 0.9; double b = 0.4; };
class Bm25Scorer {
  Bm25Scorer(std::size_t num_docs, double avgdl, Bm25Params p = {});
  double idf(std::uint32_t doc_freq) const;
  double term_score(double idf, std::uint32_t tf, std::uint32_t doc_len) const;
};

struct SearchResult { std::uint32_t doc_id; double score; };
class Searcher {
  Searcher(const InvertedIndex& index, const Tokenizer& tok, Bm25Params p = {});
  std::vector<SearchResult> search(std::string_view query, std::size_t k) const;
};
```

### How a query flows

1. `Searcher::search("Vitamin D deficiency", 10)` calls `Tokenizer::tokenize` with the
   same options used at index time: lowercase, split on non-alphanumerics, drop
   stopwords, Porter-stem → `["vitamin", "d", "defici"]`.
2. Duplicate query terms are removed, keeping first-occurrence order.
3. For each term: look up its posting list (one hash lookup), compute
   `idf = Bm25Scorer::idf(df)` once, then for every `(doc_id, tf)` posting add
   `term_score(idf, tf, doc_length(doc_id))` into a score accumulator.
4. Scan the accumulator with a size-k min-heap: push while the heap has fewer than k
   entries, otherwise replace the root if the candidate beats it.
5. Pop the heap into a vector and reverse it, giving results sorted by descending
   score (ties broken by ascending internal doc id). The caller maps internal ids
   back to external `_id` strings.

### Risks to the nDCG@10 target (0.66–0.68)

- **Tokenizer differences from Lucene.** Anserini uses Lucene's `StandardTokenizer`
  (Unicode word-break rules: keeps `3.5`, `e.g` and possessives together),
  `EnglishPossessiveFilter` (strips `'s`) and `PorterStemFilter`. We split on every
  non-alphanumeric ASCII character, so `patient's` gives `patient` + `s`, and `3.5`
  gives `3` + `5`. Non-ASCII letters (Greek letters and accented characters are common
  in SciFact) become split points. Each difference moves scores a little; together
  they could cost about ±0.01.
- **Porter variant.** Lucene uses Martin Porter's reference C implementation, which
  differs slightly from the 1980 paper (e.g. `-bli → -ble`, `-logi → -log` in step 2).
  The official test vocabulary matches the reference implementation, so we implement
  that one.
- **Lucene doc-length quantisation.** Lucene stores each document length in one byte
  (lossy SmallFloat encoding), so Anserini's `dl` is approximate. We use exact lengths.
  This is a small but real source of disagreement.
- **Duplicate query terms.** Anserini's bag-of-words query weights a repeated term by
  its count. The Phase 4 prompt says count once. SciFact queries are short claims, so
  this should matter very little, but it is a known difference.
- **Flat vs multifield.** Anserini reports two SciFact BM25 numbers: "flat" (title and
  text in one field, which is what we do) and "multifield" (separate fields). The
  0.665 figure is, as far as I recall, the multifield one; the flat number is reported
  separately and could not be checked offline. Treat the target as approximate.
- **Tie-breaking** between equal scores changes nDCG slightly. trec_eval breaks ties by
  doc id string, not by our rank column, so the pytrec_eval cross-check can differ by
  a hair on tied scores.

### Things in CLAUDE.md that are wrong or ambiguous

1. **"Lucene variant" formula.** Modern Lucene (8+) drops the `(k1 + 1)` factor from
   the numerator. That scales every score by the same constant, so rankings are
   identical. Fine to keep, but scores will not match Lucene's raw numbers.
2. **"Postings: unordered_map<string, ...>"** conflicts with the Phase 6 option to
   intern terms to integer ids. Read it as the starting design, which Phase 6 may
   change once it is measured.
3. **Anserini's 0.665** is the multifield result, but CLAUDE.md asks for title and text
   joined (flat). See Risks.
4. **Duplicate query terms:** the Phase 4 rule (count once) differs from Anserini.
   We follow the prompt and document it.
5. **Doc ids:** BEIR ids are strings. CLAUDE.md says "doc ids ascending"; we use dense
   internal `uint32_t` ids (corpus line order) and keep a separate vector that maps
   them back to the external id strings.
6. **"Python only for evaluation" vs Phase 4's "same tokenizer output".** The Python
   reference does not re-implement tokenization; it reads tokens exported by the C++
   tokenizer. That keeps the rule and makes the cross-check test the scoring alone.

---

## Phase 1: Scaffold and data

### What was built

- `CMakeLists.txt`: C++17 (no GNU extensions), Release by default, Debug available.
  `-Wall -Wextra -Wpedantic` is applied to our own targets only, through
  `bm25_set_warnings()`, so third-party warnings don't hide ours.
- Dependencies through `FetchContent`, pinned by release URL **and SHA-256**
  (nlohmann/json 3.11.3, GoogleTest 1.15.2). Pinning the hash makes builds
  reproducible and protects against a tampered download.
- `scripts/get_data.sh`: downloads SciFact, unzips into a temp dir and only then moves
  it into `data/scifact/`. An interrupted run therefore can't leave a half-extracted
  dataset that looks complete. A second run sees the files and exits.
- `Corpus` (`include/bm25/corpus.hpp`): loads `corpus.jsonl` (title + " " + text),
  `queries.jsonl` and `qrels/test.tsv`. Internal doc ids are line positions, which
  gives dense integers 0..N-1. `judged_queries()` returns the queries that have qrels.
- `apps/bm25_stats`: dataset sanity check.
- Tests: a placeholder test, plus 5 Corpus tests (title/text join, empty title, blank
  lines, qrels header skip, missing file, malformed JSON).

### Evidence

```
$ cmake --build build --clean-first -j 2>&1 | grep -ci warning
0
$ ctest --test-dir build --output-on-failure
100% tests passed out of 6
$ ./build/apps/bm25_stats
documents:                  5183
queries (all):              1109
test queries with qrels:    300
avg doc length (chars):     1500.75
```

Debug build (`build-debug/`): also 0 warnings, 6/6 tests pass.

### Design choices

- **Why dense integer doc ids?** Postings store 4-byte ints instead of strings, doc
  lengths can live in a plain vector indexed by id, and a score accumulator can later be
  an array. External ids are only needed when printing results.
- **Why `std::map` for qrels?** It is small (339 judgements) and sorted iteration keeps
  output files byte-stable between runs. A hash map's iteration order isn't guaranteed.
- **Why parse JSON with a library?** Hand-written JSON parsing (escapes, unicode,
  nesting) is a source of bugs that teaches nothing about search. CLAUDE.md allows it.
- **Why read line by line?** JSONL is one object per line, so memory is bounded by the
  longest line plus the parsed output, and an error can report its line number.

### Complexity

- Loading: O(total bytes) time; O(total text) space, because document text is kept in
  memory. SciFact is about 8 MB, so this is fine.

### Likely interview questions

1. **Why did you pin dependency hashes?** So that a build is reproducible: the same
   source always gives the same binary. It also means a compromised or changed
   download fails the build instead of getting compiled in.
2. **How do you map between internal and external doc ids?** The internal id is the
   index into `documents()`, so going from internal to external is a vector lookup. The
   reverse direction is only needed for qrels and could use a hash map. The engine
   never needs it.
3. **What makes `get_data.sh` safe to run twice?** It checks for every required file
   before downloading, and it extracts into a temp dir that is moved into place in one
   step at the end. A failed run cleans up through `trap`.

---

## Phase 2: Tokenizer and Porter stemmer

### What was built

- `Tokenizer` (`include/bm25/tokenizer.hpp`): lowercase ASCII → split on anything
  that is not `[a-z0-9]` → drop empties → optional stopword removal → optional
  Porter stemming. The two options are fields of `TokenizerOptions`, so the
  Phase 5 ablations only change a flag.
- `PorterStemmer` (`include/bm25/porter_stemmer.hpp`): a hand-written C++
  translation of Martin Porter's reference C implementation, steps 1a, 1b, 1c, 2, 3,
  4, 5a, 5b. No library.
- Tests were written first and committed against stub implementations (commit
  `test: add tokenizer and Porter stemmer tests…`). 18 of them failed against the
  stubs. Then the implementation was written until they all passed.

### Evidence

```
$ ctest --test-dir build --output-on-failure
...
28/28 Test #28: PorterStemmer.OfficialVocabulary ...........   Passed    0.01 sec
100% tests passed out of 28
```

- 12 tokenizer tests: punctuation, digits/decimals, mixed case, empty string, only
  separators, non-ASCII bytes, only stopwords, stopwords among other words, the
  stemming flag, stopword-before-stem ordering, defaults, and the exact 33-word list.
- 60 hand-picked word → stem pairs (6 from the 1980 paper, 54 from the official
  vocabulary, grouped by step), plus short-word cases.
- `PorterStemmer.OfficialVocabulary` runs **all 23,531 words** of the official
  `voc.txt` against `output.txt` with **0 mismatches**. `get_data.sh` downloads the
  files into `data/porter/`; the test skips if they're missing.
- The same suite also passed under AddressSanitizer + UndefinedBehaviorSanitizer
  (`build-asan/`, `-fsanitize=address,undefined -D_GLIBCXX_ASSERTIONS`). This
  matters because the stemmer does a lot of index arithmetic.

### Which stopword list

Lucene's `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`, which is Anserini's default. It
has exactly 33 words:
`a an and are as at be but by for if in into is it no not of on or such that the
their then there these they this to was will with`.
It is deliberately short. Words like "we", "were", "has" and "can" are **not** in it.

### Design choices

- **Stopwords are checked before stemming**, as in Lucene's analyzer chain
  (StopFilter → PorterStemFilter). The stem of a stopword can be a normal word
  ("this" → "thi"), so checking after stemming would let stopwords through.
- **ASCII-only character tests** instead of `std::isalnum`/`std::tolower`. Those depend
  on the locale and are undefined behaviour for bytes ≥ 0x80 when `char` is signed,
  which UTF-8 text is full of. The cost is that non-ASCII letters (β, ï, µ) act as
  separators: "β-cell" becomes "cell". That is a known recall loss on SciFact.
- **Which Porter version.** Martin Porter's reference implementation differs from the
  1980 paper in a few places (`-bli → -ble`, the added `-logi → -log` rule, and never
  stemming words of length ≤ 2). Lucene uses the reference version, and so does the
  official test vocabulary. Matching it is what makes the 23,531-word test possible.
- **Binary search over a sorted `std::array`** for stopwords. With 33 entries this is
  at most 6 comparisons and needs no heap allocation. A hash set would also be fine.

### Why stemming helps recall, and what it breaks

- **Helps:** a query about "infections" should match a document that says "infected"
  or "infection". Stemming maps all three to "infect", so they share one posting list.
  More relevant documents become reachable, which raises recall. It also merges the
  statistics, so the document frequency (and so the IDF) reflects the concept rather
  than one surface form.
- **Breaks (over-stemming):** unrelated words collapse together. "university" and
  "universal" both become "univers", and "generous" and "generalization" both become
  "gener". The query term "general" then also matches documents about "generous"
  behaviour. Each merge like this lowers precision a little.
- **Misses (under-stemming):** irregular forms are not merged ("mice"/"mouse",
  "ran"/"run"), and neither are British/American spellings.
- Stems are not words ("poni", "happi"), so they should never be shown to a user.

### Complexity

- `tokenize`: O(n) in the text length. Every character is visited once; the stemmer
  works on a token of length L in O(L) per step (each step tests a constant number of
  suffixes), and `m()` is O(L).
- Space: O(n) for the output tokens.

### Likely interview questions

1. **Why must documents and queries use the same tokenizer?** Matching is exact string
   equality on terms. If documents are stemmed and queries are not, "infections" never
   finds the term "infect".
2. **What is the Porter "measure" m?** Write any word as `[C](VC)^m[V]`. m counts the
   vowel-consonant pairs, a rough proxy for syllables. Rules like "remove -ment if
   m > 1" stop the algorithm from stripping short words down to nothing ("cement"
   keeps its ending, "replacement" loses it).
3. **Why not lemmatize instead of stemming?** Lemmatization needs a dictionary and
   part-of-speech tags, which is slower and language-specific. For ranking, all we
   need is that related forms collide, and they don't need to be real words. Porter is
   fast, deterministic and is what the Anserini baseline we compare against uses.

---

## Phase 3: Inverted index

### What was built

- `InvertedIndex` (`include/bm25/inverted_index.hpp`):
  `unordered_map<string, vector<Posting{doc_id, tf}>>`, plus a
  `vector<uint32_t> doc_lengths_` indexed by doc id, the external id strings, and
  `avgdl`.
- `build(docs, tokenizer)` makes one pass. For each document: tokenize, count term
  frequencies in a reusable per-document hash map, then append one `Posting` to each
  term's list. Documents are visited in increasing doc-id order, so every posting list
  comes out sorted with no sort step.
- Accessors: `postings(term)` (returns a shared static empty vector for unknown terms,
  so callers never need a null check), `doc_freq`, `doc_length`, `num_docs`, `avgdl`,
  `external_id`, and `stats()`.
- `apps/bm25_index`: loads SciFact, times the build and prints the statistics.
  `--no-stem` and `--no-stopwords` are there for the ablations.
- 10 tests on a hand-made 4-document corpus that assert exact postings, document
  frequencies, document lengths and avgdl, both with and without stopwords. Edge cases:
  an empty document, an empty corpus, and rebuilding.

### Evidence

```
$ ctest --test-dir build --output-on-failure
100% tests passed out of 38

$ ./build/apps/bm25_index
tokenizer:        stopwords=on stem=on
corpus load:      127.1 ms
index build:      386.9 ms (tokenize + invert)
documents:        5183
vocabulary size:  26156
total postings:   512968
total tokens:     850090
avgdl (tokens):   164.0150
memory (approx):  7.83 MB
```

Build time is noisy on this machine: four Release runs gave 580.3, 576.9, 499.2 and
386.9 ms. Phase 6 will measure it properly, with repeats.

### How the memory number is computed

`stats()` adds up what the containers own: for each hash-map node, its key string
object, vector object, next pointer and cached hash; key characters that don't fit in
the small-string buffer; posting-list *capacity* × 8 bytes; the bucket array; and the
doc lengths and external ids. It ignores allocator headers and padding, so the real
resident size is somewhat higher. Postings dominate: 512,968 × 8 B ≈ 3.9 MB of the 7.83.

### Design choices

- **Hash map, not a sorted map (`std::map`).** The only query operation is "give me
  the list for exactly this term". A hash map does that in expected O(1) (after
  hashing the term); a red-black tree takes O(log V) string comparisons (about 15 for
  V = 26k) and chases pointers between scattered nodes. A sorted map would only pay off
  for prefix or wildcard queries ("vitam*") or for writing terms in order. We don't
  support those queries, and if we save the index we can sort at that point.
- **`uint32_t` doc ids and tf.** 4 billion documents is plenty, and an 8-byte Posting
  halves memory compared with two `size_t`s, which also means more postings per cache
  line during scoring.
- **Doc length = tokens after stopword removal.** That is what Lucene counts (the tokens
  actually indexed). The length is what BM25 normalises by, so it must count the same
  units as tf.
- **Reusing the per-document `tf` map** (`clear()` keeps its buckets) avoids reallocating
  a hash table for each of the 5,183 documents.

### Complexity

Let T be the total tokens (850,090), V the vocabulary (26,156), P the total postings
(512,968), N the number of documents and L the average token length.

- **Build:** O(T·L) expected time. Tokenizing is linear in the characters. Each token
  costs one hash of its characters plus an expected O(1) map update. Appending postings
  is amortised O(1) per posting, and P ≤ T.
- **Term lookup:** O(L) expected for the hash, plus O(1) for the bucket. The worst case
  (every term in one bucket) is O(V), which doesn't happen in practice with a decent hash.
- **Space:** O(V·L + P + N).

### Likely interview questions

1. **Why are posting lists sorted by doc id, and how do you guarantee it?** Sorted
   lists allow linear-time intersection and merging (AND queries, document-at-a-time
   scoring, skip pointers) and compress well with delta encoding. We guarantee it for
   free by building in doc-id order and only ever appending.
2. **Hash map vs trie vs sorted array for the dictionary?** A hash map is fastest for
   exact lookups. A trie or FST supports prefixes and shares common prefixes, which saves
   memory (Lucene uses an FST). A sorted array with binary search is compact and good for
   an on-disk index. For an in-memory exact-match engine, a hash map is the simplest
   choice that is also fast.
3. **What would change for a corpus that doesn't fit in memory?** Index in blocks: build
   partial in-memory indexes, write each one sorted by term to disk, then do a k-way
   merge (SPIMI / external merge sort). Posting lists would be delta-encoded and
   compressed (variable-byte or PForDelta), and the dictionary would map each term to a
   file offset.
