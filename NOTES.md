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

---

## Phase 4: BM25 scoring and top-k

### What was built

- `Bm25Scorer` (`include/bm25/bm25_scorer.hpp`): `idf(doc_freq)` and
  `term_score(idf, tf, doc_len)`, using exactly the CLAUDE.md formula with k1 and b as
  parameters. It holds only N, avgdl, k1 and b; it doesn't know about the index, which
  makes it trivial to unit-test.
- `Searcher` (`include/bm25/searcher.hpp`): `search(query, k)`:
  1. tokenize with the same options as the index;
  2. drop repeated terms, keeping first-occurrence order;
  3. **term-at-a-time** scoring: for each term, compute idf once, then walk its
     posting list and add `term_score` into an `unordered_map<doc_id, double>`
     accumulator;
  4. select the top-k with a **size-k min-heap** (`std::priority_queue` whose top is
     the worst result kept so far);
  5. pop and reverse into best-first order. Ties go to the lower doc id, so output is
     deterministic even though the hash-map iteration order is not.
- `apps/bm25_search --batch`: reads `qid<TAB>text` from stdin and writes a TREC run
  with scores printed to 9 decimal places.
- `apps/bm25_index --export-tokens DIR`: writes the C++ tokenizer's output for every
  document and judged test query.
- `scripts/reference_bm25.py`: a separate, deliberately simple Python BM25. It reads
  the exported tokens, builds its own postings with `Counter`, scores with the same
  formula, ranks with a full `sorted()`, and then calls the C++ binary on the same
  queries and compares.

### Evidence

Unit tests (21 new; scores computed by hand in the test comments, asserted to 1e-6):

```
$ ctest --test-dir build --output-on-failure
...
100% tests passed out of 59
```

Python cross-check:

```
$ build/apps/bm25_index --export-tokens data/scifact/tokens
$ python3 scripts/reference_bm25.py
reference index: 5183 docs, 26156 terms, avgdl 164.0150
queries compared: 50 (seed 42, top-10, k1=0.9, b=0.4)
max |score diff|: 4.992e-10 (tolerance 1e-06)
matched: 50/50
```

Going beyond the requirement: all 300 queries match with the defaults (300/300, max
diff 4.993e-10), and also with top-100 at k1=1.2, b=0.75 (300/300, max diff 5.000e-10).
The ~5e-10 is only the rounding from printing 9 decimals. The doubles themselves are
identical, because both sides apply the same operations in the same order.

Edge cases. The unit tests cover an empty query, a punctuation-only query, unknown
terms, unknown terms mixed with known ones, k = 0, k larger than the number of matches,
a stopword-only query, and equal-score ties. On SciFact itself, with `-k 1000`:
`zzqxv qqqzzk` → 0 lines, an empty query → 0 lines, `the of and is` → 0 lines,
`parthenogenesis` → 2 lines (and the stemmed term `parthenogenesi` is in exactly 2
documents).

### The formula, term by term

```
idf(t)  = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
score  += idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
```

- **N** is the number of documents and **n_t** the number containing term t. The ratio
  `(N - n_t + 0.5) / (n_t + 0.5)` is roughly "documents without t / documents with t":
  the odds against a random document containing t. Rare terms have high odds, so they
  carry more information. The 0.5s are smoothing, so n_t = 0 or n_t = N never divides
  by zero.
- **The `1 +` inside the log** is the Lucene variant's fix. Classic Robertson–Spärck
  Jones IDF, `ln((N - n + 0.5)/(n + 0.5))`, goes *negative* for terms in more than half
  the documents, so matching a common word would lower a document's score. With `1 +`,
  IDF is always > 0. In the test corpus a term in all 4 documents still gets
  ln(10/9) ≈ 0.105.
- **tf** is how often t appears in the document. More occurrences is more evidence,
  but with diminishing returns.
- **`tf * (k1 + 1) / (tf + K)`** is the saturation curve. As tf → ∞ it approaches
  `k1 + 1`. The tenth "cancer" in a document adds much less than the first. The
  `(k1 + 1)` factor only scales the result so that tf = 1 gives about 1 in an
  average-length document; it doesn't change any ranking.
- **`K = k1 * (1 - b + b * dl / avgdl)`** is length normalisation. A document twice as
  long as average gets a larger K, so each occurrence counts for less, because long
  documents mention many words just by being long.

**What k1 controls:** how fast tf saturates. k1 = 0 ignores tf entirely (a binary
"contains the term" model); a large k1 makes the score close to linear in tf. Our
k1 = 0.9 saturates fairly quickly. In the tests, tf = 1000 gives 1.89 against a
ceiling of 1.9.

**What b controls:** how strongly document length is normalised, from 0 to 1. b = 0
means no normalisation (test `BZeroIgnoresLength`); b = 1 means fully dividing by
relative length. b = 0.4 is mild. SciFact abstracts are fairly uniform in length, and
long abstracts are not mostly padding.

### Why repeated query terms count once

A query like "cancer cells and cancer risk" says "cancer" twice because of grammar, not
because the user wants "cancer" weighted double. Counting once makes the score a
function of the *set* of query terms, which is the behaviour the prompt asked for and
which the Python reference copies with `dict.fromkeys`. Note that this differs from
Anserini, whose bag-of-words query boosts a repeated term by its count (see Phase 0
risks). For short SciFact claims it rarely matters. If it did, it would be a one-line
change: multiply the term's contribution by its query count.

### Design choices

- **Term-at-a-time with an accumulator** is the simplest correct strategy. The
  alternative, document-at-a-time (merging the sorted posting lists with a cursor per
  term), uses less memory and enables WAND/MaxScore early termination, but it is more
  complex. With 5,183 documents the accumulator is tiny.
- **Hash-map accumulator for now.** A dense `vector<double>(N)` would avoid hashing, but
  it costs O(N) to clear on every query. Phase 6 will measure that trade instead of
  guessing.
- **Size-k min-heap rather than sorting.** For M matching documents, a full sort is
  O(M log M); the heap is O(M log k) time and O(k) extra space. With k = 10 and M often
  in the thousands, that is a real saving, and CLAUDE.md requires it.
- **double, not float**, for scores. It gives about 16 significant digits, so summation
  order barely matters and the 1e-6 cross-check is easy to meet.
- **The C++ expression order matches the formula** (`b * dl / avgdl`, not
  `b * (dl / avgdl)`). The two round differently, and keeping them the same is what
  makes the C++ and Python doubles identical.
- **Why the Python side reads exported tokens:** CLAUDE.md forbids re-implementing the
  engine in Python, and sharing the tokens means any mismatch must come from
  indexing/scoring. The tokenizer has its own unit tests and a 23,531-word stemmer test.

### Complexity

For a query with q unique terms, posting-list lengths p_1..p_q, and M = number of
distinct matching documents (M ≤ Σp_i ≤ N):

- Tokenizing the query: O(query length).
- Scoring: O(Σ p_i) expected (one accumulator update per posting).
- Top-k: O(M log k) time, O(k) space for the heap.
- Accumulator space: O(M).
- Total: O(Σ p_i + M log k). The cost is driven by how common the query terms are,
  not by N. That's the whole point of an inverted index.

### Likely interview questions

1. **Why a min-heap for top-k, and why "min"?** We want the k *largest* scores. The heap
   keeps the k best seen so far with the *smallest* of them on top, so for each new
   candidate one comparison against the top decides whether it gets in. If it does, we
   pop the smallest and push the new one in O(log k). A max-heap of all M candidates
   would need O(M) space and O(M + k log M) time.
2. **Can a common term ever lower a document's score?** Not with this IDF. The `1 +`
   inside the log keeps IDF positive, so every matching term adds a positive amount.
   With the classic RSJ IDF it could, for terms in more than half the documents.
3. **How did you verify the scorer is correct?** In three independent ways: unit tests
   whose expected values were derived as exact fractions in the comments; a separate
   Python implementation that builds its own index from shared tokens and agrees on all
   300 queries with identical doubles (only print rounding differs); and edge-case tests
   for empty, unknown and stopword-only queries, k = 0, k > matches, and ties.

---

## Phase 5: Evaluation on SciFact

### What was built

- `Metrics` (`include/bm25/metrics.hpp`): `ndcg_at_k` and `recall_at_k`, written by
  hand and matching trec_eval's definitions (linear gain, log2(rank+1) discount, ideal
  DCG built from *all* judged relevant documents, not only the retrieved ones).
  9 unit tests with hand-derived expected values.
- `apps/bm25_eval`: runs all 300 judged test queries at depth 100, writes
  `results/bm25_run.trec` (`qid Q0 docid rank score bm25`) and per-query metrics in
  `results/bm25_metrics.tsv`, and prints the means. `--ablations FILE` runs the
  tokenizer ablations and the k1 × b grid and writes `results/ablations.md`.
- `scripts/eval.py`: evaluates any run file with **pytrec_eval** (the official trec_eval
  C code wrapped for Python) and, with `--cpp-metrics`, compares the C++ numbers per
  query and on the mean. It averages over *all* qrels queries, counting a query with no
  results as 0, because pytrec_eval on its own silently drops such queries and inflates
  the mean.
- Environment note: the `pytrec_eval` package didn't compile here (no Python headers,
  and no sudo to install them), so `scripts/requirements.txt` uses
  `pytrec-eval-terrier`. It ships prebuilt wheels of the same code under the same
  import name.

### Main result

```
$ ./build/apps/bm25_eval
config:      stem=on stopwords=on k1=0.90 b=0.40 depth=100
queries:     300
nDCG@10:     0.6774
Recall@100:  0.9287
```

**nDCG@10 = 0.6774**, inside the 0.66–0.68 target, so the parameters were left alone.

Cross-check against pytrec_eval:

```
$ .venv/bin/python scripts/eval.py results/bm25_run.trec --cpp-metrics results/bm25_metrics.tsv
results/bm25_run.trec: queries=300 missing_from_run=0 nDCG@10=0.6774 Recall@100=0.9287
C++ vs pytrec_eval:
  ndcg_cut_10  C++ 0.677399  pytrec_eval 0.677399  |diff of means| 9.12e-08  max per-query |diff| 4.80e-07  OK (tol 0.001)
  recall_100   C++ 0.928667  pytrec_eval 0.928667  |diff of means| 3.33e-07  max per-query |diff| 0.00e+00  OK (tol 0.001)
```

The C++ and trec_eval numbers agree to within 5e-7 on *every* query, far inside the
0.001 tolerance. What's left is the 6-decimal rounding in the TSV. All 10 ablation run
files were also re-scored with pytrec_eval, and every one reproduces the table below.

### Ablations (`results/ablations.md`, from `bm25_eval --ablations results/ablations.md`)

| Configuration (k1 = 0.9, b = 0.4) | Vocabulary | nDCG@10 | Recall@100 |
|---|---:|---:|---:|
| no stemming, no stopwords | 35734 | 0.6588 | 0.8886 |
| stopwords only | 35701 | 0.6608 | 0.8859 |
| stemming only | 26176 | 0.6812 | 0.9297 |
| stemming + stopwords (main) | 26156 | 0.6774 | 0.9287 |

| k1 | b | nDCG@10 | Recall@100 |
|---:|---:|---:|---:|
| 0.9 | 0.40 | 0.6774 (main) | 0.9287 |
| 0.9 | 0.75 | 0.6831 | 0.9280 |
| 1.2 | 0.40 | 0.6789 | 0.9276 |
| 1.2 | 0.75 | 0.6879 | 0.9276 |
| 1.5 | 0.40 | 0.6834 | 0.9242 |
| 1.5 | 0.75 | 0.6931 | 0.9242 |

Per-query breakdown (from pytrec_eval on the ablation runs):
- stemming vs. no stemming (stopwords off): 37 queries better, 29 worse, 234 unchanged;
- adding stopwords on top of stemming: 10 better, 11 worse, 279 unchanged;
- main configuration: 53 of the 300 queries have nDCG@10 = 0 (no relevant document in
  the top 10).

### What nDCG@10 measures, in plain words

Look at the top 10 results. Each relevant document earns points, and it earns more the
higher it is ranked: a hit at rank 1 is worth 1, at rank 2 about 0.63, at rank 10 about
0.29 (1/log2(rank+1)). Add up the points (DCG) and divide by the best score possible
for that query, if every relevant document had been placed at the top (ideal DCG). So
1.0 means "every relevant document is at the top" and 0 means "none in the top 10". The
final number is the average over the 300 queries. SciFact queries have only 1–5
relevant documents each (339 judgements for 300 queries), so in practice nDCG@10 mostly
measures *how high the first relevant abstract appears*.

**Recall@100** is simpler: what fraction of the relevant documents appear anywhere in
the top 100. It matters when BM25 is the first stage of a pipeline and a re-ranker
reorders those 100. Ours is 0.93: a re-ranker would get to see 93% of what it needs.

### What the ablations taught us

1. **Stemming is the big win:** +0.022 nDCG@10 and +0.041 Recall@100. It shrinks the
   vocabulary by 27% (35,734 → 26,156 terms) by merging forms like
   infection/infections/infected. As expected, it helps recall most. It is not a free
   lunch: it hurt 29 queries, the over-stemming cost described in Phase 2.
2. **Stopword removal barely matters here.** The Lucene list has only 33 words, and BM25's
   IDF already gives words like "the" (which appear in almost every document) a weight
   close to zero. The −0.004 versus "stemming only" comes from 10 queries winning and
   11 losing: noise, not a real effect. Stopwords still earn their place by making
   posting lists shorter (faster queries and a smaller index), not by improving quality.
3. **The k1/b grid prefers more tf weight and more length normalisation** (k1 = 1.5,
   b = 0.75 gives 0.6931). **We did not adopt it.** The grid was evaluated on the *test*
   queries, so picking the best cell would be tuning on the test set, and the number
   would be optimistic. The honest options are to keep the published defaults (which
   we did) or to tune on SciFact's separate train split and report test once.
   Recall@100 moves the other way (0.9287 → 0.9242), a reminder that the parameters
   trade top-heavy precision against depth.
4. **We land near the Anserini baseline without tuning**, which is evidence that the
   tokenizer and formula are right. Earlier phases verified correctness against a
   reference implementation; this phase checks quality against a published result.

### Complexity

- Evaluating one query: O(search) + O(k log R) for the R judged documents
  (`std::map` lookups). Building the ideal DCG is O(R log R). Negligible next to search.
- The whole run: 300 searches; a single untimed pass took about 150 ms in total. Proper
  latency numbers come in Phase 6.

### Likely interview questions

1. **Why nDCG instead of precision@10?** Precision treats rank 1 and rank 10 the same,
   while users mostly look at the top. nDCG discounts by position, supports graded
   relevance, and normalises per query, so queries with 1 or 5 relevant documents are
   comparable.
2. **Your parameter grid found 0.693; why report 0.677?** Because the grid was run on the
   test set. Choosing the best test-set configuration is overfitting the benchmark. The
   0.677 uses published defaults, chosen before looking at test results. To tune
   properly, I would use the train split.
3. **How do you know your nDCG code is right?** It agrees with trec_eval (via
   pytrec_eval) to within 5e-7 on every one of the 300 queries, and the unit tests check
   hand-derived cases: a perfect ranking, the cutoff, graded gain, an ideal DCG that
   includes unretrieved relevant documents, and no relevant documents at all.
