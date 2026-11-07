# Performance on BEIR SciFact

All numbers come from `scripts/bench.sh 3`, which runs `build/apps/bm25_eval --bench`
three times pinned to one core (`taskset -c 4`). Each run:

- spins for 15 s untimed (see "Measurement notes"),
- builds the index 5 times and discards the first,
- runs all 300 judged test queries 5 times (top-10), discards the first pass, and
  reports nearest-rank percentiles over the remaining 1,200 query timings.

Each cell is the **median of the three trials**; the per-trial values are in brackets.

Machine: AMD Ryzen 7 7435HS (laptop), Ubuntu 24.04, g++ 13.3.0, `-O3` (CMake Release),
single-threaded. The power profile was left in its default `power-saver` mode.

## Before / after, one change at a time

| Step | Change | Index build (ms) | Query p50 (µs) | Query p95 (µs) | Query p99 (µs) | Query mean (µs) |
|---|---|---:|---:|---:|---:|---:|
| 0 | Baseline (Phase 5 code) | 176.7 [169.5, 176.7, 183.0] | 159.9 [159.9, 159.6, 160.6] | 240.6 [240.6, 242.2, 232.8] | 257.3 [257.3, 257.7, 252.7] | 152.1 [152.9, 152.1, 151.5] |
| 1 | Dense score accumulator instead of `unordered_map` | 169.4 [168.4, 169.4, 172.0] | 32.3 [32.2, 32.7, 32.3] | 68.6 [68.6, 69.2, 68.6] | 78.1 [78.0, 80.2, 78.1] | 34.1 [34.0, 34.5, 34.1] |
| 2 | Precomputed per-document length normalisation | 168.6 [168.6, 167.7, 168.9] | 27.5 [27.8, 27.3, 27.5] | 57.9 [58.0, 57.5, 57.9] | 67.1 [67.3, 65.6, 67.1] | 28.7 [29.1, 28.7, 28.7] |
| 3 | Term interning to integer ids during indexing | 149.8 [149.3, 149.8, 151.0] | 27.7 [31.4, 27.7, 27.7] | 54.7 [65.5, 54.5, 54.7] | 62.5 [79.5, 62.5, 62.4] | 28.8 [33.2, 28.8, 28.8] |

Overall: query p50 **159.9 → 27.7 µs (5.8× faster)**, p99 257.3 → 62.5 µs, and index
build 176.7 → 149.8 ms (−15%).

Each step changes one thing. Step 1 does not touch the build, so its 7 ms build
difference from the baseline is run-to-run noise (the baseline trials alone span
169.5–183.0 ms). Step 3 does not touch the query path; the query numbers in its first
trial were slower than in the other two, which is also noise.

## Correctness gate

After every step, `scripts/verify.sh` was run and passed:

```
== Phase 4 cross-check
max |score diff|: 4.992e-10 (tolerance 1e-06)
matched: 50/50
== Phase 5 evaluation
nDCG@10:     0.6774
Recall@100:  0.9287
run file: identical to results/bm25_run.trec
```

The run file (300 queries × top-100, scores to 9 decimals) stayed **byte-identical** to
the Phase 5 file through every change, so nDCG@10 and Recall@100 are unchanged by
construction.

## Index save / load

`build/apps/bm25_index --save data/scifact/index.bin`, then
`scripts/bench.sh 3 --index data/scifact/index.bin` (after step 3):

| | Time (ms) | Notes |
|---|---:|---|
| Rebuild index from parsed documents | 152.5 [152.7, 152.5, 150.0] | tokenize + stem + invert |
| Load saved index | **5.7** [5.7, 5.6, 5.7] | file in the OS page cache |
| Saved file size | 4.45 MiB (4,668,814 bytes) | `corpus.jsonl` is 7.73 MiB (8,106,566 bytes) |

Loading is about **27× faster** than rebuilding. That understates the gap, because a
rebuild also has to parse `corpus.jsonl` first (`bm25_index` reported a corpus load of
168.5–169.0 ms in three runs without warm-up). The benchmark also checked that the
loaded index returns exactly the same results (doc ids and score bits) as the built
one on all 300 queries, and `bm25_eval --index data/scifact/index.bin` reproduces
`results/bm25_run.trec` byte for byte.

## Measurement notes

- **CPU frequency scaling dominated the first measurements.** With the laptop in
  `power-saver` mode, identical binaries measured a 562 ms build / 522 µs p50 in one run
  and a 170 ms build / 159 µs p50 in another. The CPU sits in a low-clock state until it
  has been under sustained load for several seconds. A 3 s warm-up was not enough (three
  runs all stayed in the slow state); a 15 s warm-up gave three consistent runs. The
  power profile was not changed, since that is a system setting.
- `perf` is unavailable (`perf_event_paranoid = 4`, no root), so profiling used gprof
  (`-pg`). See NOTES.md, Phase 6.
- Latency is measured around `Searcher::search` only (tokenize + score + top-k). Mapping
  doc ids to strings and printing are excluded.
