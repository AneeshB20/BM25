FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(bm25_tests
  placeholder_test.cpp
  corpus_test.cpp
  tokenizer_test.cpp
  porter_stemmer_test.cpp
  inverted_index_test.cpp
  bm25_scorer_test.cpp
  searcher_test.cpp
)
target_link_libraries(bm25_tests PRIVATE bm25 GTest::gtest_main)
bm25_set_warnings(bm25_tests)
# Lets tests find optional reference data under data/ (e.g. the Porter vocabulary).
target_compile_definitions(bm25_tests PRIVATE BM25_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(bm25_tests)
