add_executable(bm25_stats bm25_stats.cpp)
target_link_libraries(bm25_stats PRIVATE bm25)
bm25_set_warnings(bm25_stats)
