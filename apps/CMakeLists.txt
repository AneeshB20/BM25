add_executable(bm25_stats bm25_stats.cpp)
target_link_libraries(bm25_stats PRIVATE bm25)
bm25_set_warnings(bm25_stats)

add_executable(bm25_index bm25_index.cpp)
target_link_libraries(bm25_index PRIVATE bm25)
bm25_set_warnings(bm25_index)
