cmake_minimum_required(VERSION 3.16)
project(bm25 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
endif()

option(BM25_BUILD_TESTS "Build the GoogleTest unit tests" ON)

include(FetchContent)

# nlohmann/json: header-only, used only to parse the BEIR JSONL files.
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Warnings apply to our own targets only, never to third-party code.
function(bm25_set_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

add_library(bm25
  src/corpus.cpp
)
target_include_directories(bm25 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bm25 PRIVATE nlohmann_json::nlohmann_json)
bm25_set_warnings(bm25)

add_subdirectory(apps)

if(BM25_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
