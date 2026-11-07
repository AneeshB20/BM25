#include "bm25/corpus.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace {

// Writes `content` to a uniquely named temporary file and returns its path.
std::string write_temp(const std::string& name, const std::string& content) {
    const std::string path = ::testing::TempDir() + "bm25_corpus_test_" + name;
    std::ofstream out(path);
    out << content;
    return path;
}

}  // namespace

TEST(Corpus, JoinsTitleAndTextWithSpace) {
    const std::string path = write_temp(
        "corpus.jsonl",
        R"({"_id": "d1", "title": "Hello", "text": "world", "metadata": {}})"
        "\n"
        R"({"_id": "d2", "title": "", "text": "no title"})"
        "\n");
    const auto docs = bm25::Corpus::load_documents(path);
    ASSERT_EQ(docs.size(), 2u);
    EXPECT_EQ(docs[0].id, "d1");
    EXPECT_EQ(docs[0].text, "Hello world");
    EXPECT_EQ(docs[0].title, "Hello");
    EXPECT_EQ(docs[1].id, "d2");
    EXPECT_EQ(docs[1].text, " no title");
}

TEST(Corpus, LoadsQueries) {
    const std::string path = write_temp(
        "queries.jsonl",
        R"({"_id": "1", "text": "first query", "metadata": {}})"
        "\n\n"
        R"({"_id": "3", "text": "second"})"
        "\n");
    const auto qs = bm25::Corpus::load_queries(path);
    ASSERT_EQ(qs.size(), 2u);
    EXPECT_EQ(qs[0].id, "1");
    EXPECT_EQ(qs[0].text, "first query");
    EXPECT_EQ(qs[1].id, "3");
}

TEST(Corpus, LoadsQrelsAndSkipsHeader) {
    const std::string path = write_temp(
        "qrels.tsv", "query-id\tcorpus-id\tscore\n1\td1\t1\n1\td7\t2\n5\td2\t1\n");
    const bm25::Qrels qrels = bm25::Corpus::load_qrels(path);
    ASSERT_EQ(qrels.size(), 2u);
    EXPECT_EQ(qrels.at("1").size(), 2u);
    EXPECT_EQ(qrels.at("1").at("d7"), 2);
    EXPECT_EQ(qrels.at("5").at("d2"), 1);
}

TEST(Corpus, MissingFileThrows) {
    EXPECT_THROW(bm25::Corpus::load_documents("/nonexistent/corpus.jsonl"), std::runtime_error);
}

TEST(Corpus, MalformedJsonThrows) {
    const std::string path = write_temp("bad.jsonl", "{not json}\n");
    EXPECT_THROW(bm25::Corpus::load_documents(path), std::runtime_error);
}

TEST(Corpus, LoadDirectoryAndJudgedQueries) {
    // A tiny BEIR-style directory: 2 docs, 3 queries, qrels for 2 of them.
    const std::string dir = ::testing::TempDir() + "bm25_corpus_dir";
    std::filesystem::create_directories(dir + "/qrels");
    std::ofstream(dir + "/corpus.jsonl")
        << R"({"_id": "d1", "title": "T1", "text": "a"})" "\n"
        << R"({"_id": "d2", "title": "T2", "text": "b"})" "\n";
    std::ofstream(dir + "/queries.jsonl")
        << R"({"_id": "q1", "text": "one"})" "\n"
        << R"({"_id": "q2", "text": "two"})" "\n"
        << R"({"_id": "q3", "text": "three"})" "\n";
    std::ofstream(dir + "/qrels/test.tsv") << "query-id\tcorpus-id\tscore\nq3\td2\t1\nq1\td1\t1\n";

    const bm25::Corpus c = bm25::Corpus::load(dir);
    EXPECT_EQ(c.documents().size(), 2u);
    EXPECT_EQ(c.queries().size(), 3u);
    const auto judged = c.judged_queries();
    ASSERT_EQ(judged.size(), 2u);
    EXPECT_EQ(judged[0].id, "q1");  // queries.jsonl order, not qrels order
    EXPECT_EQ(judged[1].id, "q3");
}
