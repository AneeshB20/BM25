#pragma once

// Strict command-line number parsing shared by the apps. std::strtoul and
// friends silently turn "abc" into 0 and accept "-5" for unsigned values,
// which would quietly produce wrong results (e.g. -k abc -> zero hits).
// These helpers reject anything that is not entirely a valid number in range,
// printing an error and exiting with status 2.

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace cli {

[[noreturn]] inline void die(const std::string& flag, const char* value, const char* expected) {
    std::fprintf(stderr, "error: %s expects %s, got '%s'\n", flag.c_str(), expected, value);
    std::exit(2);
}

// An integer >= min_value.
inline std::size_t parse_count(const std::string& flag, const char* value, std::size_t min_value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(value, &end, 10);
    if (value[0] == '\0' || value[0] == '-' || *end != '\0' || errno == ERANGE || v < min_value) {
        die(flag, value, ("an integer >= " + std::to_string(min_value)).c_str());
    }
    return static_cast<std::size_t>(v);
}

// A finite number in [lo, hi].
inline double parse_real(const std::string& flag, const char* value, double lo, double hi) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(value, &end);
    if (value[0] == '\0' || *end != '\0' || errno == ERANGE || !std::isfinite(v) || v < lo || v > hi) {
        die(flag, value, ("a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]").c_str());
    }
    return v;
}

}  // namespace cli
