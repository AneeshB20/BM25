#include "bm25/porter_stemmer.hpp"

#include <cstring>

// A direct C++ translation of Martin Porter's reference ANSI C implementation
// (https://tartarus.org/martin/PorterStemmer/c.txt). Notation from the paper:
//   C = a run of consonants, V = a run of vowels.
//   Any word has the form [C](VC)^m[V]; m is the "measure" of the word.
// Each step strips or rewrites one suffix when the remaining stem satisfies a
// condition on m.
//
// The reference implementation differs from the 1980 paper in three small
// places, all kept here because the official test vocabulary relies on them:
//   step 2 "bli" -> "ble" (paper: "abli" -> "able"), step 2 "logi" -> "log"
//   (not in the paper), and words of length <= 2 are never stemmed.

namespace bm25 {

namespace {

// Holds the word being stemmed. `k` is the index of the last character of
// the current word; `j` marks the end of the stem left after a suffix
// matched in ends(). Both are signed because j may become -1.
class Stemming {
public:
    explicit Stemming(std::string_view word) : b_(word), k_(static_cast<int>(word.size()) - 1) {}

    std::string run() {
        if (k_ <= 1) {
            return b_;  // length <= 2: unchanged
        }
        step1ab();
        if (k_ > 0) {
            step1c();
            step2();
            step3();
            step4();
            step5();
        }
        return b_.substr(0, static_cast<std::size_t>(k_ + 1));
    }

private:
    std::string b_;
    int k_;
    int j_ = 0;

    char at(int i) const { return b_[static_cast<std::size_t>(i)]; }

    // True if b[i] is a consonant. 'y' is a consonant at the start of a word or
    // after a vowel, and a vowel after a consonant ("toy" vs "syzygy").
    bool cons(int i) const {
        switch (at(i)) {
            case 'a': case 'e': case 'i': case 'o': case 'u':
                return false;
            case 'y':
                return i == 0 ? true : !cons(i - 1);
            default:
                return true;
        }
    }

    // m(): the number of VC sequences in b[0..j].
    //   <c><v>       -> 0     tr, ee, tree, y, by
    //   <c>vc<v>     -> 1     trouble, oats, trees, ivy
    //   <c>vcvc<v>   -> 2     troubles, private, oaten
    int m() const {
        int n = 0;
        int i = 0;
        while (true) {  // skip the optional leading consonants
            if (i > j_) return n;
            if (!cons(i)) break;
            ++i;
        }
        ++i;
        while (true) {
            while (true) {  // skip vowels
                if (i > j_) return n;
                if (cons(i)) break;
                ++i;
            }
            ++i;
            ++n;  // completed one VC
            while (true) {  // skip consonants
                if (i > j_) return n;
                if (!cons(i)) break;
                ++i;
            }
            ++i;
        }
    }

    // True if b[0..j] contains a vowel.
    bool vowel_in_stem() const {
        for (int i = 0; i <= j_; ++i) {
            if (!cons(i)) return true;
        }
        return false;
    }

    // True if b[i-1..i] is a double consonant.
    bool double_c(int i) const {
        if (i < 1) return false;
        if (at(i) != at(i - 1)) return false;
        return cons(i);
    }

    // True if b[i-2..i] is consonant-vowel-consonant and the last consonant is
    // not w, x or y. Used to restore an 'e' (cav(e), lov(e), hop(e)) and to
    // avoid removing one (cast, snow, box stay as they are).
    bool cvc(int i) const {
        if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
        const char ch = at(i);
        return !(ch == 'w' || ch == 'x' || ch == 'y');
    }

    // True if b[0..k] ends with s; on a match, sets j to the end of the stem.
    bool ends(const char* s) {
        const int len = static_cast<int>(std::strlen(s));
        if (len > k_ + 1) return false;
        if (b_.compare(static_cast<std::size_t>(k_ - len + 1), static_cast<std::size_t>(len), s) != 0) {
            return false;
        }
        j_ = k_ - len;
        return true;
    }

    // Replaces b[j+1..k] with s and moves k to the new end.
    void set_to(const char* s) {
        const int len = static_cast<int>(std::strlen(s));
        b_.replace(static_cast<std::size_t>(j_ + 1), static_cast<std::size_t>(k_ - j_), s);
        k_ = j_ + len;
    }

    // set_to(s) only if the stem has measure > 0.
    void r(const char* s) {
        if (m() > 0) set_to(s);
    }

    // Step 1ab: plurals and -ed / -ing.
    //   caresses -> caress, ponies -> poni, cats -> cat
    //   feed -> feed, agreed -> agree, plastered -> plaster, motoring -> motor
    //   then tidy-ups: conflat(ed) -> conflate, hopp(ing) -> hop, fil(ing) -> file
    void step1ab() {
        if (at(k_) == 's') {
            if (ends("sses")) {
                k_ -= 2;
            } else if (ends("ies")) {
                set_to("i");
            } else if (at(k_ - 1) != 's') {
                --k_;
            }
        }
        if (ends("eed")) {
            if (m() > 0) --k_;
        } else if ((ends("ed") || ends("ing")) && vowel_in_stem()) {
            k_ = j_;
            if (ends("at")) {
                set_to("ate");
            } else if (ends("bl")) {
                set_to("ble");
            } else if (ends("iz")) {
                set_to("ize");
            } else if (double_c(k_)) {
                --k_;
                const char ch = at(k_);
                if (ch == 'l' || ch == 's' || ch == 'z') ++k_;
            } else if (m() == 1 && cvc(k_)) {
                // Here j == k (the failed ends() calls above leave j alone), so
                // m() measures the whole remaining stem.
                set_to("e");
            }
        }
    }

    // Step 1c: terminal y -> i when there is another vowel in the stem.
    void step1c() {
        if (ends("y") && vowel_in_stem()) {
            b_[static_cast<std::size_t>(k_)] = 'i';
        }
    }

    // Step 2: double suffixes -> single ones, when m > 0.
    // Switches on the penultimate letter to avoid testing every suffix.
    void step2() {
        switch (at(k_ - 1)) {
            case 'a':
                if (ends("ational")) { r("ate"); break; }
                if (ends("tional")) { r("tion"); break; }
                break;
            case 'c':
                if (ends("enci")) { r("ence"); break; }
                if (ends("anci")) { r("ance"); break; }
                break;
            case 'e':
                if (ends("izer")) { r("ize"); break; }
                break;
            case 'l':
                if (ends("bli")) { r("ble"); break; }
                if (ends("alli")) { r("al"); break; }
                if (ends("entli")) { r("ent"); break; }
                if (ends("eli")) { r("e"); break; }
                if (ends("ousli")) { r("ous"); break; }
                break;
            case 'o':
                if (ends("ization")) { r("ize"); break; }
                if (ends("ation")) { r("ate"); break; }
                if (ends("ator")) { r("ate"); break; }
                break;
            case 's':
                if (ends("alism")) { r("al"); break; }
                if (ends("iveness")) { r("ive"); break; }
                if (ends("fulness")) { r("ful"); break; }
                if (ends("ousness")) { r("ous"); break; }
                break;
            case 't':
                if (ends("aliti")) { r("al"); break; }
                if (ends("iviti")) { r("ive"); break; }
                if (ends("biliti")) { r("ble"); break; }
                break;
            case 'g':
                if (ends("logi")) { r("log"); break; }
                break;
            default:
                break;
        }
    }

    // Step 3: -ic-, -full, -ness etc., when m > 0.
    void step3() {
        switch (at(k_)) {
            case 'e':
                if (ends("icate")) { r("ic"); break; }
                if (ends("ative")) { r(""); break; }
                if (ends("alize")) { r("al"); break; }
                break;
            case 'i':
                if (ends("iciti")) { r("ic"); break; }
                break;
            case 'l':
                if (ends("ical")) { r("ic"); break; }
                if (ends("ful")) { r(""); break; }
                break;
            case 's':
                if (ends("ness")) { r(""); break; }
                break;
            default:
                break;
        }
    }

    // Step 4: remove -ant, -ence etc. when m > 1 (the stem is "long enough").
    void step4() {
        switch (at(k_ - 1)) {
            case 'a':
                if (ends("al")) break;
                return;
            case 'c':
                if (ends("ance")) break;
                if (ends("ence")) break;
                return;
            case 'e':
                if (ends("er")) break;
                return;
            case 'i':
                if (ends("ic")) break;
                return;
            case 'l':
                if (ends("able")) break;
                if (ends("ible")) break;
                return;
            case 'n':
                if (ends("ant")) break;
                if (ends("ement")) break;
                if (ends("ment")) break;
                if (ends("ent")) break;
                return;
            case 'o':
                // -ion is removed only after s or t: adoption -> adopt, but not onion.
                if (ends("ion") && j_ >= 0 && (at(j_) == 's' || at(j_) == 't')) break;
                if (ends("ou")) break;
                return;
            case 's':
                if (ends("ism")) break;
                return;
            case 't':
                if (ends("ate")) break;
                if (ends("iti")) break;
                return;
            case 'u':
                if (ends("ous")) break;
                return;
            case 'v':
                if (ends("ive")) break;
                return;
            case 'z':
                if (ends("ize")) break;
                return;
            default:
                return;
        }
        if (m() > 1) k_ = j_;
    }

    // Step 5a: remove a final -e if m > 1, or if m == 1 and the stem is not cvc.
    // Step 5b: -ll -> -l when m > 1.
    void step5() {
        j_ = k_;
        if (at(k_) == 'e') {
            const int a = m();
            if (a > 1 || (a == 1 && !cvc(k_ - 1))) --k_;
        }
        if (at(k_) == 'l' && double_c(k_) && m() > 1) --k_;
    }
};

}  // namespace

std::string PorterStemmer::stem(std::string_view word) const {
    return Stemming(word).run();
}

}  // namespace bm25
