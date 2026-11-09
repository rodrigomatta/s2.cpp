// The pre-tokenizer against the reference: every expected split below is what Hugging Face tokenizers
// (0.22) returns for this repo's tokenizer.json, on inputs a hand-written matcher can get wrong.
// Non-ASCII text is written as UTF-8 bytes, so the test does not depend on the compiler's source charset.
#include "s2_tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * name;
    std::string text;
    std::vector<std::string> pieces;
};

}

int main() {
    const Case cases[] = {
        {"contraction", "He's here", {"He", "'s", " here"}},
        {"contractions ignore case", "IT'S", {"IT", "'S"}},
        // U+017F LATIN SMALL LETTER LONG S case-folds to 's'
        {"long s in a contraction", "'\xC5\xBF" "a", {"'\xC5\xBF", "a"}},
        // Thai: \p{L}+ runs stop at the combining vowel marks (Mn), not at word ends
        {"Thai",
         "\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5\xE0\xB8\x84\xE0\xB8\xA3\xE0\xB8\xB1\xE0\xB8\x9A"
         " \xE0\xB8\x9B\xE0\xB8\xA3\xE0\xB8\xB0\xE0\xB9\x80\xE0\xB8\x97\xE0\xB8\xA8\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2",
         {"\xE0\xB8\xAA\xE0\xB8\xA7", "\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94", "\xE0\xB8\xB5\xE0\xB8\x84\xE0\xB8\xA3",
          "\xE0\xB8\xB1\xE0\xB8\x9A",
          " \xE0\xB8\x9B\xE0\xB8\xA3\xE0\xB8\xB0\xE0\xB9\x80\xE0\xB8\x97\xE0\xB8\xA8\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2"}},
        // U+2EBF0, a letter since Unicode 15.1, stays in its letter run
        {"letter new in Unicode 15.1", "a\xF0\xAE\xAF\xB0" "b", {"a\xF0\xAE\xAF\xB0" "b"}},
        {"digits, spaces, punctuation and newlines", "x  12 y!!\n\nz", {"x", " ", " ", "1", "2", " y", "!!\n\n", "z"}},
    };

    int failed = 0;
    for (const Case & c : cases) {
        const std::vector<std::string> got = s2::pre_tokenize(c.text);
        if (got == c.pieces) continue;
        ++failed;
        std::printf("FAIL %s:", c.name);
        for (const std::string & p : got) {
            std::printf(" [");
            for (unsigned char b : p) std::printf(b < 0x80 && b >= 0x20 ? "%c" : "\\x%02X", b);
            std::printf("]");
        }
        std::printf("\n");
    }
    std::printf("%d of %d cases failed\n", failed, (int) (sizeof(cases) / sizeof(cases[0])));
    return failed ? 1 : 0;
}
