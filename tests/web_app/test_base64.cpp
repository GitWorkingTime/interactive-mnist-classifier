#include "base64.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ─── Helper ───────────────────────────────────────────────────────────────────
// Encodes a C-string's bytes (excluding the null terminator) and checks the
// result against the expected base64 output.
void check(const char* input, const std::string& expected) {
    std::string result = encode((unsigned char*)input, std::strlen(input));
    if (result != expected) {
        std::cout << "FAIL: \"" << input << "\" -> \"" << result
                  << "\" (expected \"" << expected << "\")\n";
        assert(false);
    }
}

// ─── RFC 4648 test vectors ────────────────────────────────────────────────────
// The canonical base64 examples. The progression f/fo/foo/foob/fooba/foobar
// exercises every padding case (0, 1, and 2 leftover bytes) repeatedly.
void test_rfc4648_vectors() {
    check("", "");
    check("f", "Zg==");
    check("fo", "Zm8=");
    check("foo", "Zm9v");
    check("foob", "Zm9vYg==");
    check("fooba", "Zm9vYmE=");
    check("foobar", "Zm9vYmFy");
    std::cout << "PASSED: test_rfc4648_vectors\n";
}

// ─── Padding cases in isolation ───────────────────────────────────────────────
void test_no_padding() {
    // Length a multiple of 3 → no '=' padding
    check("abc", "YWJj");
    check("abcdef", "YWJjZGVm");
    std::cout << "PASSED: test_no_padding\n";
}

void test_one_pad() {
    // 2 leftover bytes → exactly one '='
    std::string r = encode((unsigned char*)"fo", 2);
    assert(r == "Zm8=");
    assert(r.back() == '=');
    assert(r.size() == 4);
    std::cout << "PASSED: test_one_pad\n";
}

void test_two_pads() {
    // 1 leftover byte → exactly two '=='
    std::string r = encode((unsigned char*)"f", 1);
    assert(r == "Zg==");
    assert(r.substr(2) == "==");
    assert(r.size() == 4);
    std::cout << "PASSED: test_two_pads\n";
}

// ─── Output length is always a multiple of 4 ──────────────────────────────────
void test_output_length_multiple_of_4() {
    const char* inputs[] = {"a", "ab", "abc", "abcd", "abcde", "abcdef"};
    for (const char* in : inputs) {
        std::string r = encode((unsigned char*)in, std::strlen(in));
        assert(r.size() % 4 == 0);
    }
    std::cout << "PASSED: test_output_length_multiple_of_4\n";
}

// ─── Empty input ──────────────────────────────────────────────────────────────
void test_empty() {
    std::string r = encode((unsigned char*)"", 0);
    assert(r.empty());
    std::cout << "PASSED: test_empty\n";
}

// ─── Binary data: null bytes and high bytes (>= 128) ──────────────────────────
// This is the critical test for the WebSocket use case, since SHA-1 output is
// arbitrary binary with null and high-bit bytes. A signed-char implementation
// would corrupt these; unsigned char handles them correctly.
void test_binary_null_bytes() {
    // Three zero bytes encode to "AAAA" (all six-bit groups are 0 -> 'A'),
    // NOT padding. Confirms nulls are treated as real data.
    unsigned char zeros[3] = {0x00, 0x00, 0x00};
    std::string r = encode(zeros, 3);
    assert(r == "AAAA");
    std::cout << "PASSED: test_binary_null_bytes\n";
}

void test_binary_high_bytes() {
    // Bytes with the high bit set must not be sign-extended/corrupted.
    // {0xFF, 0xFF, 0xFF} = 24 ones -> four groups of 111111 (63) -> "////".
    unsigned char highs[3] = {0xFF, 0xFF, 0xFF};
    std::string r = encode(highs, 3);
    assert(r == "////");
    std::cout << "PASSED: test_binary_high_bytes\n";
}

void test_binary_mixed() {
    // A mix of null, high, and mid bytes with a leftover (5 bytes -> one '=').
    unsigned char data[5] = {0x00, 0xFF, 0x80, 0x10, 0x7F};
    std::string r = encode(data, 5);
    assert(r.size() == 8);   // 5 bytes -> ceil(5/3)=2 groups -> 8 chars
    assert(r.back() == '='); // 5 % 3 == 2 leftover -> one pad
    std::cout << "PASSED: test_binary_mixed\n";
}

// ─── 20-byte input (the SHA-1 digest case) ────────────────────────────────────
// SHA-1 produces 20 bytes. 20 % 3 == 2, so the output is 28 chars ending in one
// '='. This is exactly the shape of a Sec-WebSocket-Accept value.
void test_sha1_digest_shape() {
    unsigned char digest[20];
    for (int i = 0; i < 20; ++i)
        digest[i] = (unsigned char)(i * 13 + 7);
    std::string r = encode(digest, 20);
    assert(r.size() == 28);
    assert(r.back() == '=');
    assert(r[26] != '='); // only ONE pad, not two
    std::cout << "PASSED: test_sha1_digest_shape\n";
}

// ─── Known WebSocket accept-key vector ────────────────────────────────────────
// From RFC 6455: SHA-1 of the client key + magic GUID produces these 20 bytes,
// which base64-encode to the known Sec-WebSocket-Accept value. This verifies
// base64 against the exact bytes the handshake produces.
void test_websocket_accept_vector() {
    // The 20-byte SHA-1 digest for the RFC 6455 example
    // (key "dGhlIHNhbXBsZSBub25jZQ==" + magic GUID).
    unsigned char digest[20] = {
        0xb3, 0x7a, 0x4f, 0x2c, 0xc0, 0x62, 0x4f, 0x16, 0x90, 0xf6,
        0x46, 0x06, 0xcf, 0x38, 0x59, 0x45, 0xb2, 0xbe, 0xc4, 0xea};
    std::string r = encode(digest, 20);
    assert(r == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    std::cout << "PASSED: test_websocket_accept_vector\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    test_rfc4648_vectors();
    test_no_padding();
    test_one_pad();
    test_two_pads();
    test_output_length_multiple_of_4();
    test_empty();
    test_binary_null_bytes();
    test_binary_high_bytes();
    test_binary_mixed();
    test_sha1_digest_shape();
    test_websocket_accept_vector();

    std::cout << "\nAll base64 tests passed\n";
    return 0;
}