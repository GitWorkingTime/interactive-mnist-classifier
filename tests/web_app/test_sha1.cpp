#include "sha-1.h"
#include <cassert>
#include <iostream>
#include <string>

// ─── Helper ───────────────────────────────────────────────────────────────────
// hash() returns the base64-encoded 20-byte SHA-1 digest. Compare it against a
// known-good base64 value (generated from a reference SHA-1 implementation and
// cross-checked with the system `sha1sum` tool).
void check(const char* input, const std::string& expected) {
    std::string result = hash(input);
    if (result != expected) {
        std::cout << "FAIL: \"" << input << "\"\n  got:      " << result
                  << "\n  expected: " << expected << "\n";
        assert(false);
    }
}

// ─── Known-answer tests ───────────────────────────────────────────────────────
// Each expected value is the base64 encoding of the SHA-1 digest of the input.

void test_empty_string() {
    // SHA-1("") hex = da39a3ee5e6b4b0d3255bfef95601890afd80709
    check("", "2jmj7l5rSw0yVb/vlWAYkK/YBwk=");
    std::cout << "PASSED: test_empty_string\n";
}

void test_abc() {
    // SHA-1("abc") hex = a9993e364706816aba3e25717850c26c9cd0d89d
    // The canonical FIPS 180-4 single-block example.
    check("abc", "qZk+NkcGgWq6PiVxeFDCbJzQ2J0=");
    std::cout << "PASSED: test_abc\n";
}

void test_fox() {
    // A longer single-message case with spaces and mixed content.
    check("The quick brown fox jumps over the lazy dog",
          "L9ThxnotKPzthJ7hu3bnORuT6xI=");
    std::cout << "PASSED: test_fox\n";
}

// ─── Multi-block cases ────────────────────────────────────────────────────────
// These inputs pad to more than 64 bytes, exercising the block loop and the
// carry of the hash state across blocks.

void test_multiblock_63_bytes() {
    // 63-byte input -> pads to 128 bytes (2 blocks).
    // SHA-1 hex = b85d6468bd3a73794bceaf812239cc1fe460ab95
    check("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno",
          "uF1kaL06c3lLzq+BIjnMH+Rgq5U=");
    std::cout << "PASSED: test_multiblock_63_bytes\n";
}

// ─── WebSocket handshake vector (RFC 6455, Section 1.3) ───────────────────────
// The exact computation the server performs: SHA-1 of the client key + GUID,
// base64-encoded, must equal the Sec-WebSocket-Accept value from the spec.
// This is a 60-byte input -> 2 blocks, and is the most important test since it
// is the real handshake case.
void test_websocket_accept() {
    check("dGhlIHNhbXBsZSBub25jZQ==258EAFA5-E914-47DA-95CA-C5AB0DC85B11",
          "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    std::cout << "PASSED: test_websocket_accept\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    test_empty_string();
    test_abc();
    test_fox();
    test_multiblock_63_bytes();
    test_websocket_accept();

    std::cout << "\nAll SHA-1 tests passed\n";
    return 0;
}