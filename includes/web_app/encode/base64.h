// ─── Imports ─────────────────────────────────────────────────────────────────
#include <cstdio>
#include <iostream>
#include <string>

// ─── Function declaration ────────────────────────────────────────────────────
/**
 * @brief Encodes binary data as a standard base64 string.
 *
 * @par Description
 * Converts arbitrary binary input into base64 text using the standard alphabet
 * (A-Z, a-z, 0-9, +, /) with '=' padding. Processes the input in 3-byte groups,
 * emitting 4 characters per group; when the input length is not a multiple of 3,
 * the final group is padded ('=' for 2 leftover bytes, '==' for 1) so the output
 * length is always a multiple of 4. Uses the standard variant (not base64url), so
 * the output is suitable for the WebSocket Sec-WebSocket-Accept header.
 *
 * @param data   Pointer to the binary bytes to encode. Treated as unsigned so
 *               byte values 128-255 and null bytes are handled correctly.
 * @param length The number of bytes to encode. Required because binary data has
 *               no terminator (null bytes are valid data, not an end marker).
 *
 * @return The base64-encoded string. Empty input yields an empty string.
 *
 * @par Example
 * @code
 * unsigned char hash[20] = { ... };          // e.g. a SHA-1 digest
 * std::string encoded = encode(hash, 20);    // 28 chars, ends in '='
 * @endcode
 */
std::string encode(unsigned char* data, size_t length);