// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base64.h"
#include <cstdint>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <vector>

// ─── Function declarations ───────────────────────────────────────────────────
/**
 * @brief Circularly rotates a 32-bit value left by a given number of positions.
 *
 * @par Description
 * Performs a left rotation (not a shift): bits shifted off the high end wrap
 * around to the low end. Implemented as (value << pos) | (value >> (32 - pos)).
 * Used throughout SHA-1's compression rounds and message schedule.
 *
 * @param value The 32-bit value to rotate.
 * @param pos   The number of positions to rotate left. Must be between 1 and 31;
 *              0 is rejected because value >> 32 is undefined behavior in C++.
 *
 * @return The rotated 32-bit value.
 *
 * @throws std::invalid_argument if pos is 0.
 */
uint32_t leftRotate(uint32_t value, size_t pos);

/**
 * @brief Applies SHA-1 message padding to a byte sequence.
 *
 * @par Description
 * Pads the message per the SHA-1 specification so its length becomes a multiple
 * of 64 bytes (512 bits): appends a single 0x80 byte, then 0x00 bytes until the
 * length is congruent to 56 mod 64, then the original message length in bits as
 * a 64-bit big-endian value. The result is always a whole number of 64-byte
 * blocks, ready for block processing.
 *
 * @param message Pointer to the message bytes to pad.
 * @param len     The number of message bytes. Required because binary data has
 *                no terminator; this count is the source of truth for how many
 *                bytes are read from message.
 *
 * @return The padded message as a byte vector whose size is a multiple of 64.
 */
std::vector<unsigned char> padding(const unsigned char* message, size_t len);

/**
 * @brief Computes the SHA-1 digest of a string and returns it base64-encoded.
 *
 * @par Description
 * Runs the full SHA-1 algorithm on the input text — padding, splitting into
 * 64-byte blocks, the message schedule, and the 80 compression rounds per block —
 * then base64-encodes the resulting 20-byte digest. This produces the value used
 * for the WebSocket handshake's Sec-WebSocket-Accept header when text is the
 * client key concatenated with the WebSocket GUID.
 *
 * @param text The null-terminated input string to hash. Its length is measured
 *             with strlen, so it must not contain embedded null bytes.
 *
 * @return The base64-encoded 20-byte SHA-1 digest (28 characters ending in '=').
 */
std::string hash(const char* text);