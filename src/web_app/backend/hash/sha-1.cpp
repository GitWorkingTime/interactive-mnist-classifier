#include "base64.h"
#include <cstdint>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <vector>

uint32_t leftRotate(uint32_t value, size_t pos) {
    if (pos == 0) {
        throw std::invalid_argument("ERROR: Pos received is 0");
    }

    return (value << pos) | (value >> (32 - pos));
}

std::vector<unsigned char> padding(const unsigned char* message, size_t len) {
    std::vector<unsigned char> buf;

    for (size_t i = 0; i < len; ++i) {
        buf.push_back(message[i]);
    }

    buf.push_back(0x80);
    while (buf.size() % 64 != 56) {
        buf.push_back(0x00);
    }

    uint64_t bitLen = (uint64_t)len * 8;

    buf.push_back((bitLen >> 56) & 0xFF);
    buf.push_back((bitLen >> 48) & 0xFF);
    buf.push_back((bitLen >> 40) & 0xFF);
    buf.push_back((bitLen >> 32) & 0xFF);
    buf.push_back((bitLen >> 24) & 0xFF);
    buf.push_back((bitLen >> 16) & 0xFF);
    buf.push_back((bitLen >> 8) & 0xFF);
    buf.push_back(bitLen & 0xFF);

    return buf;
}

std::string hash(const char* text) {
    std::vector<unsigned char> padded = padding((unsigned char*)text, strlen(text));

    // Hash constants:
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;

    int iterations = padded.size() / 64;
    for (int i = 0; i < iterations; ++i) {
        uint32_t w[80];

        // Pack this block's 16 words — note the i*64 block offset
        for (size_t j = 0; j < 16; ++j) {
            w[j] = ((uint32_t)padded[i * 64 + j * 4] << 24) | ((uint32_t)padded[i * 64 + j * 4 + 1] << 16) | ((uint32_t)padded[i * 64 + j * 4 + 2] << 8) | (uint32_t)padded[i * 64 + j * 4 + 3];
        }

        // Extend from 16 words to 80
        for (size_t j = 16; j < 80; ++j) {
            w[j] = leftRotate((w[j - 3] ^ w[j - 8] ^ w[j - 14] ^ w[j - 16]), 1);
        }

        // Compression:
        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        uint32_t f, k;

        for (size_t j = 0; j < 80; ++j) {
            if (j < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            } else if (j < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;

            } else if (j < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;

            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            uint32_t temp = leftRotate(a, 5) + f + e + k + w[j];
            e = d;
            d = c;
            c = leftRotate(b, 30);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    unsigned char digest[20];
    uint32_t hs[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        digest[i * 4] = (hs[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (hs[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (hs[i] >> 8) & 0xFF;
        digest[i * 4 + 3] = hs[i] & 0xFF;
    }

    std::string hashed = encode(digest, 20);
    return hashed;
}