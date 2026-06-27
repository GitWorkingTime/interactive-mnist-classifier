// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base64.h"

// Length is in bytes
std::string encode(unsigned char* data, size_t length) {
    const char* chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string result;
    for (size_t i = 0; i + 3 <= length; i += 3) {
        unsigned int batch = data[i] << 16 | data[i + 1] << 8 | data[i + 2];
        result += chars[(batch >> 18) & 0x3F];
        result += chars[(batch >> 12) & 0x3F];
        result += chars[(batch >> 6) & 0x3F];
        result += chars[batch & 0x3F];
    }

    int remainingBytes = length % 3;
    if (remainingBytes == 1) {
        unsigned int batch = data[length - remainingBytes] << 16;
        result += chars[(batch >> 18) & 0x3F];
        result += chars[(batch >> 12) & 0x3F];
        result += "==";
    } else if (remainingBytes == 2) {
        unsigned int batch = data[length - remainingBytes] << 16 | data[length - remainingBytes + 1] << 8;
        result += chars[(batch >> 18) & 0x3F];
        result += chars[(batch >> 12) & 0x3F];
        result += chars[(batch >> 6) & 0x3F];
        result += '=';
    }

    return result;
}