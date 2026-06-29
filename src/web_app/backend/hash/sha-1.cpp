#include <cstdint>
#include <stdexcept>
#include <stdio.h>

uint32_t leftRotate(const uint32_t& value, size_t pos) {
    if (pos == 0) {
        throw std::invalid_argument("ERROR: Pos received is 0");
    }

    return (value << pos) | (value >> (32 - pos));
}

int main() {
    uint32_t test = 0x80000000;
    printf("%X\n", leftRotate(test, 1));

    return 0;
}