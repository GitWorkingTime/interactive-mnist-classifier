#include "tensor.h"
#include <iostream>

int main() {
    std::cout << "Hello World" << std::endl;
    Tensor obj({2, 2, 1}, {1, 2, 3, 4});
    std::cout << obj.at({1, 1, 0}) << std::endl;

    return 0;
}