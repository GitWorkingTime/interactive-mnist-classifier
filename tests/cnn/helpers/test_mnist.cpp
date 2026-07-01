#include "mnist.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── reverseInt ───────────────────────────────────────────────────────────────
void test_reverse_int_known_swap() {
    // 0x12345678 byte-reversed is 0x78563412
    assert(mnist::reverseInt(0x12345678) == 0x78563412);
    std::cout << "PASSED: test_reverse_int_known_swap\n";
}

void test_reverse_int_single_byte() {
    // 0x00000001 (1) reversed → 0x01000000
    assert(mnist::reverseInt(0x00000001) == 0x01000000);
    std::cout << "PASSED: test_reverse_int_single_byte\n";
}

void test_reverse_int_mnist_image_magic() {
    // MNIST stores the image magic as big-endian bytes 00 00 08 03.
    // On a little-endian host that reads as 0x03080000; reversing gives 2051.
    assert(mnist::reverseInt(0x03080000) == 2051);
    std::cout << "PASSED: test_reverse_int_mnist_image_magic\n";
}

void test_reverse_int_mnist_label_magic() {
    // Label magic big-endian 00 00 08 01 → reads 0x01080000 → reverses to 2049.
    assert(mnist::reverseInt(0x01080000) == 2049);
    std::cout << "PASSED: test_reverse_int_mnist_label_magic\n";
}

void test_reverse_int_is_involution() {
    // Reversing twice must return the original value
    int values[] = {0, 1, 2051, 2049, 60000, 0x12345678, -1};
    for (int v : values)
        assert(mnist::reverseInt(mnist::reverseInt(v)) == v);
    std::cout << "PASSED: test_reverse_int_is_involution\n";
}

void test_reverse_int_zero() {
    assert(mnist::reverseInt(0) == 0);
    std::cout << "PASSED: test_reverse_int_zero\n";
}

// ─── oneHotEncodeLabels ───────────────────────────────────────────────────────
void test_onehot_shape() {
    Tensor t = mnist::oneHotEncodeLabels(3);
    assert(t.getShape()[0] == 1);
    assert(t.getShape()[1] == 10);
    assert(t.getShape()[2] == 1);
    std::cout << "PASSED: test_onehot_shape\n";
}

void test_onehot_correct_position() {
    // label 3 → 1.0 at index 3, 0.0 everywhere else
    Tensor t = mnist::oneHotEncodeLabels(3);
    const std::vector<float>& d = t.getData();
    for (int i = 0; i < 10; ++i) {
        if (i == 3)
            assert(floatEq(d[i], 1.0f));
        else
            assert(floatEq(d[i], 0.0f));
    }
    std::cout << "PASSED: test_onehot_correct_position\n";
}

void test_onehot_first_label() {
    // label 0 → 1.0 at index 0
    Tensor t = mnist::oneHotEncodeLabels(0);
    assert(floatEq(t.at({0, 0, 0}), 1.0f));
    assert(floatEq(t.at({0, 1, 0}), 0.0f));
    std::cout << "PASSED: test_onehot_first_label\n";
}

void test_onehot_last_label() {
    // label 9 → 1.0 at index 9
    Tensor t = mnist::oneHotEncodeLabels(9);
    assert(floatEq(t.at({0, 9, 0}), 1.0f));
    assert(floatEq(t.at({0, 8, 0}), 0.0f));
    std::cout << "PASSED: test_onehot_last_label\n";
}

void test_onehot_sums_to_one() {
    // Exactly one element is 1.0, rest 0.0 → sum is 1.0 (valid target distribution)
    for (int label = 0; label <= 9; ++label) {
        Tensor t = mnist::oneHotEncodeLabels(label);
        float sum = 0.0f;
        for (float v : t.getData())
            sum += v;
        assert(floatEq(sum, 1.0f));
    }
    std::cout << "PASSED: test_onehot_sums_to_one\n";
}

void test_onehot_all_labels_distinct_position() {
    // Each label puts its 1.0 at the matching index
    for (int label = 0; label <= 9; ++label) {
        Tensor t = mnist::oneHotEncodeLabels(label);
        assert(floatEq(t.at({0, label, 0}), 1.0f));
    }
    std::cout << "PASSED: test_onehot_all_labels_distinct_position\n";
}

void test_onehot_throws_below_range() {
    try {
        mnist::oneHotEncodeLabels(-1);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_onehot_throws_below_range\n";
}

void test_onehot_throws_above_range() {
    try {
        mnist::oneHotEncodeLabels(10);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_onehot_throws_above_range\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // reverseInt
    test_reverse_int_known_swap();
    test_reverse_int_single_byte();
    test_reverse_int_mnist_image_magic();
    test_reverse_int_mnist_label_magic();
    test_reverse_int_is_involution();
    test_reverse_int_zero();

    // oneHotEncodeLabels
    test_onehot_shape();
    test_onehot_correct_position();
    test_onehot_first_label();
    test_onehot_last_label();
    test_onehot_sums_to_one();
    test_onehot_all_labels_distinct_position();
    test_onehot_throws_below_range();
    test_onehot_throws_above_range();

    std::cout << "\nAll MNIST helper tests passed\n";
    return 0;
}