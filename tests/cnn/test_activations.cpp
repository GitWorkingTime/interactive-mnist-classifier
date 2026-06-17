#include "activations.h"
#include "tensor.h"
#include <cassert>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── ReLU ─────────────────────────────────────────────────────────────────────
void test_relu_negatives_become_zero() {
    Tensor a({2, 2, 1}, {-1, 2, 3, -4});
    Tensor b = activations::ReLU(a);
    assert(floatEq(b.at({0, 0, 0}), 0.0f)); // -1 → 0
    assert(floatEq(b.at({1, 1, 0}), 0.0f)); // -4 → 0
    std::cout << "PASSED: test_relu_negatives_become_zero\n";
}

void test_relu_positives_unchanged() {
    Tensor a({2, 2, 1}, {-1, 2, 3, -4});
    Tensor b = activations::ReLU(a);
    assert(floatEq(b.at({1, 0, 0}), 2.0f)); // 2 → 2
    assert(floatEq(b.at({0, 1, 0}), 3.0f)); // 3 → 3
    std::cout << "PASSED: test_relu_positives_unchanged\n";
}

void test_relu_zero_stays_zero() {
    Tensor a({1, 1, 1}, {0});
    Tensor b = activations::ReLU(a);
    assert(floatEq(b.at({0, 0, 0}), 0.0f)); // 0 → 0
    std::cout << "PASSED: test_relu_zero_stays_zero\n";
}

void test_relu_all_negative() {
    Tensor a({3, 1, 1}, {-5, -0.1f, -100});
    Tensor b = activations::ReLU(a);
    assert(floatEq(b.at({0, 0, 0}), 0.0f));
    assert(floatEq(b.at({1, 0, 0}), 0.0f));
    assert(floatEq(b.at({2, 0, 0}), 0.0f));
    std::cout << "PASSED: test_relu_all_negative\n";
}

void test_relu_all_positive() {
    Tensor a({3, 1, 1}, {5, 0.1f, 100});
    Tensor b = activations::ReLU(a);
    assert(floatEq(b.at({0, 0, 0}), 5.0f));
    assert(floatEq(b.at({1, 0, 0}), 0.1f));
    assert(floatEq(b.at({2, 0, 0}), 100.0f));
    std::cout << "PASSED: test_relu_all_positive\n";
}

void test_relu_preserves_shape() {
    Tensor a({3, 4, 2}, std::vector<float>(24, -1.0f));
    Tensor b = activations::ReLU(a);
    assert(b.getShape() == a.getShape());
    std::cout << "PASSED: test_relu_preserves_shape\n";
}

void test_relu_depth_greater_than_one() {
    // Each slice has its own values; ReLU is element-wise across all of them
    Tensor a({2, 1, 2}, {-1, 5, 3, -2});
    Tensor b = activations::ReLU(a);
    assert(floatEq(b.at({0, 0, 0}), 0.0f)); // slice 0: -1 → 0
    assert(floatEq(b.at({1, 0, 0}), 5.0f)); // slice 0:  5 → 5
    assert(floatEq(b.at({0, 0, 1}), 3.0f)); // slice 1:  3 → 3
    assert(floatEq(b.at({1, 0, 1}), 0.0f)); // slice 1: -2 → 0
    std::cout << "PASSED: test_relu_depth_greater_than_one\n";
}

void test_relu_does_not_mutate_original() {
    Tensor a({2, 1, 1}, {-1, 2});
    Tensor b = activations::ReLU(a);
    // original must be untouched
    assert(floatEq(a.at({0, 0, 0}), -1.0f));
    assert(floatEq(a.at({1, 0, 0}), 2.0f));
    std::cout << "PASSED: test_relu_does_not_mutate_original\n";
}

// ─── ReLUDerivative ───────────────────────────────────────────────────────────
void test_relu_deriv_negatives_are_zero() {
    Tensor a({2, 2, 1}, {-1, 2, 3, -4});
    Tensor d = activations::ReLUDerivative(a);
    assert(floatEq(d.at({0, 0, 0}), 0.0f)); // -1 → 0
    assert(floatEq(d.at({1, 1, 0}), 0.0f)); // -4 → 0
    std::cout << "PASSED: test_relu_deriv_negatives_are_zero\n";
}

void test_relu_deriv_positives_are_one() {
    Tensor a({2, 2, 1}, {-1, 2, 3, -4});
    Tensor d = activations::ReLUDerivative(a);
    assert(floatEq(d.at({1, 0, 0}), 1.0f)); // 2 → 1
    assert(floatEq(d.at({0, 1, 0}), 1.0f)); // 3 → 1
    std::cout << "PASSED: test_relu_deriv_positives_are_one\n";
}

void test_relu_deriv_zero_is_zero() {
    // Convention: derivative at exactly 0 is treated as 0 (the "off" side)
    Tensor a({1, 1, 1}, {0});
    Tensor d = activations::ReLUDerivative(a);
    assert(floatEq(d.at({0, 0, 0}), 0.0f));
    std::cout << "PASSED: test_relu_deriv_zero_is_zero\n";
}

void test_relu_deriv_is_binary_mask() {
    // Every output must be exactly 0 or 1, nothing else
    Tensor a({4, 1, 1}, {-3, 0, 0.5f, 7});
    Tensor d = activations::ReLUDerivative(a);
    const std::vector<float>& out = d.getData();
    for (float v : out)
        assert(floatEq(v, 0.0f) || floatEq(v, 1.0f));
    // and specifically:
    assert(floatEq(d.at({0, 0, 0}), 0.0f)); // -3 → 0
    assert(floatEq(d.at({1, 0, 0}), 0.0f)); //  0 → 0
    assert(floatEq(d.at({2, 0, 0}), 1.0f)); // 0.5 → 1
    assert(floatEq(d.at({3, 0, 0}), 1.0f)); //  7 → 1
    std::cout << "PASSED: test_relu_deriv_is_binary_mask\n";
}

void test_relu_deriv_preserves_shape() {
    Tensor a({3, 4, 2}, std::vector<float>(24, 1.0f));
    Tensor d = activations::ReLUDerivative(a);
    assert(d.getShape() == a.getShape());
    std::cout << "PASSED: test_relu_deriv_preserves_shape\n";
}

void test_relu_deriv_does_not_mutate_original() {
    Tensor a({2, 1, 1}, {-1, 2});
    Tensor d = activations::ReLUDerivative(a);
    assert(floatEq(a.at({0, 0, 0}), -1.0f));
    assert(floatEq(a.at({1, 0, 0}), 2.0f));
    std::cout << "PASSED: test_relu_deriv_does_not_mutate_original\n";
}

// ─── Relationship between ReLU and its derivative ─────────────────────────────
void test_relu_and_deriv_agree_on_sign() {
    // Where ReLU passes a value through (>0), derivative is 1.
    // Where ReLU zeros a value (<=0), derivative is 0.
    Tensor a({5, 1, 1}, {-2, -0.001f, 0, 0.001f, 9});
    Tensor b = activations::ReLU(a);
    Tensor d = activations::ReLUDerivative(a);
    const std::vector<float>& bd = b.getData();
    const std::vector<float>& dd = d.getData();
    for (std::size_t i = 0; i < bd.size(); ++i) {
        if (dd[i] > 0.5f)                 // derivative says "active"
            assert(bd[i] > 0.0f);         // then ReLU passed a positive value
        else                              // derivative says "off"
            assert(floatEq(bd[i], 0.0f)); // then ReLU output is 0
    }
    std::cout << "PASSED: test_relu_and_deriv_agree_on_sign\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // ReLU
    test_relu_negatives_become_zero();
    test_relu_positives_unchanged();
    test_relu_zero_stays_zero();
    test_relu_all_negative();
    test_relu_all_positive();
    test_relu_preserves_shape();
    test_relu_depth_greater_than_one();
    test_relu_does_not_mutate_original();

    // ReLUDerivative
    test_relu_deriv_negatives_are_zero();
    test_relu_deriv_positives_are_one();
    test_relu_deriv_zero_is_zero();
    test_relu_deriv_is_binary_mask();
    test_relu_deriv_preserves_shape();
    test_relu_deriv_does_not_mutate_original();

    // Relationship
    test_relu_and_deriv_agree_on_sign();

    std::cout << "\nAll activation tests passed\n";
    return 0;
}