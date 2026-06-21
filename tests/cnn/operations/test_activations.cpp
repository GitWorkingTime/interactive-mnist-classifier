#include "activations.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
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

// ─── softMax tests (append these to your existing activations test file) ──────
void test_softmax_sums_to_one() {
    Tensor a({3, 1, 1}, {1, 2, 3});
    Tensor s = activations::softMax(a);
    float sum = 0.0f;
    for (float v : s.getData())
        sum += v;
    assert(floatEq(sum, 1.0f));
    std::cout << "PASSED: test_softmax_sums_to_one\n";
}

void test_softmax_values() {
    // softmax({1,2,3}) ≈ {0.090, 0.245, 0.665}
    Tensor a({3, 1, 1}, {1, 2, 3});
    Tensor s = activations::softMax(a);
    assert(floatEq(s.at({0, 0, 0}), 0.090f));
    assert(floatEq(s.at({1, 0, 0}), 0.245f));
    assert(floatEq(s.at({2, 0, 0}), 0.665f));
    std::cout << "PASSED: test_softmax_values\n";
}

void test_softmax_all_in_unit_range() {
    Tensor a({5, 1, 1}, {-3, 0, 1, 4, 10});
    Tensor s = activations::softMax(a);
    for (float v : s.getData()) {
        assert(v > 0.0f);
        assert(v < 1.0f);
    }
    std::cout << "PASSED: test_softmax_all_in_unit_range\n";
}

void test_softmax_preserves_order() {
    // The largest input must map to the largest probability
    Tensor a({4, 1, 1}, {2, 7, 1, 5});
    Tensor s = activations::softMax(a);
    const std::vector<float>& out = s.getData();
    for (std::size_t i = 0; i < out.size(); ++i)
        if (i != 1)
            assert(out[1] > out[i]); // index 1 (value 7) is largest
    std::cout << "PASSED: test_softmax_preserves_order\n";
}

void test_softmax_uniform_input() {
    // Equal logits → equal probabilities (1/n each)
    Tensor a({3, 1, 1}, {5, 5, 5});
    Tensor s = activations::softMax(a);
    for (float v : s.getData())
        assert(floatEq(v, 1.0f / 3.0f));
    std::cout << "PASSED: test_softmax_uniform_input\n";
}

void test_softmax_numerical_stability() {
    // Large inputs would overflow exp() without the max-subtraction trick.
    Tensor a({3, 1, 1}, {1000, 1001, 1002});
    Tensor s = activations::softMax(a);
    const std::vector<float>& out = s.getData();
    float sum = 0.0f;
    for (float v : out) {
        assert(std::isfinite(v)); // not nan or inf
        sum += v;
    }
    assert(floatEq(sum, 1.0f));
    // shift-invariance: same gaps as {1,2,3} → same distribution
    assert(floatEq(out[0], 0.090f));
    assert(floatEq(out[1], 0.245f));
    assert(floatEq(out[2], 0.665f));
    std::cout << "PASSED: test_softmax_numerical_stability\n";
}

void test_softmax_preserves_shape() {
    Tensor a({4, 1, 1}, {1, 2, 3, 4});
    Tensor s = activations::softMax(a);
    assert(s.getShape() == a.getShape());
    std::cout << "PASSED: test_softmax_preserves_shape\n";
}

void test_softmax_does_not_mutate_original() {
    Tensor a({3, 1, 1}, {1, 2, 3});
    Tensor s = activations::softMax(a);
    assert(floatEq(a.at({0, 0, 0}), 1.0f));
    assert(floatEq(a.at({1, 0, 0}), 2.0f));
    assert(floatEq(a.at({2, 0, 0}), 3.0f));
    std::cout << "PASSED: test_softmax_does_not_mutate_original\n";
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

    // Softmax
    test_softmax_sums_to_one();
    test_softmax_values();
    test_softmax_all_in_unit_range();
    test_softmax_preserves_order();
    test_softmax_uniform_input();
    test_softmax_numerical_stability();
    test_softmax_preserves_shape();
    test_softmax_does_not_mutate_original();

    std::cout << "\nAll activation tests passed\n";
    return 0;
}