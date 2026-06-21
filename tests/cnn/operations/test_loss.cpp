#include "loss.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── crossEntropy: value correctness ──────────────────────────────────────────
void test_ce_basic_value() {
    // -log(0.7) ≈ 0.357
    Tensor pred({3, 1, 1}, {0.1f, 0.2f, 0.7f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    assert(floatEq(loss::crossEntropy(pred, tgt), 0.357f));
    std::cout << "PASSED: test_ce_basic_value\n";
}

void test_ce_confident_correct_is_low() {
    // Correct class gets 0.998 → -log(0.998) ≈ 0.002, very small loss
    Tensor pred({3, 1, 1}, {0.001f, 0.001f, 0.998f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    float l = loss::crossEntropy(pred, tgt);
    assert(floatEq(l, 0.002f));
    assert(l >= 0.0f);
    std::cout << "PASSED: test_ce_confident_correct_is_low\n";
}

void test_ce_confident_wrong_is_high() {
    // Correct class (index 2) gets only 0.001 → -log(0.001) ≈ 6.908, large loss
    Tensor pred({3, 1, 1}, {0.998f, 0.001f, 0.001f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    float l = loss::crossEntropy(pred, tgt);
    assert(l > 6.0f); // confidently wrong → heavily penalized
    std::cout << "PASSED: test_ce_confident_wrong_is_high\n";
}

void test_ce_uniform_prediction() {
    // Uniform 1/3 each → -log(1/3) = log(3) ≈ 1.0986
    Tensor pred({3, 1, 1}, {1.0f / 3, 1.0f / 3, 1.0f / 3});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    assert(floatEq(loss::crossEntropy(pred, tgt), 1.0986f));
    std::cout << "PASSED: test_ce_uniform_prediction\n";
}

void test_ce_perfect_prediction_near_zero() {
    // Prediction exactly matches target → loss ≈ 0 (epsilon makes it tiny positive)
    Tensor pred({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    float l = loss::crossEntropy(pred, tgt);
    assert(floatEq(l, 0.0f));
    std::cout << "PASSED: test_ce_perfect_prediction_near_zero\n";
}

void test_ce_zero_prediction_no_nan() {
    // Correct class predicted as exactly 0 — without the epsilon this is -log(0) = inf.
    // The 1e-7 guard must keep it finite: -log(1e-7) ≈ 16.1
    Tensor pred({3, 1, 1}, {0.5f, 0.5f, 0.0f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    float l = loss::crossEntropy(pred, tgt);
    assert(std::isfinite(l)); // must NOT be inf or nan
    std::cout << "PASSED: test_ce_zero_prediction_no_nan\n";
}

void test_ce_10_class() {
    // MNIST-like: 10 classes, correct class (3) gets 0.55
    std::vector<float> p(10, 0.05f);
    p[3] = 0.55f;
    std::vector<float> t(10, 0.0f);
    t[3] = 1.0f;
    Tensor pred({10, 1, 1}, p);
    Tensor tgt({10, 1, 1}, t);
    assert(floatEq(loss::crossEntropy(pred, tgt), 0.598f));
    std::cout << "PASSED: test_ce_10_class\n";
}

void test_ce_shape_mismatch_throws() {
    // exercises the validation branch in crossEntropy
    Tensor pred({3, 1, 1}, {0.1f, 0.2f, 0.7f});
    Tensor tgt({4, 1, 1}, {0.0f, 0.0f, 1.0f, 0.0f});
    try {
        loss::crossEntropy(pred, tgt);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_ce_shape_mismatch_throws\n";
}

// ─── crossEntropyGradient: value correctness ──────────────────────────────────
void test_grad_basic_value() {
    // prediction - target = {0.1, 0.2, 0.7} - {0, 0, 1} = {0.1, 0.2, -0.3}
    Tensor pred({3, 1, 1}, {0.1f, 0.2f, 0.7f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    Tensor g = loss::crossEntropyGradient(pred, tgt);
    assert(floatEq(g.at({0, 0, 0}), 0.1f));
    assert(floatEq(g.at({1, 0, 0}), 0.2f));
    assert(floatEq(g.at({2, 0, 0}), -0.3f));
    std::cout << "PASSED: test_grad_basic_value\n";
}

void test_grad_perfect_prediction_is_zero() {
    // Perfect prediction → gradient all zeros (no learning signal)
    Tensor pred({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    Tensor g = loss::crossEntropyGradient(pred, tgt);
    for (float v : g.getData())
        assert(floatEq(v, 0.0f));
    std::cout << "PASSED: test_grad_perfect_prediction_is_zero\n";
}

void test_grad_signs() {
    // Wrong classes (over-assigned) → positive; correct class (under) → negative
    Tensor pred({3, 1, 1}, {0.3f, 0.3f, 0.4f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    Tensor g = loss::crossEntropyGradient(pred, tgt);
    assert(g.at({0, 0, 0}) > 0.0f); // wrong class, push down
    assert(g.at({1, 0, 0}) > 0.0f); // wrong class, push down
    assert(g.at({2, 0, 0}) < 0.0f); // correct class below 1, push up
    std::cout << "PASSED: test_grad_signs\n";
}

void test_grad_preserves_shape() {
    Tensor pred({10, 1, 1}, std::vector<float>(10, 0.1f));
    Tensor tgt({10, 1, 1}, std::vector<float>(10, 0.1f));
    Tensor g = loss::crossEntropyGradient(pred, tgt);
    assert(g.getShape() == pred.getShape());
    std::cout << "PASSED: test_grad_preserves_shape\n";
}

void test_grad_sums_to_zero_when_both_distributions() {
    // If pred and target both sum to 1, (pred - target) sums to 0
    Tensor pred({3, 1, 1}, {0.2f, 0.3f, 0.5f});
    Tensor tgt({3, 1, 1}, {0.0f, 0.0f, 1.0f});
    Tensor g = loss::crossEntropyGradient(pred, tgt);
    float sum = 0.0f;
    for (float v : g.getData())
        sum += v;
    assert(floatEq(sum, 0.0f));
    std::cout << "PASSED: test_grad_sums_to_zero_when_both_distributions\n";
}

void test_grad_shape_mismatch_throws() {
    // exercises the validation branch in crossEntropyGradient
    Tensor pred({3, 1, 1}, {0.1f, 0.2f, 0.7f});
    Tensor tgt({2, 1, 1}, {0.0f, 1.0f});
    try {
        loss::crossEntropyGradient(pred, tgt);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_grad_shape_mismatch_throws\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // crossEntropy
    test_ce_basic_value();
    test_ce_confident_correct_is_low();
    test_ce_confident_wrong_is_high();
    test_ce_uniform_prediction();
    test_ce_perfect_prediction_near_zero();
    test_ce_zero_prediction_no_nan();
    test_ce_10_class();
    test_ce_shape_mismatch_throws();

    // crossEntropyGradient
    test_grad_basic_value();
    test_grad_perfect_prediction_is_zero();
    test_grad_signs();
    test_grad_preserves_shape();
    test_grad_sums_to_zero_when_both_distributions();
    test_grad_shape_mismatch_throws();

    std::cout << "\nAll loss tests passed\n";
    return 0;
}