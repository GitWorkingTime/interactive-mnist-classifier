#include "tensor.h"
#include "weights_init.h"
#include <cassert>
#include <cmath>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// Sample mean of a tensor's data
float mean(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float x : v)
        sum += x;
    return sum / v.size();
}

// Sample standard deviation of a tensor's data
float stddev(const std::vector<float>& v) {
    float m = mean(v);
    float sumSq = 0.0f;
    for (float x : v)
        sumSq += (x - m) * (x - m);
    return std::sqrt(sumSq / v.size());
}

// ─── Structural guarantees ────────────────────────────────────────────────────
void test_he_shape() {
    Tensor w = weights::heInit({3, 3, 1}, 9);
    assert(w.getShape()[0] == 3);
    assert(w.getShape()[1] == 3);
    assert(w.getShape()[2] == 1);
    std::cout << "PASSED: test_he_shape\n";
}

void test_he_size() {
    Tensor w = weights::heInit({4, 4, 2}, 32);
    assert(w.getSize() == 32); // 4*4*2
    std::cout << "PASSED: test_he_size\n";
}

void test_he_shape_depth() {
    Tensor w = weights::heInit({5, 5, 3}, 75);
    assert(w.getShape()[2] == 3);
    assert(w.getSize() == 75);
    std::cout << "PASSED: test_he_shape_depth\n";
}

// ─── Statistical properties ───────────────────────────────────────────────────
// With a large sample, the data should look like N(0, sqrt(2/fanIn)).
// Tolerances are loose because these are finite random samples.

void test_he_mean_near_zero() {
    // Large tensor so the sample mean is a reliable estimate
    int fanIn = 100;
    Tensor w = weights::heInit({100, 100, 1}, fanIn); // 10,000 values
    float m = mean(w.getData());
    // Sample mean of N(0, 0.141) over 10k samples should be very close to 0
    assert(std::abs(m) < 0.02f);
    std::cout << "PASSED: test_he_mean_near_zero (mean=" << m << ")\n";
}

void test_he_stddev_matches_formula() {
    // He stddev = sqrt(2 / fanIn). For fanIn=100 → 0.1414
    int fanIn = 100;
    Tensor w = weights::heInit({100, 100, 1}, fanIn); // 10,000 values
    float s = stddev(w.getData());
    float expected = std::sqrt(2.0f / fanIn); // ≈ 0.1414
    // Within 10% — finite-sample variation
    assert(std::abs(s - expected) < expected * 0.1f);
    std::cout << "PASSED: test_he_stddev_matches_formula (stddev=" << s
              << ", expected=" << expected << ")\n";
}

void test_he_stddev_scales_with_fanin() {
    // Smaller fanIn → larger spread. fanIn=9 → 0.4714, fanIn=144 → 0.1179
    Tensor wSmall = weights::heInit({100, 100, 1}, 9);
    Tensor wLarge = weights::heInit({100, 100, 1}, 144);
    float sSmall = stddev(wSmall.getData());
    float sLarge = stddev(wLarge.getData());
    // Smaller fanIn must produce a wider distribution
    assert(sSmall > sLarge);
    // And each should be near its formula value
    assert(std::abs(sSmall - std::sqrt(2.0f / 9)) < std::sqrt(2.0f / 9) * 0.1f);
    assert(std::abs(sLarge - std::sqrt(2.0f / 144)) < std::sqrt(2.0f / 144) * 0.1f);
    std::cout << "PASSED: test_he_stddev_scales_with_fanin\n";
}

// ─── Randomness guarantees ────────────────────────────────────────────────────
void test_he_values_differ() {
    // Successive calls must produce different values (persistent generator).
    // If the generator restarted each call, these two filters would be identical
    // and all conv filters would learn the same features.
    Tensor a = weights::heInit({3, 3, 1}, 9);
    Tensor b = weights::heInit({3, 3, 1}, 9);
    assert(!(a == b)); // extremely unlikely to be equal unless generator restarts
    std::cout << "PASSED: test_he_values_differ\n";
}

void test_he_not_all_same_value() {
    // Within a single tensor, values should vary (not all identical)
    Tensor w = weights::heInit({10, 10, 1}, 100);
    const std::vector<float>& data = w.getData();
    bool foundDifferent = false;
    for (std::size_t i = 1; i < data.size(); ++i) {
        if (!floatEq(data[i], data[0])) {
            foundDifferent = true;
            break;
        }
    }
    assert(foundDifferent);
    std::cout << "PASSED: test_he_not_all_same_value\n";
}

void test_he_not_all_zero() {
    // He init must NOT produce zeros (that would defeat symmetry-breaking)
    Tensor w = weights::heInit({10, 10, 1}, 100);
    bool foundNonZero = false;
    for (float v : w.getData()) {
        if (!floatEq(v, 0.0f)) {
            foundNonZero = true;
            break;
        }
    }
    assert(foundNonZero);
    std::cout << "PASSED: test_he_not_all_zero\n";
}

void test_he_has_both_signs() {
    // A mean-zero normal distribution must produce both positive and negative values
    Tensor w = weights::heInit({100, 100, 1}, 100);
    bool foundPos = false, foundNeg = false;
    for (float v : w.getData()) {
        if (v > 0)
            foundPos = true;
        if (v < 0)
            foundNeg = true;
        if (foundPos && foundNeg)
            break;
    }
    assert(foundPos && foundNeg);
    std::cout << "PASSED: test_he_has_both_signs\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // structural
    test_he_shape();
    test_he_size();
    test_he_shape_depth();

    // statistical
    test_he_mean_near_zero();
    test_he_stddev_matches_formula();
    test_he_stddev_scales_with_fanin();

    // randomness
    test_he_values_differ();
    test_he_not_all_same_value();
    test_he_not_all_zero();
    test_he_has_both_signs();

    std::cout << "\nAll He init tests passed\n";
    return 0;
}