#include "pooling.h"
#include "tensor.h"
#include <cassert>
#include <iostream>
#include <numeric>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── Basic correctness ────────────────────────────────────────────────────────
void test_pool_basic_values() {
    // 4x4 input 1..16, 2x2 window, stride 2 → 2x2 output
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    // windows: [1,2,5,6]→6  [3,4,7,8]→8  [9,10,13,14]→14  [11,12,15,16]→16
    assert(floatEq(r.pooled.at({0, 0, 0}), 6.0f));
    assert(floatEq(r.pooled.at({1, 0, 0}), 8.0f));
    assert(floatEq(r.pooled.at({0, 1, 0}), 14.0f));
    assert(floatEq(r.pooled.at({1, 1, 0}), 16.0f));
    std::cout << "PASSED: test_pool_basic_values\n";
}

void test_pool_basic_shape() {
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    assert(r.pooled.getShape()[0] == 2);
    assert(r.pooled.getShape()[1] == 2);
    assert(r.pooled.getShape()[2] == 1);
    std::cout << "PASSED: test_pool_basic_shape\n";
}

// ─── argmax correctness ───────────────────────────────────────────────────────
void test_pool_argmax_values() {
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    // flat input indices of the winners: 6@5, 8@7, 14@13, 16@15
    assert(r.argmax.size() == 4);
    assert(r.argmax[0] == 5);
    assert(r.argmax[1] == 7);
    assert(r.argmax[2] == 13);
    assert(r.argmax[3] == 15);
    std::cout << "PASSED: test_pool_argmax_values\n";
}

void test_pool_argmax_size_matches_pooled() {
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    assert(static_cast<int>(r.argmax.size()) == r.pooled.getSize());
    std::cout << "PASSED: test_pool_argmax_size_matches_pooled\n";
}

// ─── Depth independence (per-channel pooling) ─────────────────────────────────
void test_pool_depth_independent() {
    // slice 0: 1..16, slice 1: 17..32 — each pooled separately
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 2}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    assert(r.pooled.getShape()[2] == 2); // depth unchanged
    // slice 0 maxes
    assert(floatEq(r.pooled.at({0, 0, 0}), 6.0f));
    assert(floatEq(r.pooled.at({1, 1, 0}), 16.0f));
    // slice 1 maxes (17..32)
    assert(floatEq(r.pooled.at({0, 0, 1}), 22.0f));
    assert(floatEq(r.pooled.at({1, 1, 1}), 32.0f));
    std::cout << "PASSED: test_pool_depth_independent\n";
}

void test_pool_depth_argmax_indices() {
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 2}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    // slice 1 winners are in the second half of the flat array: 21,23,29,31
    assert(r.argmax[4] == 21);
    assert(r.argmax[5] == 23);
    assert(r.argmax[6] == 29);
    assert(r.argmax[7] == 31);
    std::cout << "PASSED: test_pool_depth_argmax_indices\n";
}

// ─── Larger window / stride ───────────────────────────────────────────────────
void test_pool_3x3_stride3() {
    // 6x6 input 1..36, 3x3 window, stride 3 → 2x2 output
    std::vector<float> data(36);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({6, 6, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 3, 3);

    assert(r.pooled.getShape()[0] == 2);
    assert(r.pooled.getShape()[1] == 2);
    assert(floatEq(r.pooled.at({0, 0, 0}), 15.0f));
    assert(floatEq(r.pooled.at({1, 0, 0}), 18.0f));
    assert(floatEq(r.pooled.at({0, 1, 0}), 33.0f));
    assert(floatEq(r.pooled.at({1, 1, 0}), 36.0f));
    std::cout << "PASSED: test_pool_3x3_stride3\n";
}

void test_pool_overlapping_stride1() {
    // 5x5, 2x2 window, stride 1 → (5-2)/1+1 = 4 → 4x4 output
    std::vector<float> data(25);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({5, 5, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 1);

    assert(r.pooled.getShape()[0] == 4);
    assert(r.pooled.getShape()[1] == 4);
    std::cout << "PASSED: test_pool_overlapping_stride1\n";
}

// ─── Negative values (verifies -infinity initialization) ──────────────────────
void test_pool_all_negative() {
    // If maxVal were initialized to 0 instead of -inf, this would wrongly
    // report 0 as the max. All values are negative, so the max is the
    // least-negative element in each window.
    std::vector<float> data(16);
    for (int i = 0; i < 16; ++i)
        data[i] = -static_cast<float>(i + 1); // -1..-16
    Tensor a({4, 4, 1}, data);

    pooling::PoolResult r = pooling::maxPooling(a, 2, 2);

    // window [-1,-2,-5,-6] → max -1, etc.
    assert(floatEq(r.pooled.at({0, 0, 0}), -1.0f));
    assert(floatEq(r.pooled.at({1, 0, 0}), -3.0f));
    assert(floatEq(r.pooled.at({0, 1, 0}), -9.0f));
    assert(floatEq(r.pooled.at({1, 1, 0}), -11.0f));
    // argmax should point at those least-negative positions
    assert(r.argmax[0] == 0);
    assert(r.argmax[1] == 2);
    assert(r.argmax[2] == 8);
    assert(r.argmax[3] == 10);
    std::cout << "PASSED: test_pool_all_negative\n";
}

// ─── Does not mutate input ────────────────────────────────────────────────────
void test_pool_does_not_mutate_original() {
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 1}, data);

    pooling::maxPooling(a, 2, 2);

    // original tensor must be untouched
    assert(floatEq(a.at({0, 0, 0}), 1.0f));
    assert(floatEq(a.at({3, 3, 0}), 16.0f));
    std::cout << "PASSED: test_pool_does_not_mutate_original\n";
}

// ─── Error paths ──────────────────────────────────────────────────────────────
void test_pool_invalid_non_divisible() {
    // 5x5 with 2x2 stride 2: (5-2)%2 = 1 ≠ 0 → should throw
    std::vector<float> data(25);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({5, 5, 1}, data);

    try {
        pooling::maxPooling(a, 2, 2);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_pool_invalid_non_divisible\n";
}

void test_pool_invalid_pool_larger_than_input() {
    // poolSize 5 on a 4x4 input → numerW negative → should throw
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor a({4, 4, 1}, data);

    try {
        pooling::maxPooling(a, 5, 1);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_pool_invalid_pool_larger_than_input\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // basic
    test_pool_basic_values();
    test_pool_basic_shape();

    // argmax
    test_pool_argmax_values();
    test_pool_argmax_size_matches_pooled();

    // depth
    test_pool_depth_independent();
    test_pool_depth_argmax_indices();

    // window / stride variations
    test_pool_3x3_stride3();
    test_pool_overlapping_stride1();

    // negative values (-inf init)
    test_pool_all_negative();

    // no mutation
    test_pool_does_not_mutate_original();

    // error paths
    test_pool_invalid_non_divisible();
    test_pool_invalid_pool_larger_than_input();

    std::cout << "\nAll pooling tests passed\n";
    return 0;
}