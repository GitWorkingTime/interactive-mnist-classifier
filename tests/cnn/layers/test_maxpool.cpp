#include "maxPooling_layer.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── Constructor validation ───────────────────────────────────────────────────
void test_pool_layer_invalid_poolsize() {
    try {
        MaxPoolingLayer layer(0, 2);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_pool_layer_invalid_poolsize\n";
}

void test_pool_layer_invalid_stride() {
    try {
        MaxPoolingLayer layer(2, 0);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_pool_layer_invalid_stride\n";
}

// ─── Forward ──────────────────────────────────────────────────────────────────
void test_pool_layer_forward_values() {
    // 4x4 input 1..16, 2x2 stride 2 → {6, 8, 14, 16}
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor in({4, 4, 1}, data);

    MaxPoolingLayer layer(2, 2);
    Tensor out = layer.forward(in);

    assert(out.getShape()[0] == 2);
    assert(out.getShape()[1] == 2);
    assert(out.getShape()[2] == 1);
    assert(floatEq(out.at({0, 0, 0}), 6.0f));
    assert(floatEq(out.at({1, 0, 0}), 8.0f));
    assert(floatEq(out.at({0, 1, 0}), 14.0f));
    assert(floatEq(out.at({1, 1, 0}), 16.0f));
    std::cout << "PASSED: test_pool_layer_forward_values\n";
}

// ─── Backward (the scatter) ───────────────────────────────────────────────────
void test_pool_layer_backward_routes_to_winners() {
    // Forward stashes argmax {5, 7, 13, 15} for the 1..16 input.
    // Backward must place each gradient at its winning input index, zero elsewhere.
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor in({4, 4, 1}, data);

    MaxPoolingLayer layer(2, 2);
    layer.forward(in); // stash argmax + shape

    Tensor grad({2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor dIn = layer.backward(grad);

    const std::vector<float>& d = dIn.getData();
    // gradients land at the winning flat indices
    assert(floatEq(d[5], 1.0f));  // window 0 winner (value 6)
    assert(floatEq(d[7], 2.0f));  // window 1 winner (value 8)
    assert(floatEq(d[13], 3.0f)); // window 2 winner (value 14)
    assert(floatEq(d[15], 4.0f)); // window 3 winner (value 16)
    // every other position is zero
    for (int i = 0; i < 16; ++i)
        if (i != 5 && i != 7 && i != 13 && i != 15)
            assert(floatEq(d[i], 0.0f));
    std::cout << "PASSED: test_pool_layer_backward_routes_to_winners\n";
}

void test_pool_layer_backward_output_shape_is_input_shape() {
    // Backward returns a tensor shaped like the ORIGINAL input, not the pooled output
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor in({4, 4, 1}, data);

    MaxPoolingLayer layer(2, 2);
    layer.forward(in);

    Tensor grad({2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor dIn = layer.backward(grad);

    assert(dIn.getShape() == in.getShape()); // {4,4,1}, not {2,2,1}
    std::cout << "PASSED: test_pool_layer_backward_output_shape_is_input_shape\n";
}

void test_pool_layer_backward_size_mismatch_throws() {
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 1.0f);
    Tensor in({4, 4, 1}, data);

    MaxPoolingLayer layer(2, 2);
    layer.forward(in);

    // gradOutput has the wrong number of elements (should be 4)
    Tensor wrongGrad({3, 1, 1}, {1.0f, 2.0f, 3.0f});
    try {
        layer.backward(wrongGrad);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_pool_layer_backward_size_mismatch_throws\n";
}

// ─── Overlapping windows (tests the += accumulation) ──────────────────────────
void test_pool_layer_backward_accumulates_on_overlap() {
    // 3x3 input with the max (9) at the center (index 4). With a 2x2 window and
    // stride 1, ALL FOUR windows are won by the center. Backward must ACCUMULATE
    // all four gradients there: 1+2+3+4 = 10. A '=' instead of '+=' would give 4.
    Tensor in({3, 3, 1}, {1, 2, 3, 4, 9, 5, 6, 7, 8});

    MaxPoolingLayer layer(2, 1); // overlapping: stride < poolSize
    Tensor out = layer.forward(in);
    // every window's max is the center value 9
    assert(floatEq(out.at({0, 0, 0}), 9.0f));
    assert(floatEq(out.at({1, 0, 0}), 9.0f));
    assert(floatEq(out.at({0, 1, 0}), 9.0f));
    assert(floatEq(out.at({1, 1, 0}), 9.0f));

    Tensor grad({2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor dIn = layer.backward(grad);

    const std::vector<float>& d = dIn.getData();
    assert(floatEq(d[4], 10.0f)); // 1+2+3+4 accumulated at the center
    // all other positions zero
    for (int i = 0; i < 9; ++i)
        if (i != 4)
            assert(floatEq(d[i], 0.0f));
    std::cout << "PASSED: test_pool_layer_backward_accumulates_on_overlap\n";
}

// ─── Depth independence ───────────────────────────────────────────────────────
void test_pool_layer_depth_independent() {
    // Two channels pooled separately; backward routes within each channel
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1.0f); // slice0: 1..16, slice1: 17..32
    Tensor in({4, 4, 2}, data);

    MaxPoolingLayer layer(2, 2);
    Tensor out = layer.forward(in);
    assert(out.getShape()[2] == 2); // depth preserved

    Tensor grad({2, 2, 2}, {1, 1, 1, 1, 2, 2, 2, 2});
    Tensor dIn = layer.backward(grad);
    assert(dIn.getShape() == in.getShape()); // {4,4,2}

    // slice 0 winners (indices 5,7,13,15) get gradient 1
    const std::vector<float>& d = dIn.getData();
    assert(floatEq(d[5], 1.0f));
    // slice 1 winners (indices 21,23,29,31) get gradient 2
    assert(floatEq(d[21], 2.0f));
    std::cout << "PASSED: test_pool_layer_depth_independent\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // constructor
    test_pool_layer_invalid_poolsize();
    test_pool_layer_invalid_stride();

    // forward
    test_pool_layer_forward_values();

    // backward
    test_pool_layer_backward_routes_to_winners();
    test_pool_layer_backward_output_shape_is_input_shape();
    test_pool_layer_backward_size_mismatch_throws();
    test_pool_layer_backward_accumulates_on_overlap();

    // depth
    test_pool_layer_depth_independent();

    std::cout << "\nAll max pooling layer tests passed\n";
    return 0;
}