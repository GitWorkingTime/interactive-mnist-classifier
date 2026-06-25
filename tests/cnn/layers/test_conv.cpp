#include "conv_layer.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// NOTE: ConvLayer He-initializes its filters randomly, so we cannot assert exact
// forward/backward values against random filters. To test the math precisely, we
// would need a way to set known filters. These tests assume a test-only setter or
// constructor exists (setFilters / setBias) to inject known weights. If your class
// does not have one, add a test-only setter; the gradient math CANNOT be verified
// against random filters.

// ─── Constructor validation ───────────────────────────────────────────────────
void test_conv_invalid_numfilters() {
    try {
        ConvLayer layer(0, 3, 1);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_conv_invalid_numfilters\n";
}

void test_conv_invalid_filtersize() {
    try {
        ConvLayer layer(8, 0, 1);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_conv_invalid_filtersize\n";
}

void test_conv_invalid_inputdepth() {
    try {
        ConvLayer layer(8, 3, 0);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_conv_invalid_inputdepth\n";
}

// ─── Forward shape (works with random filters) ────────────────────────────────
void test_conv_forward_output_shape() {
    // 8 filters, 3x3, input depth 1, on a 28x28 input → 26x26x8
    ConvLayer layer(8, 3, 1);
    Tensor in({28, 28, 1}, std::vector<float>(28 * 28, 0.5f));
    Tensor out = layer.forward(in);
    assert(out.getShape()[0] == 26);
    assert(out.getShape()[1] == 26);
    assert(out.getShape()[2] == 8); // depth == numFilters
    std::cout << "PASSED: test_conv_forward_output_shape\n";
}

void test_conv_forward_depth_equals_numfilters() {
    ConvLayer layer(16, 3, 1);
    Tensor in({10, 10, 1}, std::vector<float>(100, 1.0f));
    Tensor out = layer.forward(in);
    assert(out.getShape()[2] == 16);
    std::cout << "PASSED: test_conv_forward_depth_equals_numfilters\n";
}

// ─── Backward shape (works with random filters) ───────────────────────────────
void test_conv_backward_input_grad_shape() {
    ConvLayer layer(8, 3, 1);
    Tensor in({28, 28, 1}, std::vector<float>(28 * 28, 0.5f));
    layer.forward(in);

    Tensor grad({26, 26, 8}, std::vector<float>(26 * 26 * 8, 1.0f));
    Tensor dIn = layer.backward(grad);
    assert(dIn.getShape() == in.getShape()); // {28,28,1}
    std::cout << "PASSED: test_conv_backward_input_grad_shape\n";
}

// ─── Exact gradient math (requires injected known filters) ────────────────────
// These use setFilters/setBias to install a known filter so we can assert exact
// gradient values computed independently.

void test_conv_backward_depth1_exact() {
    ConvLayer layer(1, 2, 1);
    layer.setFilters({Tensor({2, 2, 1}, {1, 0, 0, 1})});
    layer.setBias({0.0f});

    Tensor in({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor out = layer.forward(in);

    // --- log forward ---
    std::cout << "forward (expect 6 8 12 14): ";
    for (float v : out.getData())
        std::cout << v << " ";
    std::cout << "\n";

    Tensor grad({2, 2, 1}, {1, 2, 3, 4});
    Tensor dIn = layer.backward(grad);

    // --- log all three gradients ---
    std::cout << "bias grad (expect 10): " << layer.getBiasGrad(0) << "\n";

    std::cout << "filter grad (expect 37 47 67 77): ";
    for (float v : layer.getFilterGrad(0).getData())
        std::cout << v << " ";
    std::cout << "\n";

    std::cout << "input grad (expect 1 2 0 3 5 2 0 3 4): ";
    for (float v : dIn.getData())
        std::cout << v << " ";
    std::cout << "\n";

    // assertions after, so we see all output first
    assert(floatEq(out.at({0, 0, 0}), 6.0f));
    assert(floatEq(out.at({1, 0, 0}), 8.0f));
    assert(floatEq(out.at({0, 1, 0}), 12.0f));
    assert(floatEq(out.at({1, 1, 0}), 14.0f));
    assert(floatEq(layer.getBiasGrad(0), 10.0f));
    Tensor fgTensor = layer.getFilterGrad(0);          // store the Tensor (stays alive)
    const std::vector<float>& fg = fgTensor.getData(); // reference into the living Tensor
    assert(floatEq(fg[0], 37.0f));
    assert(floatEq(fg[1], 47.0f));
    assert(floatEq(fg[2], 67.0f));
    assert(floatEq(fg[3], 77.0f));
    const std::vector<float>& ig = dIn.getData();
    float expected[9] = {1, 2, 0, 3, 5, 2, 0, 3, 4};
    for (int i = 0; i < 9; ++i)
        assert(floatEq(ig[i], expected[i]));
    std::cout << "PASSED: test_conv_backward_depth1_exact\n";
}

void test_conv_backward_depth2_exact() {
    // 1 filter, 2x2, input depth 2 — exercises the per-channel loops.
    // input ch0 = 1..9, ch1 = 10..18; filter ch0 = {1,0,0,1}, ch1 = {0,1,1,0}
    ConvLayer layer(1, 2, 2);
    layer.setFilters({Tensor({2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 0})});
    layer.setBias({0.0f});

    std::vector<float> inData = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                 10, 11, 12, 13, 14, 15, 16, 17, 18};
    Tensor in({3, 3, 2}, inData);
    Tensor out = layer.forward(in);
    // forward (depth collapses): {30, 34, 42, 46}
    assert(floatEq(out.at({0, 0, 0}), 30.0f));
    assert(floatEq(out.at({1, 0, 0}), 34.0f));
    assert(floatEq(out.at({0, 1, 0}), 42.0f));
    assert(floatEq(out.at({1, 1, 0}), 46.0f));

    Tensor grad({2, 2, 1}, {1, 2, 3, 4});
    Tensor dIn = layer.backward(grad);

    // bias grad = 10
    assert(floatEq(layer.getBiasGrad(0), 10.0f));

    // filter grad: ch0 {37,47,67,77}, ch1 {127,137,157,167}
    float expectedFg[8] = {37, 47, 67, 77, 127, 137, 157, 167};
    Tensor fgTensor = layer.getFilterGrad(0);          // hold the Tensor alive
    const std::vector<float>& fg = fgTensor.getData(); // safe reference
    for (int i = 0; i < 8; ++i)
        assert(floatEq(fg[i], expectedFg[i]));

    // input grad: ch0 {1,2,0,3,5,2,0,3,4}, ch1 {0,1,2,1,5,4,3,4,0}
    const std::vector<float>& ig = dIn.getData();
    float expectedIg[18] = {1, 2, 0, 3, 5, 2, 0, 3, 4,
                            0, 1, 2, 1, 5, 4, 3, 4, 0};
    for (int i = 0; i < 18; ++i)
        assert(floatEq(ig[i], expectedIg[i]));
    std::cout << "PASSED: test_conv_backward_depth2_exact\n";
}

// ─── Weight update ────────────────────────────────────────────────────────────
void test_conv_update_changes_filter() {
    ConvLayer layer(1, 2, 1);
    layer.setFilters({Tensor({2, 2, 1}, {1, 0, 0, 1})});
    layer.setBias({0.0f});

    Tensor in({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    layer.forward(in);
    Tensor grad({2, 2, 1}, {1, 2, 3, 4});
    layer.backward(grad);

    layer.updateWeights(0.01f);
    // filter[0] = {1,0,0,1} - 0.01*{37,47,67,77} = {0.63, -0.47, -0.67, 0.23}
    Tensor filterTensor = layer.getFilter(0); // hold it alive
    const std::vector<float>& fdata = filterTensor.getData();
    assert(floatEq(fdata[0], 1.0f - 0.01f * 37.0f));
    assert(floatEq(fdata[1], 0.0f - 0.01f * 47.0f));
    assert(floatEq(fdata[2], 0.0f - 0.01f * 67.0f));
    assert(floatEq(fdata[3], 1.0f - 0.01f * 77.0f));
    std::cout << "PASSED: test_conv_update_changes_filter\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // constructor
    test_conv_invalid_numfilters();
    test_conv_invalid_filtersize();
    test_conv_invalid_inputdepth();

    // forward (random filters OK)
    test_conv_forward_output_shape();
    test_conv_forward_depth_equals_numfilters();

    // backward shape (random filters OK)
    test_conv_backward_input_grad_shape();

    // exact gradient math (needs injected filters)
    test_conv_backward_depth1_exact();
    test_conv_backward_depth2_exact();

    // weight update
    test_conv_update_changes_filter();

    std::cout << "\nAll conv layer tests passed\n";
    return 0;
}