#include "ReLU_layer.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── Forward ──────────────────────────────────────────────────────────────────
void test_relu_layer_forward() {
    ReLULayer layer;
    Tensor in({2, 2, 1}, {-1, 2, -3, 4});
    Tensor out = layer.forward(in);
    // ReLU: negatives → 0, positives unchanged
    assert(floatEq(out.at({0, 0, 0}), 0.0f)); // -1 → 0
    assert(floatEq(out.at({1, 0, 0}), 2.0f)); //  2 → 2
    assert(floatEq(out.at({0, 1, 0}), 0.0f)); // -3 → 0
    assert(floatEq(out.at({1, 1, 0}), 4.0f)); //  4 → 4
    std::cout << "PASSED: test_relu_layer_forward\n";
}

void test_relu_layer_forward_preserves_shape() {
    ReLULayer layer;
    Tensor in({3, 4, 2}, std::vector<float>(24, -1.0f));
    Tensor out = layer.forward(in);
    assert(out.getShape() == in.getShape());
    std::cout << "PASSED: test_relu_layer_forward_preserves_shape\n";
}

// ─── Backward (stash verification) ────────────────────────────────────────────
void test_relu_layer_backward_gates_gradient() {
    // The key test: backward must use the input stashed during forward.
    // Positions where the forward input was <= 0 must zero the gradient.
    ReLULayer layer;
    Tensor in({2, 2, 1}, {-1, 2, -3, 4});
    layer.forward(in); // stashes {-1, 2, -3, 4}

    Tensor grad({2, 2, 1}, {0.5f, 0.5f, 0.5f, 0.5f});
    Tensor dIn = layer.backward(grad);

    // input -1 (<=0) → gradient blocked → 0
    assert(floatEq(dIn.at({0, 0, 0}), 0.0f));
    // input 2 (>0) → gradient passes → 0.5
    assert(floatEq(dIn.at({1, 0, 0}), 0.5f));
    // input -3 (<=0) → blocked → 0
    assert(floatEq(dIn.at({0, 1, 0}), 0.0f));
    // input 4 (>0) → passes → 0.5
    assert(floatEq(dIn.at({1, 1, 0}), 0.5f));
    std::cout << "PASSED: test_relu_layer_backward_gates_gradient\n";
}

void test_relu_layer_backward_uses_forward_input_not_gradient() {
    // Stash verification: the mask comes from the FORWARD input, not the gradient.
    // Here the gradient is all positive, but the input had negatives — so some
    // gradients must still be zeroed. If backward wrongly used the gradient's
    // sign (all positive) instead of the stashed input, nothing would zero out.
    ReLULayer layer;
    Tensor in({3, 1, 1}, {-5, 1, -2}); // two negatives in the input
    layer.forward(in);

    Tensor grad({3, 1, 1}, {1.0f, 1.0f, 1.0f}); // all positive gradient
    Tensor dIn = layer.backward(grad);

    assert(floatEq(dIn.at({0, 0, 0}), 0.0f)); // input -5 → must be zeroed
    assert(floatEq(dIn.at({1, 0, 0}), 1.0f)); // input  1 → passes
    assert(floatEq(dIn.at({2, 0, 0}), 0.0f)); // input -2 → must be zeroed
    std::cout << "PASSED: test_relu_layer_backward_uses_forward_input_not_gradient\n";
}

void test_relu_layer_stash_updates_each_forward() {
    // Calling forward again must re-stash with the NEW input, so backward
    // reflects the most recent forward, not the first.
    ReLULayer layer;

    Tensor first({2, 1, 1}, {3, -3});
    layer.forward(first); // stash {3, -3}

    Tensor second({2, 1, 1}, {-7, 8});
    layer.forward(second); // re-stash {-7, 8}

    Tensor grad({2, 1, 1}, {1.0f, 1.0f});
    Tensor dIn = layer.backward(grad);

    // Must use the SECOND input {-7, 8}: position 0 blocked, position 1 passes
    assert(floatEq(dIn.at({0, 0, 0}), 0.0f)); // -7 → blocked
    assert(floatEq(dIn.at({1, 0, 0}), 1.0f)); //  8 → passes
    std::cout << "PASSED: test_relu_layer_stash_updates_each_forward\n";
}

void test_relu_layer_backward_scales_gradient() {
    // Where the input is positive, the gradient passes through with its
    // actual magnitude (not just 1) — confirms it's gradient * mask, not just mask.
    ReLULayer layer;
    Tensor in({3, 1, 1}, {1, 2, 3}); // all positive → all pass
    layer.forward(in);

    Tensor grad({3, 1, 1}, {0.2f, 0.7f, 0.9f});
    Tensor dIn = layer.backward(grad);

    assert(floatEq(dIn.at({0, 0, 0}), 0.2f));
    assert(floatEq(dIn.at({1, 0, 0}), 0.7f));
    assert(floatEq(dIn.at({2, 0, 0}), 0.9f));
    std::cout << "PASSED: test_relu_layer_backward_scales_gradient\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    test_relu_layer_forward();
    test_relu_layer_forward_preserves_shape();
    test_relu_layer_backward_gates_gradient();
    test_relu_layer_backward_uses_forward_input_not_gradient();
    test_relu_layer_stash_updates_each_forward();
    test_relu_layer_backward_scales_gradient();

    std::cout << "\nAll ReLU layer tests passed\n";
    return 0;
}