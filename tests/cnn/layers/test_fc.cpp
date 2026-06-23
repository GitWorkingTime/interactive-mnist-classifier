#include "fc_layer.h"
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
void test_fc_invalid_inputs() {
    try {
        FCLayer layer(0, 10);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_fc_invalid_inputs\n";
}

void test_fc_invalid_outputs() {
    try {
        FCLayer layer(784, 0);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_fc_invalid_outputs\n";
}

// ─── Forward shape ────────────────────────────────────────────────────────────
void test_fc_forward_output_shape() {
    FCLayer layer(4, 3);                // 4 inputs → 3 outputs
    Tensor in({2, 2, 1}, {1, 2, 3, 4}); // flattens to 4 elements
    Tensor out = layer.forward(in);
    assert(out.getShape()[0] == 1);
    assert(out.getShape()[1] == 3); // numOutputs
    assert(out.getShape()[2] == 1);
    std::cout << "PASSED: test_fc_forward_output_shape\n";
}

void test_fc_forward_accepts_3d_input() {
    // Input gets flattened, so any shape whose W*H*D == numInputs works
    FCLayer layer(8, 2);                            // 8 inputs
    Tensor in({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}); // 2*2*2 = 8
    Tensor out = layer.forward(in);
    assert(out.getShape()[1] == 2);
    std::cout << "PASSED: test_fc_forward_accepts_3d_input\n";
}

void test_fc_forward_wrong_input_size_throws() {
    // Input flattens to 5, but layer expects 4 → multiply dimension mismatch
    FCLayer layer(4, 3);
    Tensor in({5, 1, 1}, {1, 2, 3, 4, 5});
    try {
        layer.forward(in);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_fc_forward_wrong_input_size_throws\n";
}

// ─── Backward shapes ──────────────────────────────────────────────────────────
void test_fc_backward_input_grad_shape() {
    // dIn must be reshaped back to the ORIGINAL input shape
    FCLayer layer(4, 3);
    Tensor in({2, 2, 1}, {1, 2, 3, 4});
    layer.forward(in);

    Tensor grad({1, 3, 1}, {0.1f, 0.2f, 0.3f});
    Tensor dIn = layer.backward(grad);
    assert(dIn.getShape() == in.getShape()); // {2,2,1}, not the flat shape
    std::cout << "PASSED: test_fc_backward_input_grad_shape\n";
}

void test_fc_backward_input_grad_size() {
    FCLayer layer(8, 2);
    Tensor in({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    layer.forward(in);

    Tensor grad({1, 2, 1}, {0.5f, 0.5f});
    Tensor dIn = layer.backward(grad);
    assert(dIn.getSize() == 8); // matches the original input element count
    std::cout << "PASSED: test_fc_backward_input_grad_size\n";
}

// ─── Weight update behavior ───────────────────────────────────────────────────
void test_fc_update_changes_output() {
    // After a gradient step, the same input should generally produce a different
    // output (weights moved). This confirms updateWeights actually mutates weights.
    FCLayer layer(4, 3);
    Tensor in({2, 2, 1}, {1, 2, 3, 4});

    Tensor before = layer.forward(in);

    // run a backward to populate gradients, then update
    Tensor grad({1, 3, 1}, {1.0f, 1.0f, 1.0f});
    layer.backward(grad);
    layer.updateWeights(0.1f);

    Tensor after = layer.forward(in);

    // At least one output should have changed
    bool changed = false;
    for (int i = 0; i < 3; ++i)
        if (!floatEq(before.at({0, i, 0}), after.at({0, i, 0})))
            changed = true;
    assert(changed);
    std::cout << "PASSED: test_fc_update_changes_output\n";
}

void test_fc_update_zero_gradient_no_change() {
    // A zero gradient should leave weights (and thus output) unchanged.
    FCLayer layer(4, 3);
    Tensor in({2, 2, 1}, {1, 2, 3, 4});

    Tensor before = layer.forward(in);

    Tensor zeroGrad({1, 3, 1}, {0.0f, 0.0f, 0.0f});
    layer.backward(zeroGrad);
    layer.updateWeights(0.1f);

    Tensor after = layer.forward(in);

    for (int i = 0; i < 3; ++i)
        assert(floatEq(before.at({0, i, 0}), after.at({0, i, 0})));
    std::cout << "PASSED: test_fc_update_zero_gradient_no_change\n";
}

void test_fc_update_zero_lr_no_change() {
    // Learning rate 0 means no step → output unchanged regardless of gradient.
    FCLayer layer(4, 3);
    Tensor in({2, 2, 1}, {1, 2, 3, 4});

    Tensor before = layer.forward(in);

    Tensor grad({1, 3, 1}, {5.0f, 5.0f, 5.0f});
    layer.backward(grad);
    layer.updateWeights(0.0f); // no movement

    Tensor after = layer.forward(in);

    for (int i = 0; i < 3; ++i)
        assert(floatEq(before.at({0, i, 0}), after.at({0, i, 0})));
    std::cout << "PASSED: test_fc_update_zero_lr_no_change\n";
}

// ─── Deterministic math check (forward with controlled bias) ──────────────────
void test_fc_forward_reproducible() {
    // Same layer, same input, called twice → identical output (no hidden state
    // corruption between forward calls).
    FCLayer layer(4, 3);
    Tensor in({2, 2, 1}, {1, 2, 3, 4});
    Tensor out1 = layer.forward(in);
    Tensor out2 = layer.forward(in);
    for (int i = 0; i < 3; ++i)
        assert(floatEq(out1.at({0, i, 0}), out2.at({0, i, 0})));
    std::cout << "PASSED: test_fc_forward_reproducible\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // constructor
    test_fc_invalid_inputs();
    test_fc_invalid_outputs();

    // forward
    test_fc_forward_output_shape();
    test_fc_forward_accepts_3d_input();
    test_fc_forward_wrong_input_size_throws();
    test_fc_forward_reproducible();

    // backward
    test_fc_backward_input_grad_shape();
    test_fc_backward_input_grad_size();

    // weight update
    test_fc_update_changes_output();
    test_fc_update_zero_gradient_no_change();
    test_fc_update_zero_lr_no_change();

    std::cout << "\nAll FC layer tests passed\n";
    return 0;
}