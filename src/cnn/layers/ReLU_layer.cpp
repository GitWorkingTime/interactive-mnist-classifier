// ─── Imports ─────────────────────────────────────────────────────────────────
#include "ReLU_layer.h"

// ─── Class definitions ───────────────────────────────────────────────────────
Tensor ReLULayer::forward(const Tensor& input) {
    // Stash input tensor
    this->input = input;

    // Perform ReLU operation
    Tensor reluTensor = activations::ReLU(input);
    return reluTensor;
}

Tensor ReLULayer::backward(const Tensor& gradOutput) {
    return gradOutput.hadamardProduct(activations::ReLUDerivative(this->input));
}