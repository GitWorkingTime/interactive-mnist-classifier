// ─── Imports ─────────────────────────────────────────────────────────────────
#include "activations.h"

// ─── Function Definitions ────────────────────────────────────────────────────
Tensor activations::ReLU(const Tensor& tensor) {
    // Make a copy to use for the new tensor. Don't mutate the original tensor
    std::vector<float> tensorData = tensor.getData();

    // Perform ReLU operation
    for (int i = 0; i < static_cast<int>(tensorData.size()); ++i) {
        if (tensorData[i] <= 0) {
            tensorData[i] = 0;
        }
    }

    return Tensor(tensor.getShape(), tensorData);
}

Tensor activations::ReLUDerivative(const Tensor& tensor) {
    // Make a copy to use for the new tensor. Don't mutate the original tensor
    std::vector<float> tensorData = tensor.getData();

    // Perform ReLU Derivative operation. At x = 0, ReLU is not differentiable
    // so we just use 0 by convention.
    for (int i = 0; i < static_cast<int>(tensorData.size()); ++i) {
        if (tensorData[i] <= 0) {
            tensorData[i] = 0;
        } else {
            tensorData[i] = 1;
        }
    }

    return Tensor(tensor.getShape(), tensorData);
}

Tensor activations::softMax(const Tensor& tensor) {
    const std::vector<float>& in = tensor.getData();
    std::vector<float> out(in.size());

    // 1. Find the max element. Subtracting it before exp() keeps every exponent
    //    <= 0, preventing overflow. Mathematically identical (max cancels in the
    //    ratio), just numerically safe.
    float maxVal = *std::max_element(in.begin(), in.end());

    // 2. Exponentiate each shifted value and accumulate the denominator.
    float sum = 0.0f;
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = std::exp(in[i] - maxVal);
        sum += out[i];
    }

    // 3. Normalize so the outputs form a probability distribution (sum to 1).
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] /= sum;
    }

    return Tensor(tensor.getShape(), out);
}