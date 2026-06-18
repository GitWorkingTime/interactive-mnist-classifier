// ─── Imports ─────────────────────────────────────────────────────────────────
#include "activations.h"

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