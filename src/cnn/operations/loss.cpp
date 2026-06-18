// ─── Imports ─────────────────────────────────────────────────────────────────
#include "loss.h"
#include <cmath>
#include <stdexcept>

// ─── Function declarations ───────────────────────────────────────────────────
float loss::crossEntropy(const Tensor& prediction, const Tensor& target) {
    // Shapes must match — each predicted probability pairs with its target
    if (prediction.getShape() != target.getShape()) {
        throw std::invalid_argument("ERROR: prediction and target shapes do not match");
    }

    const std::vector<float>& pred = prediction.getData();
    const std::vector<float>& tgt = target.getData();

    // loss = -Σ target_i * log(prediction_i)
    // The 1e-7 epsilon guards against log(0) = -infinity.
    float sum = 0.0f;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        sum += tgt[i] * std::log(pred[i] + 1e-7f);
    }

    return -sum;
}

Tensor loss::crossEntropyGradient(const Tensor& prediction, const Tensor& target) {
    if (prediction.getShape() != target.getShape()) {
        throw std::invalid_argument("ERROR: prediction and target shapes do not match");
    }

    const std::vector<float>& pred = prediction.getData();
    const std::vector<float>& tgt = target.getData();

    // For the softmax + cross-entropy pairing, the gradient simplifies to
    // prediction - target (element-wise).
    std::vector<float> grad(pred.size());
    for (std::size_t i = 0; i < pred.size(); ++i) {
        grad[i] = pred[i] - tgt[i];
    }

    return Tensor(prediction.getShape(), grad);
}