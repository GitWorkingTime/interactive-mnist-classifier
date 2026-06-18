#ifndef LOSS_H
#define LOSS_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"

// ─── Function declarations ───────────────────────────────────────────────────
namespace loss {
/**
 * @brief Computes the cross-entropy loss between a prediction and the true label.
 *
 * @par Description
 * Measures how far the predicted probability distribution is from the true
 * (one-hot) label. The prediction is expected to be a probability distribution,
 * i.e. the output of softMax. Returns a single scalar: low when the prediction
 * assigns high probability to the correct class, growing without bound as it
 * assigns lower probability to the correct class.
 *
 * loss = -Σ target_i * log(prediction_i)
 *
 * Because the target is one-hot (1 at the correct class, 0 elsewhere), every
 * term except the correct class vanishes, so this reduces to -log(p) where p is
 * the probability assigned to the correct class.
 *
 * @param prediction A probability distribution (softMax output), same shape as target
 * @param target     The true label, one-hot encoded (1 at the correct class, 0 elsewhere)
 *
 * @return The scalar cross-entropy loss (always >= 0).
 *
 * @note A small epsilon is added inside the logarithm (log(prediction + 1e-7))
 * to avoid log(0) = -infinity when a predicted probability is exactly zero.
 *
 * @par Example
 * @code
 * Tensor prediction({3, 1, 1}, {0.1f, 0.2f, 0.7f});
 * Tensor target({3, 1, 1}, {0.0f, 0.0f, 1.0f});  // correct class is index 2
 * float loss = crossEntropy(prediction, target); // -log(0.7) ≈ 0.357
 * @endcode
 */
float crossEntropy(const Tensor& prediction, const Tensor& target);

/**
 * @brief Computes the gradient of cross-entropy loss with respect to the prediction.
 *
 * @par Description
 * Returns the gradient that begins the backward pass. When the prediction is the
 * output of softMax, the gradient of the combined softmax + cross-entropy
 * simplifies to a plain element-wise difference:
 *
 * gradient = prediction - target
 *
 * Each element is positive where the network over-assigned probability (pushing
 * that class down) and negative where it under-assigned (pushing it up). This
 * simplification is why softMax does not need a separate derivative function.
 *
 * @param prediction A probability distribution (softMax output), same shape as target
 * @param target     The true label, one-hot encoded
 *
 * @return A new Tensor (same shape as the inputs) holding prediction - target.
 *
 * @note This assumes the prediction came from softMax. The clean
 * prediction - target form is only valid for the softmax + cross-entropy pairing;
 * it is not the gradient of cross-entropy alone.
 *
 * @par Example
 * @code
 * Tensor prediction({3, 1, 1}, {0.1f, 0.2f, 0.7f});
 * Tensor target({3, 1, 1}, {0.0f, 0.0f, 1.0f});
 * Tensor grad = crossEntropyGradient(prediction, target);
 * // grad data: {0.1, 0.2, -0.3}
 * @endcode
 */
Tensor crossEntropyGradient(const Tensor& prediction, const Tensor& target);

} // namespace loss

#endif