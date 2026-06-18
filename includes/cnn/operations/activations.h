#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <vector>

// ─── Activations ─────────────────────────────────────────────────────────────
namespace activations {
/**
 * @brief performs a ReLU operation on a given tensor
 *
 * @par Description
 * Given a tensor, it returns the original tensor but
 * for every negative value, it's replaced with 0
 *
 * ReLU, mathematically, is a piecewise function:
 * x <= 0: f(x) = 0
 * x > 0: f(x) = x
 *
 * @param tensor The tensor to perform the ReLU operation on
 *
 * @par Example
 * @code
 * Tensor a({2, 2, 1}, {-1, 2, 3, -4});
 * Tensor b = activations::ReLU(a); // Data: {0, 2, 3, 0};
 * @endcode
 */
Tensor ReLU(const Tensor& tensor);

/**
 * @brief performs a ReLU derivative operation on a given tensor
 *
 * @par Description
 * Given a tensor, it returns a tensor where every positive integer
 * is replaced with a 1.0 and every negative tensor is replaced with
 * a 0
 *
 * The derivative of ReLU, mathematically, is this piecewise function:
 * x <= 0: f'(x) = 0
 * x > 0: f'(x) = 1
 *
 * @param tensor The tensor to perform the ReLU derivative operation on
 *
 * @par Example
 * @code
 * Tensor a({2, 2, 1}, {-1, 2, 3, -4});
 * Tensor b = activations::ReLUDerivative(a); // Data: {0, 1, 1, 0}
 * @endcode
 */
Tensor ReLUDerivative(const Tensor& tensor);

/**
 * @brief Converts a tensor of raw scores into a probability distribution.
 *
 * @par Description
 * Applies the softmax function across ALL elements of the tensor, turning raw
 * scores (logits) into values in (0, 1) that sum to 1. Typically used on the
 * final layer's output to produce class confidences. Unlike ReLU, softmax is
 * not element-wise: each output depends on every input, since the denominator
 * is the sum of exponentials over the whole tensor.
 *
 * softMax(x_i) = exp(x_i) / Σ exp(x_j)
 *
 * @param tensor The input tensor of raw scores (logits)
 *
 * @return A new Tensor of the same shape whose elements lie in (0, 1) and sum to 1.
 *
 * @note For numerical stability the maximum element is subtracted from every
 * value before exponentiating: softMax(x_i) = exp(x_i - max) / Σ exp(x_j - max).
 * This is mathematically identical (the max cancels in the ratio) but prevents
 * overflow of exp() on large inputs.
 *
 * @note Treats the entire tensor as a single distribution. For a classifier this
 * is applied to the flattened {numClasses, 1, 1} output.
 *
 * @par Example
 * @code
 * Tensor logits({3, 1, 1}, {1.0f, 2.0f, 3.0f});
 * Tensor probs = activations::softMax(logits);
 * // probs ≈ {0.090, 0.245, 0.665}, sums to 1.0
 * @endcode
 */
Tensor softMax(const Tensor& tensor);

} // namespace activations

#endif