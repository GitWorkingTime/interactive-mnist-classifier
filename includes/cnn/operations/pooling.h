#ifndef POOLING_H
#define POOLING_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"
#include <limits>
#include <stdexcept>
#include <vector>

// ─── Pooling ─────────────────────────────────────────────────────────────────
namespace pooling {
/**
 * @brief struct that contains both the pooled tensor and the flat index into the
 *        original tensor where the max values come from.
 *
 * @par Description
 * The `pooled` tensor contains a compressed/down-sampled tensor of its input.
 * `argmax[i]` is the flat index of the input tensor that produced the `pooled` i-th
 * element.
 *
 * argmax is used in backpropagation to route each gradient back to the single position
 * that won its window.
 *
 * `argmax.size() === pooled.getSize()`
 */
struct PoolResult {
    Tensor pooled;
    std::vector<int> argmax;
};

/**
 * @brief Max pools an input tensor and stashes its argmax.
 *
 * @par Synopsis
 * pooling::PoolResult maxPooling(const Tensor& tensor, int poolSize = 2, int stride = 2)
 *
 * @par Description
 * Slides a poolSize × poolSize window across the W and H dimensions, choosing the
 * largest value in each window. Depth is unchanged — pooling is applied independently
 * per channel. Also stashes the argmax (the flat index into the input tensor of the
 * element that won each window) for later use in backpropagation.
 *
 * @param tensor   The input tensor to apply max pooling to
 * @param poolSize The side length of the pooling window (default: 2)
 * @param stride   Step size of the window (default: 2)
 *
 * @return A PoolResult containing the pooled tensor and the argmax vector.
 *         argmax.size() == pooled.getSize().
 *
 * @throws std::invalid_argument if poolSize/stride do not result in a positive
 *         integer for the resulting tensor's size
 *
 * @note The output width and height are calculated using: ((W - poolSize) / stride) + 1.
 * The numerator (W - poolSize) must be non-negative and divisible by stride, else
 * the parameters are invalid. Depth (D) is unchanged.
 *
 * For example: 4x4 input with poolSize = 2, stride = 2.
 * (4 - 2) / 2 + 1 => 2/2 + 1 => 2  →  output is 2x2
 *
 * @par Example
 * @code
 * #include <numeric>
 * std::vector<float> data(16);
 * std::iota(data.begin(), data.end(), 1.0f);   // 1..16
 *
 * Tensor a({4, 4, 1}, data);
 * pooling::PoolResult b = maxPooling(a, 2, 2);
 * // b.pooled → shape {2, 2, 1}, data {6, 8, 14, 16}
 * // b.argmax → {5, 7, 13, 15} (flat input indices of the maxes)
 * @endcode
 */
PoolResult maxPooling(const Tensor& tensor, int poolSize = 2, int stride = 2);
} // namespace pooling

#endif