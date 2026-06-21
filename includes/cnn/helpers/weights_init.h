#ifndef WEIGHTS_INIT_H
#define WEIGHTS_INIT_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"
#include <cmath>
#include <random>
#include <vector>

// ─── Function declarations ───────────────────────────────────────────────────
namespace weights {

/**
 * @brief Creates a tensor of the given shape filled with He-initialized weights.
 *
 * @par Description
 * Fills a tensor with random values drawn from a normal distribution with mean 0
 * and standard deviation sqrt(2 / fanIn). This is He initialization, designed for
 * layers using ReLU: it keeps the variance of activations roughly stable as
 * signals pass through the network, preventing them from vanishing or exploding.
 *
 * @param shape The shape of the weight tensor in {W, H, D} format
 * @param fanIn The number of input connections feeding each output. For a conv
 *              filter this is filterW * filterH * inputDepth; for a fully connected
 *              layer it is the number of input neurons.
 *
 * @return A new Tensor of the given shape filled with He-initialized values.
 *
 * @note Successive calls produce different values (the underlying random
 * generator persists across calls), so multiple filters/weight tensors will not
 * be initialized identically.
 *
 * @par Example
 * @code
 * // A 3x3 single-channel conv filter: fanIn = 3 * 3 * 1 = 9
 * Tensor filter = weights::heInit({3, 3, 1}, 9);
 *
 * // An FC weight matrix with 128 input neurons
 * Tensor w = weights::heInit({10, 128, 1}, 128);
 * @endcode
 */
Tensor heInit(const std::vector<int>& shape, int fanIn);

} // namespace weights

#endif