#ifndef MAXPOOLING_LAYER_H
#define MAXPOOLING_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base_layer.h"
#include "pooling.h"
#include "tensor.h"
#include <stdexcept>
#include <vector>

// ─── Class declarations ──────────────────────────────────────────────────────
/**
 * @brief A max-pooling layer.
 *
 * @par Description
 * On the forward pass, downsamples each channel independently by taking the max
 * of each poolSize x poolSize window (stride between windows). On the backward
 * pass, routes each output's gradient back to the single input position that won
 * its window (the argmax), leaving all other positions zero. Has no learnable
 * parameters.
 *
 * @note The argmax indices and the input shape are stashed during forward so the
 * backward pass can scatter the gradient to the winning positions and rebuild a
 * tensor of the input's original shape.
 */
class MaxPoolingLayer : public BaseLayer {
private:
    int poolSize;
    int stride;
    std::vector<int> inputShape; // {W, H, D} of the forward input, to size backward output
    std::vector<int> argmax;     // flat input index that won each output window

public:
    /**
     * @brief Constructs a max-pooling layer with a given window size and stride.
     *
     * @par Description
     * Stores the pooling configuration used on every forward pass. With the
     * defaults (2x2 window, stride 2) the layer halves the width and height of
     * each channel, the most common pooling setup. Depth is never changed.
     *
     * @param poolSize Side length of the square pooling window (default 2).
     * @param stride   Step between successive window positions (default 2).
     *
     * @throws if either poolSize or stride is less than or equal to 0
     */
    MaxPoolingLayer(int poolSize = 2, int stride = 2);

    /**
     * @brief Pools the input and stashes the argmax and input shape for backward.
     *
     * @par Description
     * Applies max pooling over the width and height of each channel independently,
     * producing a smaller output of the same depth. Internally calls the pooling
     * operation, which returns both the pooled values and, for each output element,
     * the flat index into the input that held the window's maximum. This argmax
     * vector and the input's shape are stored so backward() can route gradients
     * correctly.
     *
     * @param input The output of the previous layer, shape {W, H, D}.
     *
     * @return The pooled tensor, shape {(W - poolSize)/stride + 1,
     *         (H - poolSize)/stride + 1, D}.
     *
     * @throws std::invalid_argument if poolSize/stride do not divide the input
     *         evenly (propagated from the pooling operation).
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Routes each output gradient back to the input position that won its window.
     *
     * @par Description
     * Builds a gradient tensor of the original input's shape, initialized to zero.
     * For each element of gradOutput, the gradient is added to the single input
     * position recorded in the stashed argmax (the position that produced that
     * output's maximum during forward). All input positions that did not win a
     * window receive zero gradient, because they had no effect on the output.
     *
     * @par
     * Gradients are accumulated (+=) rather than overwritten, so that if pooling
     * windows overlap (stride < poolSize) and one input position won more than one
     * window, the contributions from each are summed.
     *
     * @param gradOutput The gradient of the loss with respect to this layer's
     *                   output. Must match the pooled output's shape.
     *
     * @return The gradient of the loss with respect to this layer's input,
     *         shaped like the original input — zero everywhere except at the
     *         winning positions.
     */
    Tensor backward(const Tensor& gradOutput) override;
};

#endif