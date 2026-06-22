#ifndef RELU_LAYER_H
#define RELU_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "activations.h"
#include "base_layer.h"
#include "tensor.h"

// ─── Class declarations ──────────────────────────────────────────────────────
/**
 * @brief A ReLU activation layer.
 *
 * @par Description
 * Applies the ReLU activation (max(0, x)) element-wise on the forward pass, and
 * gates the incoming gradient on the backward pass so that only positions whose
 * input was positive let gradient through. Has no learnable parameters.
 *
 * @note The forward input is stashed so the backward pass can rebuild the
 * derivative mask (1 where the input was positive, 0 otherwise) from it.
 */
class ReLULayer : public BaseLayer {

private:
    // Stashed during forward, needed for backward. The value is
    // overwritten on the first forward() call before it is ever read.
    Tensor input;

public:
    /**
     * @brief Applies ReLU to the input and stashes the input for backward.
     *
     * @param input The output of the previous layer.
     * @return A tensor of the same shape with negatives set to 0.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Gates the incoming gradient by which inputs were positive.
     *
     * @par Description
     * Returns gradOutput element-wise multiplied by the ReLU derivative of the
     * stashed forward input. Gradient passes through where the input was
     * positive and is zeroed where it was <= 0.
     *
     * @param gradOutput The gradient of the loss with respect to this layer's output.
     * @return The gradient of the loss with respect to this layer's input.
     */
    Tensor backward(const Tensor& gradOutput) override;
};

#endif