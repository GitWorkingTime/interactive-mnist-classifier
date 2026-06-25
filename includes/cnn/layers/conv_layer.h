#ifndef CONV_LAYER_H
#define CONV_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base_layer.h"
#include "tensor.h"
#include "weights_init.h"
#include <stdexcept>
#include <vector>

// ─── Class declarations ──────────────────────────────────────────────────────
/**
 * @brief A convolutional layer with multiple filters.
 *
 * @par Description
 * Holds a set of filters (each spanning the full input depth) and convolves the
 * input with each one, stacking the results into a multi-channel output (one
 * channel per filter). Forward produces a {outW, outH, numFilters} feature map.
 * Backward computes gradients for the filters and biases (kept for the optimizer)
 * and the gradient for the input (returned to the previous layer).
 *
 * @note Has learnable parameters (filters, biases). The forward input is stashed
 * for backward. Assumes stride 1 for the backward input-gradient computation.
 */
class ConvLayer : public BaseLayer {
private:
    int numFilters;
    int filterSize; // side length of each square filter
    int inputDepth; // depth each filter spans (must match input depth)
    int stride;
    int padding;

    // Parameters: one filter and one bias per output channel
    std::vector<Tensor> filters; // each {filterSize, filterSize, inputDepth}
    std::vector<float> biases;   // one per filter

    // Stash for backward
    Tensor input; // the forward input

    // Gradients (computed in backward, used by the optimizer)
    std::vector<Tensor> filterGrads; // same shapes as filters
    std::vector<float> biasGrads;    // one per filter

public:
    /**
     * @brief Constructs a conv layer and He-initializes its filters.
     * @param numFilters Number of filters (= output depth).
     * @param filterSize Side length of each square filter.
     * @param inputDepth Depth of the input the filters span.
     * @param stride     Convolution stride (default 1).
     * @param padding    Zero-padding applied in the forward convolution (default 0).
     */
    ConvLayer(int numFilters, int filterSize, int inputDepth, int stride = 1, int padding = 0);

    /**
     * @brief Convolves the input with each filter and stacks the results.
     * @param input The input feature map, shape {W, H, inputDepth}.
     * @return The output feature map, shape {outW, outH, numFilters}.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Computes filter/bias gradients and the input gradient.
     * @param gradOutput Gradient of the loss w.r.t. this layer's output.
     * @return Gradient w.r.t. this layer's input, shaped like the input.
     */
    Tensor backward(const Tensor& gradOutput) override;

    /**
     * @brief Updates filters and biases using the gradients from backward.
     * @param learningRate The step size for gradient descent.
     */
    void updateWeights(float learningRate);

    // Add to ConvLayer's public section, for testing:
    void setFilters(const std::vector<Tensor>& f) { filters = f; }
    void setBias(const std::vector<float>& b) { biases = b; }
    Tensor getFilter(int i) const { return filters[i]; }
    Tensor getFilterGrad(int i) const { return filterGrads[i]; }
    float getBiasGrad(int i) const { return biasGrads[i]; }
};

#endif