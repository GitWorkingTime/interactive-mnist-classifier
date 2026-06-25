#ifndef CONV_LAYER_H
#define CONV_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base_layer.h"
#include "tensor.h"
#include "weights_init.h"
#include <fstream>
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
     *
     * @par Description
     * Builds numFilters filters, each of shape {filterSize, filterSize, inputDepth},
     * He-initialized with fanIn = filterSize * filterSize * inputDepth. Biases start
     * at zero.
     *
     * @param numFilters Number of filters (= output depth).
     * @param filterSize Side length of each square filter.
     * @param inputDepth Depth of the input the filters span.
     * @param stride     Convolution stride (default 1).
     * @param padding    Zero-padding applied in the forward convolution (default 0).
     *
     * @throws std::invalid_argument if numFilters, filterSize, or inputDepth < 1,
     *         if stride < 1, or if padding < 0.
     */
    ConvLayer(int numFilters, int filterSize, int inputDepth, int stride = 1, int padding = 0);

    /**
     * @brief Convolves the input with each filter and stacks the results.
     *
     * @par Description
     * Runs the input through each filter (collapsing depth per filter) and stacks
     * the per-filter outputs into a multi-channel feature map. Stashes the input
     * for use in backward.
     *
     * @param input The input feature map, shape {W, H, inputDepth}.
     * @return The output feature map, shape {outW, outH, numFilters}.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Computes filter/bias gradients and the input gradient.
     *
     * @par Description
     * From the incoming gradient, computes the bias gradient (sum of each filter's
     * gradient slice), the filter gradient (input convolved with the gradient slice,
     * per channel), and the input gradient (full convolution of the gradient slice
     * with the 180-rotated filter, per channel, accumulated across filters). The
     * filter and bias gradients are stored for the optimizer; the input gradient is
     * returned.
     *
     * @param gradOutput Gradient of the loss w.r.t. this layer's output.
     * @return Gradient w.r.t. this layer's input, shaped like the input.
     */
    Tensor backward(const Tensor& gradOutput) override;

    /**
     * @brief Updates filters and biases using the gradients from backward.
     *
     * @par Description
     * Applies one gradient-descent step to every filter and bias:
     * parameter -= learningRate * gradient. Must be called after backward().
     *
     * @param learningRate The step size for gradient descent.
     */
    void updateWeights(float learningRate) override;

    // ─── Test-only accessors ──────────────────────────────────────────────────
    // These exist so tests can inject known filters/biases and read back the
    // computed gradients, allowing exact-value verification against hand-computed
    // results (since He-initialized filters are random and not predictable).

    /** @brief Test-only: replaces all filters with the given set. */
    void setFilters(const std::vector<Tensor>& f) { filters = f; }

    /** @brief Test-only: replaces the biases with the given values. */
    void setBias(const std::vector<float>& b) { biases = b; }

    /** @brief Test-only: returns a copy of filter i. */
    Tensor getFilter(int i) const { return filters[i]; }

    /** @brief Test-only: returns a copy of filter i's gradient (valid after backward). */
    Tensor getFilterGrad(int i) const { return filterGrads[i]; }

    /** @brief Test-only: returns bias i's gradient (valid after backward). */
    float getBiasGrad(int i) const { return biasGrads[i]; }

    /**
     * @brief Writes the filters and biases to a binary file stream.
     *
     * @par Description
     * Serializes each filter (its shape followed by its data) and then the biases,
     * in that order. load() must read them back in the same order. Overrides the
     * BaseLayer no-op since this layer has learnable parameters.
     *
     * @param file An open binary output stream positioned where this layer writes.
     */
    void save(std::ofstream& file) const override;

    /**
     * @brief Reads the filters and biases from a binary file stream.
     *
     * @par Description
     * Deserializes the filters and biases written by save(), in the same order.
     * Relies on numFilters (set at construction) to know how many filters to read,
     * so the layer must be constructed with the same configuration that was saved.
     *
     * @param file An open binary input stream positioned at this layer's data.
     */
    void load(std::ifstream& file) override;
};

#endif