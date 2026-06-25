#ifndef FC_LAYER_H
#define FC_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base_layer.h"
#include "tensor.h"
#include "weights_init.h"
#include <fstream>
#include <stdexcept>
#include <vector>

// ─── Class declarations ──────────────────────────────────────────────────────
/**
 * @brief A fully connected (dense) layer.
 *
 * @par Description
 * Connects every input value to every output via a weight matrix plus a bias.
 * Forward flattens the input, computes W * input + bias, producing one value per
 * output neuron. Used as the network's final layer to produce class scores.
 *
 * @note Has learnable parameters (weights, biases). The forward input is stashed
 * for backward, which computes gradients for the weights and biases (kept for the
 * optimizer) and the gradient for the input (returned to the previous layer).
 */
class FCLayer : public BaseLayer {
private:
    int numInputs;  // flattened input size
    int numOutputs; // number of output neurons

    // Parameters
    Tensor weights; // {numInputs, numOutputs, 1}
    Tensor bias;    // {1, numOutputs, 1}

    // Stash (the flattened input) + the original input shape to reshape dIn back
    Tensor input;
    std::vector<int> inputShape;

    // Gradients (computed in backward, used by the optimizer)
    Tensor weightGrad; // same shape as weights
    Tensor biasGrad;   // same shape as bias

public:
    /**
     * @brief Constructs a fully connected layer and initializes its parameters.
     *
     * @par Description
     * Creates the weight matrix of shape {numInputs, numOutputs, 1} and a bias of
     * shape {1, numOutputs, 1}. Weights are He-initialized (random, scaled by
     * sqrt(2/numInputs)) to break symmetry and keep activation variance stable;
     * biases start at zero, the standard choice since they need no symmetry breaking.
     *
     * @param numInputs  Size of the flattened input vector. Must equal the product
     *                   of the previous layer's output dimensions (W*H*D).
     * @param numOutputs Number of output neurons (e.g. 10 for MNIST digit classes).
     *
     * @throws std::invalid_argument if numInputs or numOutputs is <= 0.
     */
    FCLayer(int numInputs, int numOutputs);

    /**
     * @brief Computes the layer's output: W * input + bias.
     *
     * @par Description
     * Flattens the input to a column vector, multiplies it by the weight matrix,
     * and adds the bias, producing one score per output neuron. The flattened
     * input and the original input shape are stashed: the flattened input is
     * needed to compute the weight gradient in backward, and the original shape
     * is needed to reshape the input gradient back for the previous layer.
     *
     * @param input The output of the previous layer. Any shape; it is flattened
     *              internally, so its element count (W*H*D) must equal numInputs.
     *
     * @return The output scores, shape {1, numOutputs, 1}.
     *
     * @throws std::invalid_argument if the flattened input size does not match
     *         numInputs (propagated from the matrix multiply).
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Computes parameter gradients and the input gradient.
     *
     * @par Description
     * From the incoming gradient (the loss gradient w.r.t. this layer's output),
     * computes three quantities:
     * - the weight gradient (outer product of the gradient with the stashed input),
     *   stored for the optimizer;
     * - the bias gradient (the incoming gradient itself), stored for the optimizer;
     * - the input gradient (the incoming gradient pushed back through the
     *   transposed weights), returned to the previous layer.
     *
     * The returned input gradient is reshaped to the original input shape so the
     * previous layer receives a gradient matching what it output.
     *
     * @param gradOutput The gradient of the loss w.r.t. this layer's output,
     *                   shape {1, numOutputs, 1}.
     *
     * @return The gradient of the loss w.r.t. this layer's input, reshaped to the
     *         original (pre-flatten) input shape.
     */
    Tensor backward(const Tensor& gradOutput) override;

    /**
     * @brief Applies one gradient-descent step to the weights and biases.
     *
     * @par Description
     * Updates each parameter in the direction that decreases the loss:
     * param = param - learningRate * gradient, applied element-wise to the weights
     * (using the weight gradient) and the bias (using the bias gradient). Must be
     * called after backward(), which computes those gradients.
     *
     * @param learningRate The step size. Larger values take bigger steps (faster
     *                     but risk overshooting); smaller values learn more slowly.
     */
    void updateWeights(float learningRate) override;

    /**
     * @brief Writes the weights and bias to a binary file stream.
     *
     * @par Description
     * Serializes the weight matrix (its shape followed by its data) and then the
     * bias (its shape followed by its data), in that order. load() must read them
     * back in the same order. Overrides the BaseLayer no-op since this layer has
     * learnable parameters.
     *
     * @param file An open binary output stream positioned where this layer writes.
     */
    void save(std::ofstream& file) const override;

    /**
     * @brief Reads the weights and bias from a binary file stream.
     *
     * @par Description
     * Deserializes the weights and bias written by save(), in the same order. Each
     * is read by first reading its shape (which gives the element count) and then
     * its data, so the loaded tensors reconstruct exactly. The layer must be
     * constructed with the same dimensions that were saved.
     *
     * @param file An open binary input stream positioned at this layer's data.
     */
    void load(std::ifstream& file) override;
};

#endif