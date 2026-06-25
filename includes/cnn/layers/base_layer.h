#ifndef BASE_LAYER_H
#define BASE_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"
#include <fstream>

// ─── Class declarations ──────────────────────────────────────────────────────
/**
 * @brief Abstract base class defining the interface every network layer implements.
 *
 * @par Description
 * BaseLayer is the common contract for all layers in the network (ReLU, pooling,
 * convolution, fully connected, etc.). Each layer transforms a tensor on the way
 * forward and propagates a gradient on the way back. Because every layer shares
 * this interface, the network can have a collection of different types of layers
 * (e.g. std::vector<BaseLayer*>, the pointer is needed to specify the subclass)
 * and run a forward pass by chaining forward() calls, then a backward pass
 * by chaining backward() calls in reverse.
 *
 * This class cannot be instantiated directly. It's only needed for the interface.
 * Child layers inherit from it and provide their own forward() and backward().
 *
 * @note Layers that hold state needed for backpropagation (e.g. the input, a
 * mask, or argmax indices) stash it during forward() and read it during
 * backward(). Layers with learnable parameters (weights, filters, biases)
 * additionally manage those parameters and their gradients.
 *
 * @note updateWeights(), save(), and load() have empty default implementations so
 * that layers without learnable parameters (ReLU, pooling) inherit no-ops, while
 * parameterized layers (convolution, fully connected) override them. This lets the
 * network update or serialize every layer uniformly through a single loop.
 */
class BaseLayer {
public:
    /**
     * @brief Runs the layer's forward pass.
     *
     * @par Description
     * Transforms the input tensor into this layer's output, which becomes the
     * input to the next layer. Implementations typically stash whatever state
     * their backward pass will need (the input, a mask, argmax indices, etc.).
     *
     * @param input The output of the previous layer (or the network input for
     *              the first layer).
     *
     * @return This layer's output tensor.
     */
    virtual Tensor forward(const Tensor& input) = 0;

    /**
     * @brief Runs the layer's backward pass.
     *
     * @par Description
     * Receives the gradient of the loss with respect to this layer's output and
     * returns the gradient with respect to this layer's input, which is passed
     * to the previous layer. Layers with learnable parameters also use this step
     * to compute their parameter gradients (stored internally for the optimizer).
     *
     * @param gradOutput The gradient of the loss with respect to this layer's
     *                   output (flowing back from the next layer).
     *
     * @return The gradient of the loss with respect to this layer's input.
     */
    virtual Tensor backward(const Tensor& gradOutput) = 0;

    /**
     * @brief Virtual destructor.
     *
     * @par Description
     * Ensures that deleting a derived layer through a BaseLayer pointer runs the
     * derived class's destructor, so layer-owned resources (weights, filters)
     * are properly released.
     */
    virtual ~BaseLayer() = default;

    /**
     * @brief Applies one gradient-descent step to the layer's parameters.
     *
     * @par Description
     * Updates any learnable parameters using the gradients computed during
     * backward(). The default implementation does nothing, which is correct for
     * layers without parameters (ReLU, pooling). Parameterized layers override
     * this to update their weights/filters and biases.
     *
     * @param learningRate The step size for gradient descent.
     */
    virtual void updateWeights(float learningRate) {}

    /**
     * @brief Writes the layer's learnable parameters to a binary file stream.
     *
     * @par Description
     * Serializes any learnable parameters (and their shapes) to the given output
     * stream so the trained model can be saved and reloaded later. The default
     * implementation writes nothing, which is correct for layers without
     * parameters. Parameterized layers override this to write their weights/filters
     * and biases. Layers must write their data in the same order load() reads it.
     *
     * @param file An open binary output stream positioned where this layer's
     *             data should be written.
     */
    virtual void save(std::ofstream& file) const {}

    /**
     * @brief Reads the layer's learnable parameters from a binary file stream.
     *
     * @par Description
     * Deserializes any learnable parameters from the given input stream, reversing
     * what save() wrote. The default implementation reads nothing, matching layers
     * that have no parameters. Parameterized layers override this to read their
     * weights/filters and biases. The data must be read in the same order it was
     * written, and the network architecture must match the one that was saved.
     *
     * @param file An open binary input stream positioned at this layer's data.
     */
    virtual void load(std::ifstream& file) {}
};

#endif