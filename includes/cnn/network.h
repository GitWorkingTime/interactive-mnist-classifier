#ifndef NETWORK_H
#define NETWORK_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "base_layer.h"
#include "tensor.h"
#include <string>
#include <vector>

// ─── Class declaration ───────────────────────────────────────────────────────
/**
 * @brief A fixed-architecture CNN for MNIST digit classification.
 *
 * @par Description
 * Wraps the layer stack and provides high-level operations: running an input
 * forward, predicting a class, and loading/saving trained parameters. The
 * architecture is fixed at construction, so a loaded model always matches.
 *
 *   input        {28, 28, 1}
 *   Conv 3x3     {26, 26, 8}
 *   ReLU         {26, 26, 8}
 *   MaxPool 2x2  {13, 13, 8}
 *   FC: 13*13*8 = 1352 -> 10 classes
 */
class Network {
private:
    std::vector<BaseLayer*> network;

public:
    /** @brief Builds the fixed network architecture. */
    Network();

    /** @brief Deletes all owned layers. */
    ~Network();

    // Owns raw pointers — disable copying to prevent double-free.
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    /**
     * @brief Runs the input forward through every layer.
     * @param input The input tensor (e.g. an image).
     * @return The final raw scores (before softmax).
     */
    Tensor forward(const Tensor& input);

    /**
     * @brief Returns the class probability distribution for an input.
     * @param input The input tensor.
     * @return A {numClasses, 1, 1} tensor of probabilities (softmax of the scores).
     */
    Tensor predictProbabilities(const Tensor& input);

    /**
     * @brief Predicts the class index for an input.
     * @param input The input tensor.
     * @return The index of the highest-probability class.
     */
    int predict(const Tensor& input);

    /**
     * @brief Loads trained parameters from a binary file into the layers.
     * @param path Filesystem path to the saved model.
     * @throws std::runtime_error if the file cannot be opened.
     */
    void load(const std::string& path);

    /**
     * @brief Saves trained parameters to a binary file.
     * @param path Filesystem path to write the model to.
     * @throws std::runtime_error if the file cannot be opened.
     */
    void save(const std::string& path) const;
};

#endif