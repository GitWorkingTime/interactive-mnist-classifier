#include "ReLU_layer.h"
#include "activations.h"
#include "base_layer.h"
#include "conv_layer.h"
#include "fc_layer.h"
#include "loss.h"
#include "maxPooling_layer.h"
#include "mnist.h"
#include "tensor.h"
#include <iostream>
#include <vector>

// Returns the index of the largest value in a {N,1,1} prediction tensor —
// i.e. the digit class the network is most confident about.
int argmax(const Tensor& prediction) {
    const std::vector<float>& data = prediction.getData();
    int best = 0;
    for (std::size_t i = 1; i < data.size(); ++i) {
        if (data[i] > data[best]) {
            best = (int)i;
        }
    }
    return best;
}

int main() {
    // ── Load the data ─────────────────────────────────────────────────────────
    std::vector<Tensor> images = mnist::loadImages("../data/train-images.idx3-ubyte");
    std::vector<int> labels = mnist::loadLabels("../data/train-labels.idx1-ubyte");

    std::cout << "Loaded " << images.size() << " images\n";

    // ── Build the network ─────────────────────────────────────────────────────
    // Architecture: Conv(8 filters, 3x3) -> ReLU -> MaxPool(2x2) -> FC(10)
    // Dimension trace:
    //   input        {28, 28, 1}
    //   Conv 3x3     {26, 26, 8}
    //   ReLU         {26, 26, 8}
    //   MaxPool 2x2  {13, 13, 8}
    //   FC input = 13*13*8 = 1352  ->  10 output classes
    std::vector<BaseLayer*> network = {
        new ConvLayer(8, 3, 1), // 8 filters, 3x3, input depth 1
        new ReLULayer(),
        new MaxPoolingLayer(2, 2),   // 2x2 window, stride 2
        new FCLayer(13 * 13 * 8, 10) // flattened pool output -> 10 classes
    };

    // ── Training settings ─────────────────────────────────────────────────────
    float learningRate = 0.01f;
    int epochs = 3;
    int trainSize = (int)images.size(); // use all images; lower this for quick tests

    // ── Training loop ─────────────────────────────────────────────────────────
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epochLoss = 0.0f;
        int correct = 0;

        for (int n = 0; n < trainSize; ++n) {
            // 1. Forward pass through every layer
            Tensor activation = images[n];
            for (BaseLayer* layer : network) {
                activation = layer->forward(activation);
            }

            // 2. Softmax to turn scores into probabilities, then loss
            Tensor prediction = activations::softMax(activation);
            Tensor target = mnist::oneHotEncodeLabels(labels[n]);
            epochLoss += loss::crossEntropy(prediction, target);

            // Track accuracy: did the most confident class match the label?
            if (argmax(prediction) == labels[n]) {
                correct++;
            }

            // 3. Initial gradient: prediction - target (softmax + cross-entropy)
            Tensor delta = loss::crossEntropyGradient(prediction, target);

            // 4. Backward pass through every layer in REVERSE order
            for (auto it = network.rbegin(); it != network.rend(); ++it) {
                delta = (*it)->backward(delta);
            }

            // 5. Update the weights of every parameterized layer
            for (BaseLayer* layer : network) {
                layer->updateWeights(learningRate);
            }

            // Progress print every 1000 images
            if ((n + 1) % 1000 == 0) {
                std::cout << "  epoch " << epoch
                          << " | image " << (n + 1) << "/" << trainSize
                          << " | running avg loss: " << epochLoss / (n + 1)
                          << " | running acc: " << (float)correct / (n + 1)
                          << "\n";
            }
        }

        std::cout << "=== Epoch " << epoch
                  << " complete | avg loss: " << epochLoss / trainSize
                  << " | accuracy: " << (float)correct / trainSize
                  << " ===\n";
    }

    // ── Clean up ──────────────────────────────────────────────────────────────
    for (BaseLayer* layer : network) {
        delete layer;
    }

    return 0;
}