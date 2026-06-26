#include "mnist.h"
#include "network.h"
#include "tensor.h"
#include <iostream>

int main() {
    // ── Load the trained model ────────────────────────────────────────────────
    Network net;           // builds the fixed architecture
    net.load("model.bin"); // fills it with trained parameters
    std::cout << "Model loaded.\n";

    // ── Load the TEST set (data the network never trained on) ─────────────────
    std::vector<Tensor> testImages = mnist::loadImages("../data/t10k-images.idx3-ubyte");
    std::vector<int> testLabels = mnist::loadLabels("../data/t10k-labels.idx1-ubyte");
    std::cout << "Loaded " << testImages.size() << " test images\n";

    // ── Evaluate accuracy on the test set ─────────────────────────────────────
    int correct = 0;
    for (std::size_t i = 0; i < testImages.size(); ++i) {
        int predicted = net.predict(testImages[i]);
        if (predicted == testLabels[i]) {
            correct++;
        }
    }

    float accuracy = (float)correct / testImages.size();
    std::cout << "Test accuracy: " << accuracy
              << " (" << correct << "/" << testImages.size() << ")\n";

    // ── Show a few individual predictions ─────────────────────────────────────
    std::cout << "\nSample predictions:\n";
    for (int i = 0; i < 5; ++i) {
        Tensor probs = net.predictProbabilities(testImages[i]);
        int predicted = net.predict(testImages[i]);

        std::cout << "  image " << i
                  << " | actual: " << testLabels[i]
                  << " | predicted: " << predicted
                  << " | confidence: " << probs.getData()[predicted] << "\n";
    }

    return 0;
}