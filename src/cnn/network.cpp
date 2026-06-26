#include "network.h"
#include "ReLU_layer.h"
#include "activations.h"
#include "conv_layer.h"
#include "fc_layer.h"
#include "maxPooling_layer.h"
#include <fstream>
#include <stdexcept>

// ─── Construction ─────────────────────────────────────────────────────────────
// Builds the fixed architecture:
//   input        {28, 28, 1}
//   Conv 3x3     {26, 26, 8}
//   ReLU         {26, 26, 8}
//   MaxPool 2x2  {13, 13, 8}
//   FC: 13*13*8 = 1352 -> 10 classes
Network::Network() {
    network.push_back(new ConvLayer(8, 3, 1)); // 8 filters, 3x3, input depth 1
    network.push_back(new ReLULayer());
    network.push_back(new MaxPoolingLayer(2, 2));    // 2x2 window, stride 2
    network.push_back(new FCLayer(13 * 13 * 8, 10)); // flattened pool -> 10 classes
}

Network::~Network() {
    for (BaseLayer* layer : network) {
        delete layer;
    }
}

// ─── Forward ──────────────────────────────────────────────────────────────────
Tensor Network::forward(const Tensor& input) {
    Tensor activation = input;
    for (BaseLayer* layer : network) {
        activation = layer->forward(activation);
    }
    return activation;
}

// ─── Prediction ───────────────────────────────────────────────────────────────
Tensor Network::predictProbabilities(const Tensor& input) {
    return activations::softMax(forward(input));
}

int Network::predict(const Tensor& input) {
    Tensor probs = predictProbabilities(input);
    const std::vector<float>& data = probs.getData();
    int best = 0;
    for (std::size_t i = 1; i < data.size(); ++i) {
        if (data[i] > data[best]) {
            best = (int)i;
        }
    }
    return best;
}

// ─── Save / Load ──────────────────────────────────────────────────────────────
void Network::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for loading: " + path);
    }
    for (BaseLayer* layer : network) {
        layer->load(file);
    }
}

void Network::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving: " + path);
    }
    for (BaseLayer* layer : network) {
        layer->save(file);
    }
}