// ─── Imports ─────────────────────────────────────────────────────────────────
#include "fc_layer.h"

// ─── Class definitions ───────────────────────────────────────────────────────
FCLayer::FCLayer(int numInputs, int numOutputs) {
    // Validate parameters
    if (numInputs <= 0) {
        throw std::invalid_argument("ERROR: numInputs is less than or equal to 0 - numInputs received: " + std::to_string(numInputs));
    }

    if (numOutputs <= 0) {
        throw std::invalid_argument("ERROR: numOutputs is less than or equal to 0 - numOutputs received: " + std::to_string(numOutputs));
    }

    this->numInputs = numInputs;
    this->numOutputs = numOutputs;

    this->weights = weights::heInit({numInputs, numOutputs, 1}, numInputs);
    bias = Tensor::zeros({1, numOutputs, 1});
}

Tensor FCLayer::forward(const Tensor& input) {
    // Stash information about the original input
    inputShape = input.getShape();
    this->input = input.flatten().transpose();

    Tensor out = weights.multiply(this->input);
    return out.add(bias);
}

Tensor FCLayer::backward(const Tensor& gradOutput) {
    // 1. Bias gradient = the delta directly
    biasGrad = gradOutput;

    // 2. Weight gradient = delta (outer) input  →  {numInputs, numOutputs, 1}
    weightGrad = gradOutput.multiply(this->input.transpose());

    // 3. Input gradient = Wᵀ · delta  →  {1, numInputs, 1}
    Tensor dInFlat = weights.transpose().multiply(gradOutput);

    // Reshape the flat input gradient back to the original input shape
    Tensor dIn(inputShape, dInFlat.getData());
    return dIn;
}

void FCLayer::updateWeights(float learningRate) {
    // Gradient descent: param = param - learningRate * gradient
    const std::vector<float>& wData = weights.getData();
    const std::vector<float>& wGrad = weightGrad.getData();
    std::vector<float> newWeights(wData.size());
    for (std::size_t i = 0; i < wData.size(); ++i) {
        newWeights[i] = wData[i] - learningRate * wGrad[i];
    }
    weights = Tensor(weights.getShape(), newWeights);

    const std::vector<float>& bData = bias.getData();
    const std::vector<float>& bGrad = biasGrad.getData();
    std::vector<float> newBias(bData.size());
    for (std::size_t i = 0; i < bData.size(); ++i) {
        newBias[i] = bData[i] - learningRate * bGrad[i];
    }
    bias = Tensor(bias.getShape(), newBias);
}