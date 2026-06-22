// ─── Imports ─────────────────────────────────────────────────────────────────
#include "maxPooling_layer.h"

// ─── Class definitions ───────────────────────────────────────────────────────
MaxPoolingLayer::MaxPoolingLayer(int poolSize, int stride) {
    // Verify parameters
    if (poolSize <= 0) {
        throw std::invalid_argument("ERROR: Max Pooling Size is less than or equal to 0 - poolSize received: " + std::to_string(poolSize));
    }

    if (stride <= 0) {
        throw std::invalid_argument("ERROR: Stride is less than or equal to 0 - stride received: " + std::to_string(stride));
    }

    // Initialize private member variables
    this->poolSize = poolSize;
    this->stride = stride;
}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    // Stash input shape. Needed for backpropagation to build a gradient tensor of the same shape
    this->inputShape = input.getShape();

    // Perform maxpooling
    pooling::PoolResult buffer = pooling::maxPooling(input, poolSize, stride);

    // Stash argmax. Needed for backpropagation
    this->argmax = buffer.argmax;
    return buffer.pooled;
}

Tensor MaxPoolingLayer::backward(const Tensor& gradOutput) {
    // Initialize data with 0
    std::vector<float> buf(inputShape[0] * inputShape[1] * inputShape[2], 0.0f);
    const std::vector<float>& gradData = gradOutput.getData();

    if (gradData.size() != argmax.size()) {
        throw std::invalid_argument("ERROR: gradOutput size does not match stashed argmax");
    }

    // Map the gradData to the argmax index
    for (int i = 0; i < gradData.size(); ++i) {
        buf[argmax[i]] += gradData[i];
    }

    return Tensor(inputShape, buf);
}