// ─── Imports ─────────────────────────────────────────────────────────────────
#include "conv_layer.h"

// ─── Helpers ─────────────────────────────────────────────────────────────────
static void requireAtLeast(int value, int minimum, const std::string& name) {
    if (value < minimum) {
        throw std::invalid_argument("ERROR: " + name + " is less than " + std::to_string(minimum) + " - received: " + std::to_string(value));
    }
}

// ─── Class definitions ───────────────────────────────────────────────────────
ConvLayer::ConvLayer(int numFilters, int filterSize, int inputDepth, int stride, int padding) {
    // validation
    requireAtLeast(numFilters, 1, "numFilters");
    requireAtLeast(filterSize, 1, "filterSize");
    requireAtLeast(inputDepth, 1, "inputDepth");
    requireAtLeast(stride, 1, "stride");
    requireAtLeast(padding, 0, "padding");

    this->numFilters = numFilters;
    this->filterSize = filterSize;
    this->inputDepth = inputDepth;
    this->stride = stride;
    this->padding = padding;

    int fanIn = filterSize * filterSize * inputDepth;
    for (int f = 0; f < numFilters; ++f) {
        filters.push_back(weights::heInit({filterSize, filterSize, inputDepth}, fanIn));
        biases.push_back(0.0f); // biases start at zero, like FC
    }
}

Tensor ConvLayer::forward(const Tensor& input) {
    // Stash input
    this->input = input;

    // Store conv results
    std::vector<Tensor> buffer;
    buffer.reserve(numFilters);

    // Compute convolution and store results
    for (int i = 0; i < numFilters; ++i) {
        buffer.push_back(input.convolve(filters[i], biases[i], stride, padding));
    }

    // Stack the per-filter results into a {outW, outH, numFilters} tensor by
    // concatenating their flat data (depth is the outermost dimension).
    const std::vector<int>& sliceShape = buffer[0].getShape();
    std::vector<float> stacked;
    stacked.reserve(sliceShape[0] * sliceShape[1] * numFilters);

    for (int i = 0; i < numFilters; ++i) {
        const std::vector<float>& sliceData = buffer[i].getData();
        stacked.insert(stacked.end(), sliceData.begin(), sliceData.end());
    }

    return Tensor({sliceShape[0], sliceShape[1], numFilters}, stacked);
}

Tensor ConvLayer::backward(const Tensor& gradOutput) {
    filterGrads.clear();
    biasGrads.clear();
    filterGrads.reserve(numFilters);
    biasGrads.reserve(numFilters);

    const std::vector<int>& inShape = input.getShape();
    std::vector<Tensor> inputGradChannels;
    for (int d = 0; d < inputDepth; ++d)
        inputGradChannels.push_back(Tensor::zeros({inShape[0], inShape[1], 1}));

    for (int f = 0; f < numFilters; ++f) {
        Tensor gradSlice = gradOutput.getSlice(f);

        // bias gradient
        float biasGrad = 0.0f;
        for (float v : gradSlice.getData())
            biasGrad += v;
        biasGrads.push_back(biasGrad);

        // filter gradient — per input channel
        std::vector<Tensor> fgChannels;
        for (int d = 0; d < inputDepth; ++d) {
            Tensor inChannel = input.getSlice(d);
            Tensor fg = inChannel.convolve(gradSlice, 0.0f, 1, 0);
            fgChannels.push_back(fg);
        }
        filterGrads.push_back(Tensor::stackSlices(fgChannels));

        // input gradient — per input channel of the rotated filter (preserve depth)
        Tensor rotated = filters[f].rotate180();
        for (int d = 0; d < inputDepth; ++d) {
            Tensor rotChannel = rotated.getSlice(d);
            Tensor contribution = gradSlice.convolve(rotChannel, 0.0f, 1, filterSize - 1);
            inputGradChannels[d] = inputGradChannels[d].add(contribution);
        }
    }

    return Tensor::stackSlices(inputGradChannels);
}

void ConvLayer::updateWeights(float learningRate) {
    // Gradient descent on each filter: filter = filter - learningRate * filterGrad
    for (int f = 0; f < numFilters; ++f) {
        const std::vector<float>& fData = filters[f].getData();
        const std::vector<float>& fGrad = filterGrads[f].getData();

        std::vector<float> newFilter(fData.size());
        for (std::size_t i = 0; i < fData.size(); ++i) {
            newFilter[i] = fData[i] - learningRate * fGrad[i];
        }
        filters[f] = Tensor(filters[f].getShape(), newFilter);

        // and the bias: bias = bias - learningRate * biasGrad
        biases[f] -= learningRate * biasGrads[f];
    }
}

void ConvLayer::save(std::ofstream& file) const {
    // Write each filter: shape then data
    for (const Tensor& filter : filters) {
        const std::vector<int>& fShape = filter.getShape();
        file.write(reinterpret_cast<const char*>(fShape.data()), 3 * sizeof(int));
        const std::vector<float>& fData = filter.getData();
        file.write(reinterpret_cast<const char*>(fData.data()), fData.size() * sizeof(float));
    }
    // Write biases (numFilters floats)
    file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

void ConvLayer::load(std::ifstream& file) {
    // Read each filter (we know there are numFilters of them)
    for (int f = 0; f < numFilters; ++f) {
        std::vector<int> fShape(3);
        file.read(reinterpret_cast<char*>(fShape.data()), 3 * sizeof(int));
        int fSize = fShape[0] * fShape[1] * fShape[2];
        std::vector<float> fData(fSize);
        file.read(reinterpret_cast<char*>(fData.data()), fSize * sizeof(float));
        filters[f] = Tensor(fShape, fData);
    }
    // Read biases
    file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
}